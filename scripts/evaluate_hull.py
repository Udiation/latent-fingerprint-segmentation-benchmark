#!/usr/bin/env python3
"""Evaluate segmentation masks against SD302g minutiae-region lower bounds.

This evaluator reports exactly four complementary quantities:

``MC``
    Fraction of expert minutiae covered by the prediction.
``HR``
    Fraction of convex-hull pixels covered by the prediction.
``PAR``
    Predicted foreground area divided by hull area. Values above one are valid.
``OHF``
    Predicted foreground outside the hull divided by all predicted foreground.
    An empty prediction has OHF zero by the empty-set convention.

MC and HR are lower-bound coverage measures; they must be presented together
with PAR. OHF describes where predicted area lies. None measures a complete
foreground outline, and the output deliberately contains no single ranking.
No tolerance radius, dilation, erosion, or other selectable geometry is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


REQUIRED_MANIFEST_COLUMNS = {
    "image_id",
    "hull_path",
    "minutiae_path",
    "subject_id",
    "split",
    "group",
    "device",
    "capture",
    "width",
    "height",
    "ppi",
}
PER_IMAGE_COLUMNS = (
    "method",
    "display_name",
    "image_id",
    "subject_id",
    "split",
    "group",
    "device",
    "capture",
    "verification_status",
    "n_minutiae",
    "minutiae_covered",
    "hull_pixels",
    "hull_covered",
    "predicted_pixels",
    "outside_hull_pixels",
    "mc",
    "hr",
    "par",
    "ohf",
    "missing",
)
SUMMARY_COLUMNS = (
    "method",
    "display_name",
    "split",
    "verification_status",
    "aggregate",
    "value",
    "n_images",
    "n_subjects",
    "n_minutiae",
    "n_missing",
    "bootstrap_seed",
    "bootstrap_replicates",
    "bootstrap_unit",
    "bootstrap_status",
    "mc_mean",
    "mc_std",
    "mc_ci_low",
    "mc_ci_high",
    "mc_pooled",
    "hr_mean",
    "hr_std",
    "hr_ci_low",
    "hr_ci_high",
    "hr_pooled",
    "par_mean",
    "par_std",
    "par_ci_low",
    "par_ci_high",
    "par_pooled",
    "ohf_mean",
    "ohf_std",
    "ohf_ci_low",
    "ohf_ci_high",
    "ohf_pooled",
)
REFERENCE_METHODS = (
    ("all_background", "All background"),
    ("all_foreground", "All foreground"),
    ("hull_as_prediction", "Hull as prediction"),
)


class ProtocolError(ValueError):
    """Input or output violates the preregistered lower-bound protocol."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _outside_repo(path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    try:
        path.relative_to(repo)
    except ValueError:
        return
    raise ValueError(f"private evaluation output must be outside the repository: {path}")


def load_manifest(path: Path, split: str) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = REQUIRED_MANIFEST_COLUMNS - fields
        if missing:
            raise ProtocolError(f"manifest is missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader if row["split"] == split]
    if not rows:
        raise ProtocolError(f"manifest contains no rows for split {split!r}")
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ProtocolError("manifest image_id values are not unique")
    return rows


def load_provenance(manifest_path: Path) -> Mapping[str, object]:
    path = manifest_path.parent / "provenance.json"
    if not path.is_file():
        raise ProtocolError(f"missing builder provenance: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ProtocolError("builder provenance is not an object")
    if payload.get("schema") != "cureblock.sd302g.build-provenance.v1":
        raise ProtocolError("unsupported builder provenance schema")
    if payload.get("protocol") != "sd302g-minutiae-lower-bound-v2":
        raise ProtocolError("builder provenance names the wrong protocol")
    if payload.get("manifest_sha256") != _sha256_file(manifest_path):
        raise ProtocolError("manifest hash disagrees with builder provenance")
    status = payload.get("verification_status")
    if status not in {
        "REFERENCE_VERIFIED",
        "UNVERIFIED_COORDINATES",
        "SYNTHETIC_TEST_ONLY",
    }:
        raise ProtocolError(f"unsupported coordinate verification status {status!r}")
    bootstrap = payload.get("bootstrap")
    if (
        not isinstance(bootstrap, dict)
        or not isinstance(bootstrap.get("seed"), int)
        or bootstrap.get("replicates") != 1000
        or bootstrap.get("unit") != "subject"
        or bootstrap.get("interval") != "percentile_95"
        or bootstrap.get("paired") is not True
    ):
        raise ProtocolError("builder provenance lacks the frozen bootstrap plan")
    return payload


def _manifest_artifact(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ProtocolError(
            f"private manifest artifacts must be relative to the manifest: {value!r}"
        )
    if ".." in path.parts:
        raise ProtocolError(f"manifest artifact escapes its directory: {value!r}")
    return manifest_path.parent / path


def _read_binary_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ProtocolError(f"could not read mask: {path}")
    values = set(int(value) for value in np.unique(image))
    if not values.issubset({0, 255}):
        raise ProtocolError(f"mask is not binary 0/255: {path} has values {sorted(values)[:10]}")
    return image > 0


def _read_points(path: Path, width: int, height: int) -> List[Tuple[int, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not {"x_px", "y_px"}.issubset(fields):
            raise ProtocolError(f"minutiae file lacks x_px/y_px columns: {path}")
        points: List[Tuple[int, int]] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                x, y = int(row["x_px"]), int(row["y_px"])
            except (TypeError, ValueError) as exc:
                raise ProtocolError(
                    f"invalid minutia coordinate at {path}:{row_number}"
                ) from exc
            if not (0 <= x < width and 0 <= y < height):
                raise ProtocolError(
                    f"minutia {(x, y)} at {path}:{row_number} is outside "
                    f"{(width, height)}"
                )
            points.append((x, y))
    if not points:
        raise ProtocolError(f"minutiae file is empty: {path}")
    return points


def _prediction_path(template: str, row: Mapping[str, str]) -> Path:
    try:
        rendered = template.format(**row)
    except KeyError as exc:
        raise ProtocolError(f"prediction template uses unknown manifest field {exc}") from exc
    return Path(rendered)


def _load_prediction(
    path: Path,
    width: int,
    height: int,
    native_ppi: float,
    pred_ppi: float,
) -> Tuple[np.ndarray, bool]:
    if not path.exists():
        return np.zeros((height, width), dtype=bool), True
    prediction = _read_binary_mask(path)
    if pred_ppi > 0:
        expected = (
            max(1, round(height * pred_ppi / native_ppi)),
            max(1, round(width * pred_ppi / native_ppi)),
        )
        if prediction.shape != expected:
            raise ProtocolError(
                f"prediction {path} shape {prediction.shape} != expected {expected} "
                f"at {pred_ppi:g} ppi"
            )
        if prediction.shape != (height, width):
            prediction = cv2.resize(
                prediction.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
    elif prediction.shape != (height, width):
        raise ProtocolError(
            f"native-resolution prediction {path} shape {prediction.shape} != "
            f"{(height, width)}; provide --pred-ppi only when appropriate"
        )
    return prediction, False


def score_lower_bound(
    prediction: np.ndarray,
    hull: np.ndarray,
    points: Sequence[Tuple[int, int]],
) -> Dict[str, object]:
    """Return the four frozen metrics plus transparent integer numerators."""
    pred = np.asarray(prediction, dtype=bool)
    region = np.asarray(hull, dtype=bool)
    if pred.shape != region.shape:
        raise ProtocolError(f"prediction shape {pred.shape} != hull shape {region.shape}")
    hull_pixels = int(np.count_nonzero(region))
    if hull_pixels == 0:
        raise ProtocolError("minutiae hull is empty")
    if not points:
        raise ProtocolError("expert minutiae list is empty")
    covered_points = sum(bool(pred[y, x]) for x, y in points)
    hull_covered = int(np.count_nonzero(pred & region))
    predicted_pixels = int(np.count_nonzero(pred))
    outside_hull = int(np.count_nonzero(pred & ~region))
    return {
        "n_minutiae": len(points),
        "minutiae_covered": covered_points,
        "hull_pixels": hull_pixels,
        "hull_covered": hull_covered,
        "predicted_pixels": predicted_pixels,
        "outside_hull_pixels": outside_hull,
        "mc": covered_points / len(points),
        "hr": hull_covered / hull_pixels,
        "par": predicted_pixels / hull_pixels,
        "ohf": outside_hull / predicted_pixels if predicted_pixels else 0.0,
    }


def _per_image_row(
    method: str,
    display_name: str,
    manifest_row: Mapping[str, str],
    metrics: Mapping[str, object],
    missing: bool,
    verification_status: str,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "method": method,
        "display_name": display_name,
        "image_id": manifest_row["image_id"],
        "subject_id": manifest_row["subject_id"],
        "split": manifest_row["split"],
        "group": manifest_row["group"],
        "device": manifest_row["device"],
        "capture": manifest_row["capture"],
        "verification_status": verification_status,
        "missing": int(missing),
    }
    row.update(metrics)
    return row


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _bootstrap_intervals(
    rows: Sequence[Mapping[str, object]],
    metrics: Sequence[str],
    seed: int,
    replicates: int,
    aggregate: str,
    value: str,
) -> Tuple[Dict[str, Tuple[float, float]], str]:
    """Subject-cluster bootstrap of image-macro means.

    The derived stratum seed omits method identity. Consequently methods with
    identical subject/image coverage receive identical subject draws, which is
    required for later paired contrasts.
    """
    subjects = sorted({str(row["subject_id"]) for row in rows})
    if len(subjects) < 2:
        return ({metric: (float("nan"), float("nan")) for metric in metrics},
                "insufficient_subjects")
    by_subject = {
        subject: [row for row in rows if str(row["subject_id"]) == subject]
        for subject in subjects
    }
    counts = np.asarray([len(by_subject[subject]) for subject in subjects], dtype=np.int64)
    key = f"{seed}|{aggregate}|{value}".encode("utf-8")
    stratum_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    rng = np.random.default_rng(stratum_seed)
    draws = rng.integers(0, len(subjects), size=(replicates, len(subjects)))
    sampled_counts = counts[draws].sum(axis=1)
    intervals: Dict[str, Tuple[float, float]] = {}
    for metric in metrics:
        sums = np.asarray(
            [
                sum(float(row[metric]) for row in by_subject[subject])
                for subject in subjects
            ],
            dtype=np.float64,
        )
        means = sums[draws].sum(axis=1) / sampled_counts
        low, high = np.percentile(means, (2.5, 97.5))
        intervals[metric] = (float(low), float(high))
    return intervals, "ok"


def _summary_row(
    rows: Sequence[Mapping[str, object]],
    aggregate: str,
    value: str,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> Dict[str, object]:
    if not rows:
        raise ProtocolError("cannot summarise an empty aggregate")
    metrics = ("mc", "hr", "par", "ohf")
    statuses = {str(row["verification_status"]) for row in rows}
    if len(statuses) != 1:
        raise ProtocolError("aggregate mixes coordinate verification statuses")
    intervals, bootstrap_status = _bootstrap_intervals(
        rows,
        metrics,
        bootstrap_seed,
        bootstrap_replicates,
        aggregate,
        value,
    )
    result: Dict[str, object] = {
        "method": rows[0]["method"],
        "display_name": rows[0]["display_name"],
        "split": rows[0]["split"],
        "verification_status": statuses.pop(),
        "aggregate": aggregate,
        "value": value,
        "n_images": len(rows),
        "n_subjects": len({str(row["subject_id"]) for row in rows}),
        "n_minutiae": sum(int(row["n_minutiae"]) for row in rows),
        "n_missing": sum(int(row["missing"]) for row in rows),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_unit": "subject",
        "bootstrap_status": bootstrap_status,
    }
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = float(values.std(ddof=0))
        result[f"{metric}_ci_low"] = intervals[metric][0]
        result[f"{metric}_ci_high"] = intervals[metric][1]
    result["mc_pooled"] = _ratio(
        sum(int(row["minutiae_covered"]) for row in rows),
        sum(int(row["n_minutiae"]) for row in rows),
    )
    result["hr_pooled"] = _ratio(
        sum(int(row["hull_covered"]) for row in rows),
        sum(int(row["hull_pixels"]) for row in rows),
    )
    result["par_pooled"] = _ratio(
        sum(int(row["predicted_pixels"]) for row in rows),
        sum(int(row["hull_pixels"]) for row in rows),
    )
    result["ohf_pooled"] = _ratio(
        sum(int(row["outside_hull_pixels"]) for row in rows),
        sum(int(row["predicted_pixels"]) for row in rows),
    )
    return result


def aggregate_rows(
    per_image: Sequence[Mapping[str, object]],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> List[Dict[str, object]]:
    """Aggregate every method overall and separately by release metadata."""
    by_method: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in per_image:
        by_method[str(row["method"])].append(row)
    summary: List[Dict[str, object]] = []
    for method_rows in by_method.values():
        summary.append(
            _summary_row(
                method_rows,
                "overall",
                "all",
                bootstrap_seed,
                bootstrap_replicates,
            )
        )
        for column in ("group", "device", "capture"):
            values = sorted({str(row[column]) for row in method_rows})
            for value in values:
                selected = [row for row in method_rows if str(row[column]) == value]
                summary.append(
                    _summary_row(
                        selected,
                        column,
                        value,
                        bootstrap_seed,
                        bootstrap_replicates,
                    )
                )
    return summary


def evaluate(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    manifest_path = Path(args.manifest).resolve()
    out = Path(args.out).resolve()
    _outside_repo(manifest_path)
    _outside_repo(out)
    provenance = load_provenance(manifest_path)
    verification_status = str(provenance["verification_status"])
    if verification_status != "SYNTHETIC_TEST_ONLY" and args.split != "test":
        raise ProtocolError(
            "production lower-bound scoring is restricted to the frozen test split"
        )
    if verification_status == "UNVERIFIED_COORDINATES":
        print(
            "WARNING: coordinates have not passed the independent private "
            "reference oracle; results are diagnostic, not paper-ready",
            file=sys.stderr,
        )
    bootstrap = provenance["bootstrap"]
    if not isinstance(bootstrap, dict):
        raise ProtocolError("builder provenance bootstrap plan is not an object")
    bootstrap_seed = int(bootstrap["seed"])
    bootstrap_replicates = int(bootstrap["replicates"])
    rows = load_manifest(manifest_path, args.split)
    per_image: List[Dict[str, object]] = []
    missing_paths: List[Path] = []
    for row in rows:
        width, height = int(row["width"]), int(row["height"])
        if width <= 0 or height <= 0:
            raise ProtocolError(f"invalid dimensions for {row['image_id']!r}")
        native_ppi = float(row["ppi"])
        if not math.isfinite(native_ppi) or native_ppi <= 0:
            raise ProtocolError(f"invalid PPI for {row['image_id']!r}")
        hull = _read_binary_mask(
            _manifest_artifact(manifest_path, row["hull_path"])
        )
        if hull.shape != (height, width):
            raise ProtocolError(
                f"hull shape {hull.shape} != manifest {(height, width)} for "
                f"{row['image_id']!r}"
            )
        points = _read_points(
            _manifest_artifact(manifest_path, row["minutiae_path"]), width, height
        )
        prediction_path = _prediction_path(args.pred_template, row)
        prediction, missing = _load_prediction(
            prediction_path, width, height, native_ppi, args.pred_ppi
        )
        if missing:
            missing_paths.append(prediction_path)

        predictions = [
            (args.method, args.display_name, prediction, missing),
            (REFERENCE_METHODS[0][0], REFERENCE_METHODS[0][1], np.zeros_like(hull), False),
            (REFERENCE_METHODS[1][0], REFERENCE_METHODS[1][1], np.ones_like(hull), False),
            (REFERENCE_METHODS[2][0], REFERENCE_METHODS[2][1], hull, False),
        ]
        for method, display_name, current, current_missing in predictions:
            metrics = score_lower_bound(current, hull, points)
            if method == "hull_as_prediction" and (
                metrics["mc"] != 1.0 or metrics["hr"] != 1.0
            ):
                raise ProtocolError(
                    f"hull gate failed for {row['image_id']!r}: "
                    f"MC={metrics['mc']}, HR={metrics['hr']}"
                )
            per_image.append(
                _per_image_row(
                    method,
                    display_name,
                    row,
                    metrics,
                    current_missing,
                    verification_status,
                )
            )
    if missing_paths:
        examples = ", ".join(str(path) for path in missing_paths[:5])
        print(
            f"WARNING: {len(missing_paths)} prediction(s) missing; scored as "
            f"all-background. Examples: {examples}",
            file=sys.stderr,
        )
    summary = aggregate_rows(
        per_image, bootstrap_seed, bootstrap_replicates
    )
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / f"per_image_{args.method}_{args.split}.csv", per_image, PER_IMAGE_COLUMNS)
    _write_csv(out / f"summary_{args.method}_{args.split}.csv", summary, SUMMARY_COLUMNS)
    for row in summary:
        if row["aggregate"] == "overall":
            print(
                f"{row['display_name']}: n={row['n_images']} missing={row['n_missing']} "
                f"MC={100 * float(row['mc_mean']):.2f}% "
                f"HR={100 * float(row['hr_mean']):.2f}% "
                f"PAR={float(row['par_mean']):.3f} "
                f"OHF={100 * float(row['ohf_mean']):.2f}%"
            )
    return per_image, summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pred-template", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--pred-ppi",
        type=float,
        default=0.0,
        help="prediction-mask PPI; omit/zero when predictions are native resolution",
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", args.method):
        parser.error("--method must be a filesystem-safe lowercase key")
    if args.method in {key for key, _ in REFERENCE_METHODS}:
        parser.error("--method collides with a reserved reference row")
    if not math.isfinite(args.pred_ppi) or args.pred_ppi < 0:
        parser.error("--pred-ppi must be finite and non-negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
