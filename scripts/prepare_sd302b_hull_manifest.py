#!/usr/bin/env python3
"""Build an authenticated SD302b-to-SD302g lower-bound mapping.

This is a mapping and equivalence validator, not an inference or scoring tool.
It proves which SD302b archive images correspond to the independently verified
SD302g embedded exemplar images and writes a ground-truth-blind inference
manifest.  The observed archive transform is recorded explicitly: native
500-ppi records compare directly, while native 1000-ppi records compare after
Pillow LANCZOS resizing to the decoded SD302b canvas.  The latter is an
exhaustive pixel-equivalence observation, not an authenticated archive recipe.

All outputs contain restricted biometric identifiers and must remain outside
the source repository.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from PIL import Image


EXPECTED_MANIFEST_SHA256 = (
    "a6d1adc3d5536dac472414ba7c9cea4753eab5cbcbf27558ff9785ce54d2de0c"
)
EXPECTED_SPLIT_SHA256 = (
    "1d3af693767192f03c16ed8ac79e93029ee130f03cc68a09e755acdf8b018be8"
)
EXPECTED_ARCHIVE_SHA256 = (
    "a0cc9906fad84a98cca7fdb4e1634a7c11b22d9a0f29a2732a4ed8409847ca82"
)
MAPPING_COLUMNS = (
    "image_id",
    "subject_id",
    "split",
    "group",
    "device",
    "source_capture",
    "archive_capture",
    "report_capture",
    "finger_position",
    "source_ppi",
    "actual_ppi",
    "sd302g_image_path",
    "sd302b_image_path",
    "sd302g_width",
    "sd302g_height",
    "sd302b_width",
    "sd302b_height",
    "native_dimensions_equal",
    "native_pixels_equal",
    "canonical_transform",
    "canonical_pixels_equal",
    "sd302g_image_sha256",
    "sd302b_image_sha256",
)
INFERENCE_COLUMNS = (
    "image_id",
    "subject_id",
    "split",
    "subset",
    "group",
    "device",
    "capture",
    "capture_label",
    "finger_position",
    "image_path",
    "mask_path",
    "width",
    "height",
    "ppi",
    "source_ppi",
)
REQUIRED_G_COLUMNS = {
    "image_id",
    "image_path",
    "subject_id",
    "split",
    "group",
    "device",
    "capture",
    "finger_position",
    "width",
    "height",
    "ppi",
}


class MappingError(ValueError):
    """A source identity or exact-equivalence gate failed."""


@dataclass(frozen=True)
class UniverseExpectations:
    records: int
    test_records: int
    native_exact: int
    split_seed: int
    subject_split_counts: Mapping[str, int]
    split_counts: Mapping[str, int]
    capture_counts: Mapping[str, int]
    test_capture_counts: Mapping[str, int]
    device_counts: Mapping[str, int]
    test_device_counts: Mapping[str, int]


PRODUCTION_EXPECTATIONS = UniverseExpectations(
    records=2_194,
    test_records=439,
    native_exact=27,
    split_seed=20260908,
    subject_split_counts={"train": 136, "val": 24, "test": 40},
    split_counts={"train": 1_496, "val": 259, "test": 439},
    capture_counts={"roll": 2_146, "slap": 48},
    test_capture_counts={"roll": 430, "slap": 9},
    device_counts={"R": 21, "S": 27, "U": 147, "V": 1_999},
    test_device_counts={"R": 6, "S": 3, "U": 30, "V": 400},
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def outside_repository(path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    try:
        path.resolve().relative_to(repository)
    except ValueError:
        return
    raise MappingError(f"restricted output must be outside repository: {path}")


def load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MappingError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise MappingError(f"{label} must be a JSON object")
    return value


def load_csv(path: Path, label: str) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise MappingError(f"{label} is not readable: {path}") from exc
    if not columns:
        raise MappingError(f"{label} has no header")
    return columns, rows


def confined_file(root: Path, value: str, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise MappingError(f"{label} is not a confined relative path: {value!r}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise MappingError(f"{label} escapes its artifact root") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise MappingError(f"{label} is missing or a symlink: {resolved}")
    return resolved


def _decoded(path: Path) -> tuple[np.ndarray, tuple[float, float] | None]:
    try:
        with Image.open(path) as image:
            image.load()
            pixels = np.asarray(image)
            raw_dpi = image.info.get("dpi")
    except (OSError, ValueError) as exc:
        raise MappingError(f"could not decode image: {path}") from exc
    if pixels.ndim != 2 or pixels.dtype != np.uint8:
        raise MappingError(
            f"expected one-channel uint8 PNG, got {pixels.shape}/{pixels.dtype}: {path}"
        )
    dpi: tuple[float, float] | None = None
    if isinstance(raw_dpi, (tuple, list)) and len(raw_dpi) >= 2:
        dpi = (float(raw_dpi[0]), float(raw_dpi[1]))
    return np.array(pixels, copy=True), dpi


def _archive_path(root: Path, row: Mapping[str, str]) -> tuple[Path, str]:
    capture = row["capture"]
    if capture == "roll":
        archive_capture = "roll"
    elif capture == "slap":
        archive_capture = "slap-segmented"
    else:
        raise MappingError(f"unsupported SD302g baseline capture: {capture!r}")
    try:
        source_ppi = int(round(float(row["ppi"])))
    except (TypeError, ValueError) as exc:
        raise MappingError(f"invalid source PPI for {row.get('image_id')!r}") from exc
    if source_ppi not in {500, 1000}:
        raise MappingError(f"unsupported source PPI {source_ppi}")
    relative = (
        Path(row["device"])
        / str(source_ppi)
        / archive_capture
        / "png"
        / f"{row['image_id']}.png"
    )
    return confined_file(root, str(relative), "SD302b image"), archive_capture


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _counter(rows: Sequence[Mapping[str, object]], key: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def _require_count(observed: Mapping[str, int], expected: Mapping[str, int], label: str) -> None:
    if dict(observed) != dict(expected):
        raise MappingError(f"{label} differs: observed {dict(observed)}, expected {dict(expected)}")


def build(
    verified_manifest: Path,
    splits_path: Path,
    sd302b_root: Path,
    output: Path,
    *,
    sd302b_archive: Path | None = None,
    expectations: UniverseExpectations = PRODUCTION_EXPECTATIONS,
    expected_manifest_sha256: str | None = EXPECTED_MANIFEST_SHA256,
    expected_split_sha256: str | None = EXPECTED_SPLIT_SHA256,
    expected_archive_sha256: str | None = EXPECTED_ARCHIVE_SHA256,
) -> Mapping[str, Any]:
    """Validate every baseline pair and atomically publish the private bundle."""
    verified_manifest = verified_manifest.resolve()
    splits_path = splits_path.resolve()
    sd302b_root = sd302b_root.resolve()
    output = output.resolve()
    outside_repository(output)
    if output.exists():
        raise MappingError(f"refusing to overwrite mapping bundle: {output}")
    if not verified_manifest.is_file() or not splits_path.is_file():
        raise MappingError("verified manifest or frozen split is missing")
    if not sd302b_root.is_dir() or sd302b_root.is_symlink():
        raise MappingError("SD302b baseline root is missing or a symlink")

    manifest_hash = sha256_file(verified_manifest)
    split_hash = sha256_file(splits_path)
    if expected_manifest_sha256 and manifest_hash != expected_manifest_sha256:
        raise MappingError("verified SD302g manifest identity changed")
    if expected_split_sha256 and split_hash != expected_split_sha256:
        raise MappingError("frozen split identity changed")

    provenance_path = verified_manifest.parent / "provenance.json"
    provenance = load_json(provenance_path, "verified SD302g provenance")
    if (
        provenance.get("schema") != "cureblock.sd302g.build-provenance.v1"
        or provenance.get("verification_status") != "REFERENCE_VERIFIED"
        or provenance.get("manifest_sha256") != manifest_hash
        or provenance.get("split_sha256") != split_hash
    ):
        raise MappingError("SD302g provenance does not bind the verified manifest/split")

    split_payload = load_json(splits_path, "frozen split")
    split_map = split_payload.get("splits")
    if not isinstance(split_map, dict):
        raise MappingError("frozen split lacks its subject mapping")
    split_counts = Counter(str(value) for value in split_map.values())
    if (
        split_payload.get("seed") != expectations.split_seed
        or split_counts != Counter(expectations.subject_split_counts)
    ):
        raise MappingError("frozen subject split policy changed")

    columns, source_rows = load_csv(verified_manifest, "verified SD302g manifest")
    missing = REQUIRED_G_COLUMNS - set(columns)
    if missing:
        raise MappingError(f"verified manifest lacks columns: {sorted(missing)}")
    rows = [row for row in source_rows if row["group"] == "baseline"]
    if len(rows) != expectations.records:
        raise MappingError(
            f"baseline mapping has {len(rows)} rows; expected {expectations.records}"
        )
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise MappingError("baseline image identities are not unique")

    archive_identity: Dict[str, object] | None = None
    if sd302b_archive is not None:
        archive = sd302b_archive.resolve()
        if not archive.is_file() or archive.is_symlink():
            raise MappingError("SD302b archive is missing or a symlink")
        archive_hash = sha256_file(archive)
        if expected_archive_sha256 and archive_hash != expected_archive_sha256:
            raise MappingError("SD302b archive identity changed")
        archive_identity = {
            "path": str(archive),
            "sha256": archive_hash,
            "bytes": archive.stat().st_size,
        }

    mapped: list[Dict[str, object]] = []
    inference: list[Dict[str, object]] = []
    for row in sorted(rows, key=lambda item: item["image_id"]):
        subject = row["subject_id"]
        if split_map.get(subject) != row["split"]:
            raise MappingError(
                f"verified manifest/frozen split disagreement for subject {subject}"
            )
        g_path = confined_file(verified_manifest.parent, row["image_path"], "SD302g image")
        b_path, archive_capture = _archive_path(sd302b_root, row)
        g_pixels, _ = _decoded(g_path)
        b_pixels, b_dpi = _decoded(b_path)
        if b_dpi is None or any(not 495.0 <= value <= 505.0 for value in b_dpi):
            raise MappingError(f"SD302b image lacks an approximately 500-ppi PNG tag: {b_path}")
        expected_g_shape = (int(row["height"]), int(row["width"]))
        if g_pixels.shape != expected_g_shape:
            raise MappingError(
                f"verified manifest dimensions disagree with decoded SD302g image: {g_path}"
            )

        native_dimensions_equal = g_pixels.shape == b_pixels.shape
        native_pixels_equal = bool(
            native_dimensions_equal and np.array_equal(g_pixels, b_pixels)
        )
        source_ppi = int(round(float(row["ppi"])))
        if source_ppi == 500:
            canonical = g_pixels
            transform = "identity_native_500"
        else:
            canonical = np.asarray(
                Image.fromarray(g_pixels, mode="L").resize(
                    (b_pixels.shape[1], b_pixels.shape[0]),
                    resample=Image.Resampling.LANCZOS,
                )
            )
            transform = "observed_PIL.Image.Resampling.LANCZOS_to_archive_canvas"
        canonical_equal = bool(np.array_equal(canonical, b_pixels))
        if not canonical_equal:
            raise MappingError(f"canonical archive-500 pixels differ for {row['image_id']}")

        report_capture = (
            "Rolled" if row["capture"] == "roll" else "Plain (segmented slap)"
        )
        record: Dict[str, object] = {
            "image_id": row["image_id"],
            "subject_id": subject,
            "split": row["split"],
            "group": "baseline",
            "device": row["device"],
            "source_capture": row["capture"],
            "archive_capture": archive_capture,
            "report_capture": report_capture,
            "finger_position": row["finger_position"],
            "source_ppi": source_ppi,
            "actual_ppi": 500,
            "sd302g_image_path": str(g_path),
            "sd302b_image_path": str(b_path),
            "sd302g_width": g_pixels.shape[1],
            "sd302g_height": g_pixels.shape[0],
            "sd302b_width": b_pixels.shape[1],
            "sd302b_height": b_pixels.shape[0],
            "native_dimensions_equal": native_dimensions_equal,
            "native_pixels_equal": native_pixels_equal,
            "canonical_transform": transform,
            "canonical_pixels_equal": canonical_equal,
            "sd302g_image_sha256": sha256_file(g_path),
            "sd302b_image_sha256": sha256_file(b_path),
        }
        mapped.append(record)
        inference.append(
            {
                "image_id": row["image_id"],
                "subject_id": subject,
                "split": row["split"],
                "subset": "finger_exemplar_sd302b",
                "group": "baseline",
                "device": row["device"],
                "capture": "roll" if row["capture"] == "roll" else "slap_segmented",
                "capture_label": report_capture,
                "finger_position": row["finger_position"],
                "image_path": str(b_path),
                "mask_path": "",
                "width": b_pixels.shape[1],
                "height": b_pixels.shape[0],
                "ppi": 500,
                "source_ppi": source_ppi,
            }
        )

    test_rows = [row for row in mapped if row["split"] == "test"]
    if len(test_rows) != expectations.test_records:
        raise MappingError(
            f"test mapping has {len(test_rows)} rows; expected {expectations.test_records}"
        )
    native_exact = sum(bool(row["native_pixels_equal"]) for row in mapped)
    if native_exact != expectations.native_exact:
        raise MappingError(
            f"native-exact count {native_exact}; expected {expectations.native_exact}"
        )
    _require_count(_counter(mapped, "split"), expectations.split_counts, "split counts")
    _require_count(_counter(mapped, "source_capture"), expectations.capture_counts, "capture counts")
    _require_count(
        _counter(test_rows, "source_capture"),
        expectations.test_capture_counts,
        "test capture counts",
    )
    _require_count(_counter(mapped, "device"), expectations.device_counts, "device counts")
    _require_count(
        _counter(test_rows, "device"), expectations.test_device_counts, "test device counts"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        full_path = staging / "mapping_full_private.csv"
        test_path = staging / "mapping_test_private.csv"
        inference_path = staging / "inference_manifest.csv"
        _write_csv(full_path, mapped, MAPPING_COLUMNS)
        _write_csv(test_path, test_rows, MAPPING_COLUMNS)
        _write_csv(inference_path, inference, INFERENCE_COLUMNS)
        receipt: Dict[str, Any] = {
            "schema_version": 1,
            "kind": "sd302b_sd302g_exact_mapping_receipt_v1",
            "status": "REFERENCE_VERIFIED_MAPPING",
            "private_biometric_artifact": True,
            "inputs": {
                "verified_manifest": {
                    "path": str(verified_manifest),
                    "sha256": manifest_hash,
                    "bytes": verified_manifest.stat().st_size,
                },
                "verified_provenance": {
                    "path": str(provenance_path),
                    "sha256": sha256_file(provenance_path),
                    "bytes": provenance_path.stat().st_size,
                },
                "frozen_split": {
                    "path": str(splits_path),
                    "sha256": split_hash,
                    "bytes": splits_path.stat().st_size,
                    "seed": split_payload["seed"],
                },
                "sd302b_root": str(sd302b_root),
                "sd302b_archive": archive_identity,
            },
            "mapping": {
                "records": len(mapped),
                "test_records": len(test_rows),
                "split_counts": _counter(mapped, "split"),
                "capture_counts": _counter(mapped, "source_capture"),
                "test_capture_counts": _counter(test_rows, "source_capture"),
                "device_counts": _counter(mapped, "device"),
                "test_device_counts": _counter(test_rows, "device"),
                "native_exact_records": native_exact,
                "native_exact_test_records": sum(
                    bool(row["native_pixels_equal"]) for row in test_rows
                ),
                "canonical_archive_500_exact_records": sum(
                    bool(row["canonical_pixels_equal"]) for row in mapped
                ),
                "canonical_archive_500_exact_test_records": sum(
                    bool(row["canonical_pixels_equal"]) for row in test_rows
                ),
                "mapping_records_sha256": canonical_sha256(mapped),
            },
            "equivalence_policy": {
                "actual_input_ppi": 500,
                "native_500": "decoded pixels compared directly",
                "native_1000": (
                    "decoded SD302g pixels resized to the decoded SD302b canvas with "
                    "PIL.Image.Resampling.LANCZOS, then compared exactly"
                ),
                "transform_provenance": "observed_exhaustive_not_authenticated_archive_recipe",
                "inference_reused": False,
                "ground_truth_opened_for_inference": False,
            },
            "capture_semantics": {
                "roll": "Rolled",
                "slap": "Plain (segmented slap)",
                "warning": (
                    "SD302b documents rolls and 4-4-2 slaps; 'Plain' here denotes "
                    "the individual slap-segmented archive images, not a separately "
                    "documented plain-impression capture."
                ),
            },
            "outputs": {
                "mapping_full_private.csv": {
                    "sha256": sha256_file(full_path),
                    "bytes": full_path.stat().st_size,
                    "rows": len(mapped),
                },
                "mapping_test_private.csv": {
                    "sha256": sha256_file(test_path),
                    "bytes": test_path.stat().st_size,
                    "rows": len(test_rows),
                },
                "inference_manifest.csv": {
                    "sha256": sha256_file(inference_path),
                    "bytes": inference_path.stat().st_size,
                    "rows": len(inference),
                    "mask_paths_empty": True,
                    "ground_truth_columns_absent": True,
                },
            },
        }
        _write_json(staging / "mapping_receipt.json", receipt)
        os.replace(staging, output)
        return receipt
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified-manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--sd302b-root", required=True, type=Path)
    parser.add_argument("--sd302b-archive", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        receipt = build(
            args.verified_manifest,
            args.splits,
            args.sd302b_root,
            args.out,
            sd302b_archive=args.sd302b_archive,
        )
    except (MappingError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
