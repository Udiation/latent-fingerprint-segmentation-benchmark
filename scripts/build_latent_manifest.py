#!/usr/bin/env python3
"""Build and audit the private SD302e/SD302h latent benchmark manifest.

The 2026 SD302e release stores full-resolution source PNGs at their actual
capture resolution, while SD302h annotation rasters are on the official
1,000-PPI annotation grid. The manifest therefore records image and mask PPI
separately. Matching removes only the resolution/bit-depth tokens and the
annotation-only impression suffix; all remaining filename fields must agree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


PPI_RE = re.compile(r"_(\d+)PPI_")
BPC_RE = re.compile(r"_(\d+)BPC_")
LP_IMPRESSION_RE = re.compile(r"(LP\d+)-\d+_")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def canonical_image_key(stem: str) -> str:
    """Return the cross-release identity key for one latent source image."""
    key, n_ppi = PPI_RE.subn("_{PPI}_", stem, count=1)
    key, n_bpc = BPC_RE.subn("_{BPC}_", key, count=1)
    key = LP_IMPRESSION_RE.sub(r"\1_", key, count=1)
    if n_ppi != 1 or n_bpc != 1:
        raise ValueError(f"filename lacks one PPI/BPC token: {stem!r}")
    return key


def filename_ppi(stem: str) -> int:
    """Read, rather than infer, the source PPI encoded in an SD302 filename."""
    match = PPI_RE.search(stem)
    if not match:
        raise ValueError(f"filename has no PPI token: {stem!r}")
    ppi = int(match.group(1))
    if ppi <= 0:
        raise ValueError(f"filename has invalid PPI: {stem!r}")
    return ppi


def png_shape(path: Path) -> Tuple[int, int]:
    """Return ``(height, width)`` from a PNG IHDR without decoding pixels."""
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"not a valid PNG header: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid PNG dimensions {width}x{height}: {path}")
    return height, width


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a source file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_binary_mask(path: Path) -> np.ndarray:
    """Read one examiner mask as a boolean array."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"could not decode mask: {path}")
    return mask > 0


def write_binary_mask(path: Path, mask: np.ndarray) -> None:
    """Write a deterministic 0/255 PNG, atomically replacing an older result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not ok:
        raise OSError(f"could not encode mask: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded.tobytes())
    temporary.replace(path)


def union_masks(paths: Sequence[Path]) -> np.ndarray:
    """Return the Boolean union of masks that share one annotation grid."""
    if not paths:
        raise ValueError("cannot union an empty mask list")
    masks = [read_binary_mask(path) for path in paths]
    shapes = {mask.shape for mask in masks}
    if len(shapes) != 1:
        raise ValueError(f"cannot union masks with different shapes: {sorted(shapes)}")
    return np.logical_or.reduce(masks)


def ordered_gt_digest(rows: Sequence[Mapping[str, object]]) -> str:
    """Hash the ordered source-to-union-GT content mapping.

    Each line is canonical JSON so identifiers and hashes cannot be
    concatenated ambiguously. A schema prefix makes future format changes
    explicit rather than silently producing an incomparable digest.
    """
    digest = hashlib.sha256(b"segbench-source-union-gt-v1\n")
    for row in sorted(rows, key=lambda item: str(item["image_id"])):
        item = {
            "image_id": str(row["image_id"]),
            "quality_union_sha256": str(row["quality_union_sha256"]),
            "roi_union_sha256": str(row["roi_union_sha256"]),
        }
        digest.update(
            (json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        )
    return digest.hexdigest()


def load_splits(path: Path) -> Tuple[int, Dict[str, str]]:
    """Load the frozen subject split and verify its schema and counts."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"seed", "splits"} or not isinstance(payload["splits"], dict):
        raise ValueError("split JSON must contain exactly seed and splits")
    splits = {str(key): str(value) for key, value in payload["splits"].items()}
    invalid = sorted(set(splits.values()) - {"train", "val", "test"})
    if invalid:
        raise ValueError(f"invalid split labels: {invalid}")
    counts = Counter(splits.values())
    expected = {"train": 136, "val": 24, "test": 40}
    if dict(counts) != expected:
        raise ValueError(f"subject split counts {dict(counts)} != {expected}")
    return int(payload["seed"]), dict(sorted(splits.items()))


def index_source_images(root: Path) -> Dict[str, Path]:
    """Index source PNGs by strict cross-release filename identity."""
    index: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*.png")):
        index[canonical_image_key(path.stem)].append(path.resolve())
    collisions = {key: paths for key, paths in index.items() if len(paths) != 1}
    if collisions:
        examples = {key: [str(path) for path in paths] for key, paths in list(collisions.items())[:5]}
        raise ValueError(f"source-image key collisions: {examples}")
    return {key: paths[0] for key, paths in index.items()}


def build_rows(
    source_root: Path,
    gt_root: Path,
    splits: Mapping[str, str],
    target_ppi: float,
    union_root: Path,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    """Build one audited manifest row per source image.

    SD302h has one annotation record per impression, while a small number of
    SD302e source images contain multiple impressions. A segmentation model
    produces one foreground mask for the complete source image, so all examiner
    ROIs associated with that image are unioned into one Table-A target. Quality
    maps are likewise unioned only when every annotated impression has a valid
    map; otherwise the complete source image is excluded from Table A-prime.
    The returned mapping retains every impression-level association for audit.
    """
    summary_path = gt_root / "summary.csv"
    summary = pd.read_csv(summary_path, dtype={"subject_id": str})
    required = {
        "id", "image_name", "subject_id", "examiner_value", "ppi", "width", "height"
    }
    missing_columns = sorted(required - set(summary.columns))
    if missing_columns:
        raise ValueError(f"GT summary missing columns: {missing_columns}")
    if summary["id"].duplicated().any():
        raise ValueError("GT summary contains duplicate impression IDs")

    source_index = index_source_images(source_root)
    source_paths = set(source_index.values())
    used_sources: set[Path] = set()
    annotations: List[Dict[str, object]] = []
    physical_deltas: List[Tuple[int, int]] = []
    for record in summary.sort_values("id").to_dict("records"):
        image_id = str(record["id"])
        image_name = str(record["image_name"])
        subject = str(record["subject_id"])
        if subject not in splits:
            raise ValueError(f"{image_id}: subject {subject!r} is absent from frozen split")
        key = canonical_image_key(image_name)
        source = source_index.get(key)
        if source is None:
            raise ValueError(f"{image_id}: no SD302e source image for {image_name!r}")
        used_sources.add(source)

        source_ppi = filename_ppi(source.stem)
        mask_ppi = float(record["ppi"])
        source_h, source_w = png_shape(source)
        roi_path = (gt_root / "gt_roi" / f"{image_id}.png").resolve()
        if not roi_path.is_file():
            raise FileNotFoundError(f"missing ROI mask: {roi_path}")
        mask_h, mask_w = png_shape(roi_path)
        if (mask_w, mask_h) != (int(record["width"]), int(record["height"])):
            raise ValueError(
                f"{image_id}: ROI mask size {(mask_w, mask_h)} disagrees with summary "
                f"{(record['width'], record['height'])}"
            )

        expected_h = round(source_h * mask_ppi / source_ppi)
        expected_w = round(source_w * mask_ppi / source_ppi)
        delta = (expected_h - mask_h, expected_w - mask_w)
        physical_deltas.append(delta)
        if abs(delta[0]) > 1 or abs(delta[1]) > 1:
            raise ValueError(
                f"{image_id}: source {source_w}x{source_h}@{source_ppi} PPI does not "
                f"align with mask {mask_w}x{mask_h}@{mask_ppi:g} PPI; delta={delta}"
            )

        quality_path = (gt_root / "gt_quality" / f"{image_id}.png").resolve()
        quality_value = str(quality_path) if quality_path.is_file() else ""
        if quality_value and png_shape(quality_path) != (mask_h, mask_w):
            raise ValueError(f"{image_id}: quality mask size disagrees with ROI mask")
        annotations.append(
            {
                "impression_id": image_id,
                "image_path": str(source),
                "impression_roi_path": str(roi_path),
                "impression_quality_path": quality_value,
                "subject_id": subject,
                "examiner_value": str(record["examiner_value"]),
                "annotation_ppi": mask_ppi,
                "source_width": source_w,
                "source_height": source_h,
                "mask_width": mask_w,
                "mask_height": mask_h,
                "source_image_name": source.name,
            }
        )

    observed_subjects = {str(row["subject_id"]) for row in annotations}
    if observed_subjects != set(splits):
        missing = sorted(set(splits) - observed_subjects)
        extra = sorted(observed_subjects - set(splits))
        raise ValueError(f"manifest/split subject coverage differs: missing={missing} extra={extra}")

    annotation_frame = pd.DataFrame(annotations)
    annotation_split_series = annotation_frame["subject_id"].map(splits)
    annotation_split_value_counts = (
        annotation_frame.assign(split=annotation_split_series)
        .groupby(["split", "examiner_value"])
        .size()
        .unstack(fill_value=0)
    )

    # Hash the complete source partition, including the four unannotated release
    # images. An exact-content duplicate crossing a frozen subject split is a
    # leakage stop condition rather than an item to remove after evaluation.
    source_hashes = {path: sha256_file(path) for path in sorted(source_paths)}
    hash_groups: Dict[str, List[Path]] = defaultdict(list)
    for path, digest in source_hashes.items():
        hash_groups[digest].append(path)
    duplicate_groups = {key: value for key, value in hash_groups.items() if len(value) > 1}
    cross_split_duplicates: List[Dict[str, object]] = []
    for digest, paths in sorted(duplicate_groups.items()):
        subjects = {path.name[:8] for path in paths}
        unknown = sorted(subjects - set(splits))
        if unknown:
            raise ValueError(f"duplicate-content audit found unknown subjects: {unknown}")
        labels = {splits[subject] for subject in subjects}
        if len(labels) > 1:
            cross_split_duplicates.append(
                {
                    "sha256": digest,
                    "subjects": sorted(subjects),
                    "splits": sorted(labels),
                    "n_files": len(paths),
                }
            )
    if cross_split_duplicates:
        raise ValueError(
            "exact source-image content crosses frozen splits; examples="
            f"{cross_split_duplicates[:5]}"
        )

    rows: List[Dict[str, object]] = []
    mapping_rows: List[Dict[str, object]] = []
    grouped = annotation_frame.groupby("image_path", sort=True, dropna=False)
    for image_path, group in grouped:
        source = Path(str(image_path)).resolve()
        subjects = sorted(set(group["subject_id"].astype(str)))
        if len(subjects) != 1:
            raise ValueError(f"source image maps to multiple subjects: {source}")
        subject = subjects[0]
        annotation_ppi = sorted(set(group["annotation_ppi"].astype(float)))
        mask_shapes = sorted(set(zip(group["mask_height"], group["mask_width"])))
        source_shapes = sorted(set(zip(group["source_height"], group["source_width"])))
        if len(annotation_ppi) != 1 or len(mask_shapes) != 1 or len(source_shapes) != 1:
            raise ValueError(
                f"source annotations disagree on geometry: {source}; "
                f"ppi={annotation_ppi}, masks={mask_shapes}, sources={source_shapes}"
            )

        roi_paths = [Path(path) for path in group["impression_roi_path"]]
        roi_union = union_masks(roi_paths)
        source_id = source.stem
        roi_union_path = (union_root / "gt_roi" / f"{source_id}.png").resolve()
        write_binary_mask(roi_union_path, roi_union)
        roi_union_sha256 = sha256_file(roi_union_path)

        quality_paths = [
            Path(path) for path in group["impression_quality_path"]
            if isinstance(path, str) and path
        ]
        quality_complete = len(quality_paths) == len(group)
        quality_union_path_value = ""
        quality_union_sha256 = ""
        if quality_complete:
            quality_union = union_masks(quality_paths)
            if quality_union.shape != roi_union.shape:
                raise ValueError(f"quality/ROI union shape mismatch: {source}")
            quality_union_path = (union_root / "gt_quality" / f"{source_id}.png").resolve()
            write_binary_mask(quality_union_path, quality_union)
            quality_union_path_value = str(quality_union_path)
            quality_union_sha256 = sha256_file(quality_union_path)

        values = sorted(set(group["examiner_value"].astype(str)))
        examiner_value = values[0] if len(values) == 1 else "MIXED"
        source_ppi = filename_ppi(source.stem)
        source_h, source_w = source_shapes[0]
        mask_h, mask_w = mask_shapes[0]
        digest = source_hashes[source]
        split = splits[subject]
        rows.append(
            {
                "image_id": source_id,
                "image_path": str(source),
                "mask_path": str(roi_union_path),
                "quality_mask_path": quality_union_path_value,
                "subject_id": subject,
                "subset": "finger_latent",
                "ppi": source_ppi,
                "mask_ppi": annotation_ppi[0],
                "quality_mask_ppi": annotation_ppi[0] if quality_complete else "",
                "examiner_value": examiner_value,
                "examiner_values": "|".join(values),
                "n_impressions": len(group),
                "quality_complete": quality_complete,
                "roi_union_sha256": roi_union_sha256,
                "quality_union_sha256": quality_union_sha256,
                "source_sha256": digest,
                "source_width": source_w,
                "source_height": source_h,
                "mask_width": mask_w,
                "mask_height": mask_h,
                "target_width": max(1, round(source_w * target_ppi / source_ppi)),
                "target_height": max(1, round(source_h * target_ppi / source_ppi)),
                "target_ppi": target_ppi,
                "source_image_name": source.name,
            }
        )
        for record in group.sort_values("impression_id").to_dict("records"):
            mapping_rows.append(
                {
                    "source_image_id": source_id,
                    "source_image_path": str(source),
                    "source_sha256": digest,
                    "subject_id": subject,
                    "split": split,
                    "impression_id": record["impression_id"],
                    "examiner_value": record["examiner_value"],
                    "impression_roi_path": record["impression_roi_path"],
                    "impression_quality_path": record["impression_quality_path"],
                    "quality_valid": bool(record["impression_quality_path"]),
                }
            )

    frame = pd.DataFrame(rows)
    split_series = frame["subject_id"].map(splits)
    split_value_counts = (
        frame.assign(split=split_series)
        .groupby(["split", "examiner_value"])
        .size()
        .unstack(fill_value=0)
    )
    delta_counts = Counter(f"{dh},{dw}" for dh, dw in physical_deltas)
    n_annotation_quality = int(annotation_frame["impression_quality_path"].astype(bool).sum())
    multiplicities = Counter(frame["n_impressions"])
    audit: Dict[str, object] = {
        # Legacy count keys retain their impression-level meaning so release
        # validation remains comparable with the official annotation inventory.
        "gt_rows": len(annotations),
        "source_images": len(source_paths),
        "matched_source_images": len(used_sources),
        "unmatched_source_images": len(source_paths - used_sources),
        "roi_masks": len(annotations),
        "quality_masks": n_annotation_quality,
        "annotation_impressions": len(annotations),
        "annotation_roi_masks": len(annotations),
        "annotation_quality_masks": n_annotation_quality,
        "annotation_quality_exclusions": len(annotations) - n_annotation_quality,
        "source_manifest_rows": len(rows),
        "source_roi_union_masks": int(frame["mask_path"].astype(bool).sum()),
        "source_quality_union_masks": int(frame["quality_mask_path"].astype(bool).sum()),
        "source_quality_exclusions": int((~frame["quality_complete"]).sum()),
        "mixed_examiner_sources": int((frame["examiner_value"] == "MIXED").sum()),
        "multi_impression_sources": int((frame["n_impressions"] > 1).sum()),
        "multi_impression_annotation_rows": int(
            frame.loc[frame["n_impressions"] > 1, "n_impressions"].sum()
        ),
        "impression_multiplicity_counts": {
            str(key): int(value) for key, value in sorted(multiplicities.items())
        },
        "subjects": len(observed_subjects),
        "split_subject_counts": dict(Counter(splits.values())),
        "split_image_counts": {
            str(key): int(value) for key, value in Counter(split_series).items()
        },
        "split_by_examiner_value": {
            split: {str(key): int(value) for key, value in values.items()}
            for split, values in split_value_counts.to_dict(orient="index").items()
        },
        "split_annotation_counts": {
            str(key): int(value) for key, value in Counter(annotation_split_series).items()
        },
        "split_by_examiner_value_annotations": {
            split: {str(key): int(value) for key, value in values.items()}
            for split, values in annotation_split_value_counts.to_dict(orient="index").items()
        },
        "source_ppi_counts": {
            str(key): int(value)
            for key, value in sorted(Counter(frame["ppi"]).items())
        },
        "annotation_ppi_counts": {
            str(key): int(value)
            for key, value in sorted(Counter(annotation_frame["annotation_ppi"]).items())
        },
        "source_mask_ppi_counts": {
            str(key): int(value)
            for key, value in sorted(Counter(frame["mask_ppi"]).items())
        },
        "physical_shape_delta_counts": dict(sorted(delta_counts.items())),
        "content_hashed_source_images": len(source_hashes),
        "duplicate_content_groups": len(duplicate_groups),
        "duplicate_content_files": sum(len(paths) for paths in duplicate_groups.values()),
        "cross_split_duplicate_groups": 0,
        "ordered_source_gt_sha256": ordered_gt_digest(rows),
        "ordered_source_gt_digest_scheme": "segbench-source-union-gt-v1",
    }
    return rows, mapping_rows, audit


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--audit-out", type=Path)
    parser.add_argument(
        "--union-root", required=True, type=Path,
        help="restricted output directory for source-level union masks",
    )
    parser.add_argument(
        "--mapping-out", type=Path,
        help="restricted impression-to-source audit CSV (default: beside manifest)",
    )
    parser.add_argument("--target-ppi", type=float, default=500.0)
    parser.add_argument("--expect-gt-rows", type=int, default=10092)
    parser.add_argument("--expect-source-images", type=int, default=9990)
    parser.add_argument("--expect-matched-source-images", type=int, default=9986)
    parser.add_argument("--expect-quality-masks", type=int, default=10024)
    parser.add_argument("--expect-quality-source-masks", type=int, default=9920)
    parser.add_argument("--expect-multi-impression-sources", type=int, default=102)
    parser.add_argument("--expect-unmatched-sources", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    seed, splits = load_splits(args.splits.resolve())
    rows, mapping_rows, audit = build_rows(
        args.images_root.resolve(), args.gt_root.resolve(), splits, args.target_ppi,
        args.union_root.resolve(),
    )
    expectations = {
        "gt_rows": args.expect_gt_rows,
        "source_images": args.expect_source_images,
        "matched_source_images": args.expect_matched_source_images,
        "quality_masks": args.expect_quality_masks,
        "source_quality_union_masks": args.expect_quality_source_masks,
        "multi_impression_sources": args.expect_multi_impression_sources,
        "unmatched_source_images": args.expect_unmatched_sources,
    }
    mismatches = {
        key: {"expected": expected, "observed": audit[key]}
        for key, expected in expectations.items()
        if audit[key] != expected
    }
    if mismatches:
        raise ValueError(f"release count validation failed: {mismatches}")

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    mapping_out = (
        args.mapping_out or out.with_name(f"{out.stem}_impressions.csv")
    ).resolve()
    mapping_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mapping_rows).to_csv(mapping_out, index=False)
    audit.update(
        {
            "split_seed": seed,
            "target_ppi": args.target_ppi,
            "manifest": str(out),
            "manifest_sha256": sha256_file(out),
            "impression_mapping": str(mapping_out),
            "impression_mapping_sha256": sha256_file(mapping_out),
            "union_root": str(args.union_root.resolve()),
        }
    )
    audit_out = (args.audit_out or out.with_suffix(".audit.json")).resolve()
    audit_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
