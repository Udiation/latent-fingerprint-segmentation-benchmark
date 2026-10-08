#!/usr/bin/env python3
"""Build SD302g minutiae-region lower-bound masks in restricted storage.

SD302g contains EFS Profile 2 exemplar annotations.  Field 9.331 supplies
expert minutiae, but the release does not supply fingerprint-outline ground
truth.  This script therefore rasterises only the convex hull of the expert
minutiae.  The result is a *minutiae-region lower bound*, never a full
foreground mask and never a supervised-training target.

The traditional ANSI/NIST parser and Type-14 metadata decoder are shared with
``build_sd302h_gt.py``.  Logical records are bounded by their declared length;
binary image payloads are not split on separator bytes.

Normative references:

* NIST SP 500-290e3 field definitions for EFS ROI and minutiae records.
* NIST TN 2190 and TN 2367 (SD302 supplement and release descriptions).

All decoded images, masks, point files, manifests, and diagnostics are private
biometric outputs.  The output directory is therefore required to be outside
the source repository.
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
import tempfile
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


# The sibling module owns the already-tested length-delimited traditional
# ANSI/NIST parser.  Add only this script directory, never a machine-specific
# location, so imports work both as a CLI and in synthetic tests.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_sd302h_gt import (  # noqa: E402
    FormatError,
    ROI,
    RS,
    TaggedRecord,
    US,
    parse_image_record,
    parse_measurement_units,
    parse_roi,
    parse_transaction,
)


REFERENCE_RECORDS = 2_380
REFERENCE_GROUP_COUNTS: Mapping[str, int] = {
    "baseline": 2_194,
    "challengers": 186,
}
REFERENCE_CAPTURE_COUNTS: Mapping[str, int] = {
    "roll": 2_332,
    "slap": 48,
}
REFERENCE_SPLIT_COUNTS: Mapping[str, int] = {
    "train": 136,
    "val": 24,
    "test": 40,
}

MANIFEST_COLUMNS = (
    "image_id",
    "image_path",
    "hull_path",
    "minutiae_path",
    "source_member",
    "subject_id",
    "split",
    "group",
    "device",
    "capture",
    "finger_position",
    "finger_segment",
    "impression_type_code",
    "idc",
    "width",
    "height",
    "ppi",
    "compression",
    "n_minutiae",
    "n_unique_minutiae",
    "hull_area_px",
    "hull_area_ratio",
)
MINUTIA_COLUMNS = (
    "index",
    "x_units",
    "y_units",
    "x_px_float",
    "y_px_float",
    "x_px",
    "y_px",
    "theta",
    "type",
    "radius_uncertainty_units",
    "direction_uncertainty_degrees",
)
EXCLUSION_COLUMNS = ("image_id", "source_member", "reason", "detail")
ERROR_COLUMNS = ("source_member", "stage", "error")

_FILENAME = re.compile(
    r"^(?P<subject>[0-9]{8})_(?P<device>[^_]+)_(?P<ppi>[0-9]+)_"
    r"(?P<capture>[^_]+)_(?P<finger>[0-9]{2})$"
)


class HullUnavailable(FormatError):
    """An otherwise valid record cannot define a two-dimensional hull."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


@dataclass(frozen=True)
class MemberMetadata:
    """Release metadata represented redundantly by path and filename."""

    member: str
    image_id: str
    subject_id: str
    group: str
    device: str
    named_ppi: int
    capture: str
    finger_position: str


@dataclass(frozen=True)
class Minutia:
    """One expert Field 9.331 minutia in physical and raster coordinates."""

    index: int
    x_units: int
    y_units: int
    x_px_float: float
    y_px_float: float
    x_px: int
    y_px: int
    theta: int
    kind: str
    radius_uncertainty_units: Optional[int]
    direction_uncertainty_degrees: Optional[int]


def _ascii(value: bytes, label: str) -> str:
    try:
        return value.decode("ascii")
    except UnicodeDecodeError as exc:
        raise FormatError(f"{label} is not ASCII") from exc


def _uint(value: bytes, label: str, *, upper: Optional[int] = None) -> int:
    text = _ascii(value, label)
    if not re.fullmatch(r"[0-9]+", text):
        raise FormatError(f"{label} is not an unsigned decimal integer: {text!r}")
    number = int(text)
    if upper is not None and number > upper:
        raise FormatError(f"{label} must not exceed {upper}, got {number}")
    return number


def _optional_uint(value: bytes, label: str, *, upper: Optional[int] = None) -> Optional[int]:
    return None if value == b"" else _uint(value, label, upper=upper)


def _unit_to_pixel(value: int, ppi: float) -> float:
    """Convert EFS lambda units (0.01 mm) to a continuous pixel coordinate."""
    return float(value) * float(ppi) / 2540.0


def _nearest_pixel(value: float) -> int:
    """Map a non-negative continuous coordinate to its nearest raster index.

    ``floor(x + 0.5)`` makes the tie rule explicit and avoids Python's
    round-to-even behavior.  The unrounded coordinate is retained in the point
    file, and out-of-frame results fail rather than being clipped.
    """
    if not math.isfinite(value) or value < 0:
        raise FormatError(f"invalid continuous pixel coordinate {value!r}")
    return int(math.floor(value + 0.5))


def parse_member_metadata(member: str) -> MemberMetadata:
    """Parse and cross-check the documented SD302g archive naming tuple."""
    path = PurePosixPath(member)
    parts = path.parts
    try:
        ebts = parts.index("ebts")
    except ValueError as exc:
        raise FormatError(f"member is not below an ebts directory: {member!r}") from exc
    tail = parts[ebts + 1 :]
    if len(tail) != 6 or tail[1] != "irr":
        raise FormatError(
            "expected ebts/{baseline|challengers}/irr/device/ppi/subject/file.irr, "
            f"got {member!r}"
        )
    group, _, device, ppi_text, subject, filename = tail
    if group not in REFERENCE_GROUP_COUNTS:
        raise FormatError(f"unsupported SD302g group {group!r}")
    if not re.fullmatch(r"[0-9]+", ppi_text):
        raise FormatError(f"archive PPI component is not numeric: {ppi_text!r}")
    if path.suffix.lower() != ".irr":
        raise FormatError(f"SD302g transaction is not .irr: {member!r}")
    match = _FILENAME.fullmatch(path.stem)
    if match is None:
        raise FormatError(f"unsupported SD302g filename: {filename!r}")
    expected = {
        "subject": subject,
        "device": device,
        "ppi": ppi_text,
    }
    observed = {key: match.group(key) for key in expected}
    if observed != expected:
        raise FormatError(
            f"filename metadata {observed} disagrees with archive path {expected}"
        )
    capture = match.group("capture").lower()
    if capture not in REFERENCE_CAPTURE_COUNTS:
        raise FormatError(f"unsupported SD302g capture label {capture!r}")
    return MemberMetadata(
        member=member,
        image_id=path.stem,
        subject_id=subject,
        group=group,
        device=device,
        named_ppi=int(ppi_text),
        capture=capture,
        finger_position=match.group("finger"),
    )


def parse_finger_position(record: TaggedRecord, expected: str) -> Tuple[str, str]:
    """Parse Field 9.302 and require its FRGP to match the release filename."""
    items = record.one("9.302").split(US)
    if not items or not items[0]:
        raise FormatError("9.302 has no friction-ridge position")
    position = f"{_uint(items[0], '9.302 FGP'):02d}"
    if position != expected:
        raise FormatError(
            f"9.302 finger position {position!r} disagrees with filename {expected!r}"
        )
    segment = _ascii(items[1], "9.302 FSG") if len(items) >= 2 and items[1] else ""
    return position, segment


def _roi_feature_origin(roi: ROI) -> Tuple[int, int]:
    """Return the image-relative origin of ROI-relative EFS feature points."""
    if roi.vertices:
        # SP 500-290 defines feature locations relative to the bounding rectangle
        # of the ROI polygon; polygon vertices themselves are relative to EHO/EVO.
        return (
            roi.x_offset_units + min(x for x, _ in roi.vertices),
            roi.y_offset_units + min(y for _, y in roi.vertices),
        )
    return roi.x_offset_units, roi.y_offset_units


def validate_full_image_roi(
    roi: ROI, width: int, height: int, ppi: float, tolerance_px: float = 2.0
) -> None:
    """Require the release's measured full-canvas ROI invariant.

    A smaller rectangle or polygon would change the coordinate origin and would
    also constitute genuine foreground geometry.  Such a record must be
    reviewed under a different protocol rather than silently treated as one of
    the measured full-image SD302g records.
    """
    roi_width = _unit_to_pixel(roi.width_units, ppi)
    roi_height = _unit_to_pixel(roi.height_units, ppi)
    if roi.vertices or roi.x_offset_units or roi.y_offset_units:
        raise FormatError(
            "SD302g production protocol requires a zero-offset rectangular "
            "full-image 9.300 ROI"
        )
    if abs(roi_width - width) > tolerance_px or abs(roi_height - height) > tolerance_px:
        raise FormatError(
            f"9.300 extent {(roi_width, roi_height)} px is not the full "
            f"Type-14 image {(width, height)} within {tolerance_px:g} px"
        )


def parse_minutiae(
    record: TaggedRecord,
    roi: ROI,
    width: int,
    height: int,
    ppi: float,
) -> List[Minutia]:
    """Parse Field 9.331 and convert its RS/US entries to image coordinates."""
    raw = record.one("9.331")
    subfields = raw.split(RS)
    if not subfields or any(value == b"" for value in subfields):
        raise FormatError("9.331 must contain non-empty RS-separated minutiae")
    origin_x, origin_y = _roi_feature_origin(roi)
    parsed: List[Minutia] = []
    for index, subfield in enumerate(subfields, start=1):
        items = subfield.split(US)
        if len(items) not in (4, 5, 6):
            raise FormatError(
                f"9.331 minutia {index} has {len(items)} items; expected 4--6"
            )
        x_units = _uint(items[0], f"9.331[{index}] MXC")
        y_units = _uint(items[1], f"9.331[{index}] MYC")
        if x_units >= roi.width_units or y_units >= roi.height_units:
            raise FormatError(
                f"9.331[{index}] coordinate {(x_units, y_units)} exceeds "
                f"9.300 extent {(roi.width_units, roi.height_units)}"
            )
        theta = _uint(items[2], f"9.331[{index}] MTD", upper=359)
        kind = _ascii(items[3], f"9.331[{index}] MTY")
        if kind not in {"E", "B", "X"}:
            raise FormatError(f"9.331[{index}] unsupported minutia type {kind!r}")
        radius = (
            _optional_uint(items[4], f"9.331[{index}] MRU", upper=999)
            if len(items) >= 5
            else None
        )
        direction_uncertainty = (
            _optional_uint(items[5], f"9.331[{index}] MDU", upper=180)
            if len(items) >= 6
            else None
        )
        x_float = _unit_to_pixel(origin_x + x_units, ppi)
        y_float = _unit_to_pixel(origin_y + y_units, ppi)
        x_pixel, y_pixel = _nearest_pixel(x_float), _nearest_pixel(y_float)
        if not (0 <= x_pixel < width and 0 <= y_pixel < height):
            raise FormatError(
                f"9.331[{index}] pixel {(x_pixel, y_pixel)} is outside "
                f"image {(width, height)}"
            )
        parsed.append(
            Minutia(
                index=index,
                x_units=x_units,
                y_units=y_units,
                x_px_float=x_float,
                y_px_float=y_float,
                x_px=x_pixel,
                y_px=y_pixel,
                theta=theta,
                kind=kind,
                radius_uncertainty_units=radius,
                direction_uncertainty_degrees=direction_uncertainty,
            )
        )
    return parsed


def rasterize_minutiae_hull(
    minutiae: Sequence[Minutia], width: int, height: int
) -> np.ndarray:
    """Fill the exact convex hull of unique expert minutia raster locations."""
    unique = sorted({(point.x_px, point.y_px) for point in minutiae})
    if len(unique) < 3:
        raise HullUnavailable(
            "fewer_than_three_unique_minutiae",
            f"only {len(unique)} unique raster locations",
        )
    points = np.asarray(unique, dtype=np.int32)
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    if len(hull) < 3 or cv2.contourArea(hull) <= 0:
        raise HullUnavailable("degenerate_minutiae_hull", "minutiae are collinear")
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.reshape(-1, 2), 255)
    if not np.any(mask):
        raise HullUnavailable("empty_minutiae_hull", "filled hull is empty")
    missed = [
        point.index for point in minutiae if mask[point.y_px, point.x_px] == 0
    ]
    if missed:
        raise FormatError(f"hull rasterisation missed expert minutiae {missed[:10]}")
    return mask


def decode_type14_image(record: TaggedRecord, image_name: str) -> Tuple[object, np.ndarray]:
    """Decode and integrity-check an associated eight-bit Type-14 image."""
    image_record = parse_image_record(record, image_name)
    bits = _uint(record.one("14.012"), "14.012 BPX")
    if bits != 8:
        raise FormatError(f"unsupported Type-14 bit depth {bits}; expected 8")
    declared_hash = record.one("14.996", required=False)
    if declared_hash is not None:
        expected_hash = _ascii(declared_hash, "14.996 SHPS").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise FormatError("14.996 is not a canonical SHA-256 digest")
        actual_hash = hashlib.sha256(image_record.payload).hexdigest()
        if actual_hash != expected_hash:
            raise FormatError(
                f"Type-14 payload SHA-256 mismatch: {actual_hash} != {expected_hash}"
            )
    decoded = cv2.imdecode(
        np.frombuffer(image_record.payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    if decoded is None:
        raise FormatError(
            f"OpenCV cannot decode Type-14 {image_record.compression!r} payload"
        )
    if decoded.ndim != 2 or decoded.dtype != np.uint8:
        raise FormatError(
            f"decoded Type-14 image must be 8-bit grayscale, got "
            f"shape={decoded.shape}, dtype={decoded.dtype}"
        )
    if decoded.shape != (image_record.height, image_record.width):
        raise FormatError(
            f"decoded image shape {decoded.shape} disagrees with metadata "
            f"{(image_record.height, image_record.width)}"
        )
    return image_record, decoded


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.png")
    if not cv2.imwrite(str(temporary), image):
        raise OSError(f"OpenCV failed to write {temporary}")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    """Hash an input or private artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_minutiae(path: Path, points: Sequence[Minutia]) -> None:
    rows = [
        {
            "index": point.index,
            "x_units": point.x_units,
            "y_units": point.y_units,
            "x_px_float": f"{point.x_px_float:.9f}",
            "y_px_float": f"{point.y_px_float:.9f}",
            "x_px": point.x_px,
            "y_px": point.y_px,
            "theta": point.theta,
            "type": point.kind,
            "radius_uncertainty_units": (
                "" if point.radius_uncertainty_units is None else point.radius_uncertainty_units
            ),
            "direction_uncertainty_degrees": (
                ""
                if point.direction_uncertainty_degrees is None
                else point.direction_uncertainty_degrees
            ),
        }
        for point in points
    ]
    _write_csv(path, rows, MINUTIA_COLUMNS)


def _load_splits(path: Path) -> Tuple[object, Dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or set(payload) != {"seed", "splits"}:
        raise FormatError("split JSON must contain exactly 'seed' and 'splits'")
    raw = payload["splits"]
    if not isinstance(raw, dict):
        raise FormatError("split JSON 'splits' must be an object")
    splits: Dict[str, str] = {}
    for subject, label in raw.items():
        if not isinstance(subject, str) or not subject:
            raise FormatError("split subject IDs must be non-empty strings")
        if label not in REFERENCE_SPLIT_COUNTS:
            raise FormatError(f"unsupported split label for {subject!r}: {label!r}")
        splits[subject] = label
    counts = Counter(splits.values())
    if dict(counts) != dict(REFERENCE_SPLIT_COUNTS):
        raise FormatError(
            f"split subject counts {dict(counts)} != {dict(REFERENCE_SPLIT_COUNTS)}"
        )
    return payload["seed"], splits


def _canonical_digest(payload: object) -> str:
    """Digest canonical JSON so private reference identities remain external."""
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reference_inventory(members: Sequence[str]) -> Tuple[str, str]:
    canonical_members = sorted(members)
    identities = []
    for member in canonical_members:
        item = parse_member_metadata(member)
        identities.append(
            {
                "source_member": item.member,
                "image_id": item.image_id,
                "subject_id": item.subject_id,
                "group": item.group,
                "device": item.device,
                "named_ppi": item.named_ppi,
                "capture": item.capture,
                "finger_position": item.finger_position,
            }
        )
    return _canonical_digest(canonical_members), _canonical_digest(identities)


def _load_reference_gate(
    path: Path,
    *,
    archive_path: Path,
    split_path: Path,
    split_seed: object,
    splits: Mapping[str, str],
) -> Mapping[str, object]:
    """Validate the private, independently prepared production reference gate."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    required = {
        "schema",
        "protocol",
        "trust_level",
        "archive",
        "split",
        "release_invariants",
        "coordinate_convention",
        "coordinate_oracle",
        "expected_outputs",
        "bootstrap",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise FormatError(
            f"reference gate must contain exactly {sorted(required)}"
        )
    if payload["schema"] != "cureblock.sd302g.reference-gate.v1":
        raise FormatError("unsupported SD302g reference-gate schema")
    if payload["protocol"] != "sd302g-minutiae-lower-bound-v2":
        raise FormatError("reference gate names the wrong protocol")
    if payload["trust_level"] not in {"archive_split_only", "reference_verified"}:
        raise FormatError("unsupported reference-gate trust level")

    archive = payload["archive"]
    if not isinstance(archive, dict):
        raise FormatError("reference-gate archive section must be an object")
    expected_archive = {
        "sha256": _sha256_file(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "irr_count": REFERENCE_RECORDS,
    }
    for key, expected in expected_archive.items():
        if archive.get(key) != expected:
            raise FormatError(
                f"reference-gate archive {key} {archive.get(key)!r} != {expected!r}"
            )
    for key in ("canonical_members_sha256", "canonical_record_identities_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(archive.get(key, ""))):
            raise FormatError(f"reference-gate archive {key} is not SHA-256")

    split = payload["split"]
    if not isinstance(split, dict):
        raise FormatError("reference-gate split section must be an object")
    canonical_assignments = [
        {"subject_id": subject, "split": splits[subject]}
        for subject in sorted(splits)
    ]
    expected_split = {
        "seed": split_seed,
        "raw_file_sha256": _sha256_file(split_path),
        "canonical_assignment_sha256": _canonical_digest(canonical_assignments),
        "subject_counts": dict(REFERENCE_SPLIT_COUNTS),
    }
    for key, expected in expected_split.items():
        if split.get(key) != expected:
            raise FormatError(
                f"reference-gate split {key} {split.get(key)!r} != {expected!r}"
            )

    invariants = payload["release_invariants"]
    expected_invariants = {
        "group_counts": dict(REFERENCE_GROUP_COUNTS),
        "capture_counts": dict(REFERENCE_CAPTURE_COUNTS),
        "roi_types": {
            "full_image": REFERENCE_RECORDS,
            "smaller_rectangle": 0,
            "polygon": 0,
        },
        "quality_map_present": 0,
    }
    if invariants != expected_invariants:
        raise FormatError(
            f"reference-gate release invariants {invariants!r} != "
            f"{expected_invariants!r}"
        )
    expected_coordinates = {
        "units": "0.01_mm",
        "continuous_formula": "units*ppi/2540",
        "origin": "image_top_left",
        "y_axis": "down",
        "raster_rounding": "floor(x+0.5)",
    }
    if payload["coordinate_convention"] != expected_coordinates:
        raise FormatError("reference-gate coordinate convention disagrees with code")
    bootstrap = payload["bootstrap"]
    if not isinstance(bootstrap, dict):
        raise FormatError("reference-gate bootstrap section must be an object")
    if (
        not isinstance(bootstrap.get("seed"), int)
        or bootstrap.get("replicates") != 1000
        or bootstrap.get("unit") != "subject"
        or bootstrap.get("interval") != "percentile_95"
        or bootstrap.get("paired") is not True
    ):
        raise FormatError("reference-gate bootstrap configuration is not v2")
    return payload


def _validate_reference_inventory(
    gate: Mapping[str, object], members: Sequence[str]
) -> None:
    member_digest, identity_digest = _reference_inventory(members)
    archive = gate["archive"]
    if not isinstance(archive, dict):
        raise FormatError("reference-gate archive section must be an object")
    if archive["canonical_members_sha256"] != member_digest:
        raise FormatError("archive member inventory disagrees with reference gate")
    if archive["canonical_record_identities_sha256"] != identity_digest:
        raise FormatError("archive record identities disagree with reference gate")


def _select_members(names: Sequence[str], *, validate_release: bool) -> List[str]:
    members = sorted(name for name in names if name.lower().endswith(".irr"))
    if not members:
        raise FormatError("archive contains no .irr transactions")
    metadata = [parse_member_metadata(member) for member in members]
    ids = [item.image_id for item in metadata]
    if len(ids) != len(set(ids)):
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        raise FormatError(f"duplicate SD302g transaction stems: {duplicates[:5]}")
    if validate_release:
        if len(members) != REFERENCE_RECORDS:
            raise FormatError(
                f"SD302g transaction count {len(members)} != {REFERENCE_RECORDS}"
            )
        groups = Counter(item.group for item in metadata)
        captures = Counter(item.capture for item in metadata)
        if dict(groups) != dict(REFERENCE_GROUP_COUNTS):
            raise FormatError(
                f"SD302g group counts {dict(groups)} != {dict(REFERENCE_GROUP_COUNTS)}"
            )
        if dict(captures) != dict(REFERENCE_CAPTURE_COUNTS):
            raise FormatError(
                f"SD302g capture counts {dict(captures)} != "
                f"{dict(REFERENCE_CAPTURE_COUNTS)}"
            )
    return members


def process_member(
    member: str,
    transaction: bytes,
    staging: Path,
    splits: Mapping[str, str],
    split_seed: object,
) -> Mapping[str, object]:
    """Decode one transaction and write its image, hull, and point manifest."""
    metadata = parse_member_metadata(member)
    if metadata.subject_id not in splits:
        raise FormatError(
            f"subject {metadata.subject_id!r} is absent from the frozen split"
        )
    records = parse_transaction(transaction)
    annotations = [record for record in records if record.record_type == 9]
    if len(annotations) != 1:
        raise FormatError(f"transaction has {len(annotations)} Type-9 records; expected one")
    annotation = annotations[0]
    parse_measurement_units(annotation)
    idc = _ascii(annotation.one("9.002"), "9.002 IDC")
    finger_position, finger_segment = parse_finger_position(
        annotation, metadata.finger_position
    )
    image_records: Dict[str, TaggedRecord] = {}
    for record in records:
        if record.record_type != 14:
            continue
        candidate_idc = _ascii(record.one("14.002"), "14.002 IDC")
        if candidate_idc in image_records:
            raise FormatError(f"multiple Type-14 images have IDC {candidate_idc!r}")
        image_records[candidate_idc] = record
    if idc not in image_records:
        raise FormatError(f"Type-9 IDC {idc!r} has no associated Type-14 image")
    image_record, decoded = decode_type14_image(image_records[idc], metadata.image_id)
    if not math.isclose(image_record.ppi, metadata.named_ppi, abs_tol=1.0, rel_tol=0.0):
        raise FormatError(
            f"Type-14 PPI {image_record.ppi:g} disagrees with archive PPI "
            f"{metadata.named_ppi}"
        )
    impression_type = _ascii(
        image_records[idc].one("14.003"), "14.003 IMP"
    )
    roi = parse_roi(annotation.one("9.300"))
    validate_full_image_roi(
        roi,
        image_record.width,
        image_record.height,
        image_record.ppi,
    )
    points = parse_minutiae(
        annotation,
        roi,
        image_record.width,
        image_record.height,
        image_record.ppi,
    )
    hull = rasterize_minutiae_hull(points, image_record.width, image_record.height)

    image_path = staging / "images" / f"{metadata.image_id}.png"
    hull_path = staging / "hulls" / f"{metadata.image_id}.png"
    minutiae_path = staging / "minutiae" / f"{metadata.image_id}.csv"
    _write_png(image_path, decoded)
    _write_png(hull_path, hull)
    _write_minutiae(minutiae_path, points)
    area = int(np.count_nonzero(hull))
    return {
        "image_id": metadata.image_id,
        # Relative paths survive atomic publication from the sibling staging
        # directory and avoid embedding a private machine path in the manifest.
        "image_path": str(Path("images") / image_path.name),
        "hull_path": str(Path("hulls") / hull_path.name),
        "minutiae_path": str(Path("minutiae") / minutiae_path.name),
        "source_member": metadata.member,
        "subject_id": metadata.subject_id,
        "split": splits[metadata.subject_id],
        "group": metadata.group,
        "device": metadata.device,
        "capture": metadata.capture,
        "finger_position": finger_position,
        "finger_segment": finger_segment,
        "impression_type_code": impression_type,
        "idc": idc,
        "width": image_record.width,
        "height": image_record.height,
        "ppi": f"{image_record.ppi:g}",
        "compression": image_record.compression,
        "n_minutiae": len(points),
        "n_unique_minutiae": len({(point.x_px, point.y_px) for point in points}),
        "hull_area_px": area,
        "hull_area_ratio": f"{area / hull.size:.12f}",
        # The seed is intentionally carried in README, not repeated per row.
        "_split_seed": split_seed,
    }


def _resolve_private_artifact(root: Path, value: object) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise FormatError(f"manifest artifact path is not confined to its output: {value!r}")
    return root / relative


def _validate_artifacts(rows: Sequence[Mapping[str, object]], staging: Path) -> None:
    ids = [str(row["image_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise FormatError("manifest contains duplicate image IDs")
    expected = set(ids)
    for directory, suffix in (("images", ".png"), ("hulls", ".png"), ("minutiae", ".csv")):
        paths = list((staging / directory).glob(f"*{suffix}"))
        actual = {path.stem for path in paths}
        if len(paths) != len(actual) or actual != expected:
            raise FormatError(
                f"{directory} identities disagree with manifest: "
                f"files={len(paths)}, unique={len(actual)}, expected={len(expected)}"
            )
    for row in rows:
        image = cv2.imread(
            str(_resolve_private_artifact(staging, row["image_path"])),
            cv2.IMREAD_UNCHANGED,
        )
        if image is None or image.shape != (int(row["height"]), int(row["width"])):
            raise FormatError(f"invalid decoded image for {row['image_id']!r}")
        hull = cv2.imread(
            str(_resolve_private_artifact(staging, row["hull_path"])),
            cv2.IMREAD_GRAYSCALE,
        )
        if hull is None or hull.shape != (int(row["height"]), int(row["width"])):
            raise FormatError(f"invalid generated hull for {row['image_id']!r}")
        with _resolve_private_artifact(staging, row["minutiae_path"]).open(
            newline="", encoding="utf-8"
        ) as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != MINUTIA_COLUMNS:
                raise FormatError(f"minutiae schema mismatch for {row['image_id']!r}")
            points = list(reader)
        if len(points) != int(row["n_minutiae"]):
            raise FormatError(f"minutiae count mismatch for {row['image_id']!r}")
        if any(hull[int(point["y_px"]), int(point["x_px"])] == 0 for point in points):
            raise FormatError(f"hull gate failed for {row['image_id']!r}: MC < 100%")


def _point_digests(root: Path, row: Mapping[str, object]) -> Tuple[str, str]:
    with _resolve_private_artifact(root, row["minutiae_path"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        points = list(csv.DictReader(handle))
    raw = [
        {
            "x_units": int(point["x_units"]),
            "y_units": int(point["y_units"]),
            "theta": int(point["theta"]),
            "type": point["type"],
            "radius_uncertainty_units": (
                None
                if point["radius_uncertainty_units"] == ""
                else int(point["radius_uncertainty_units"])
            ),
            "direction_uncertainty_degrees": (
                None
                if point["direction_uncertainty_degrees"] == ""
                else int(point["direction_uncertainty_degrees"])
            ),
        }
        for point in points
    ]
    rounded = [
        {"x_px": int(point["x_px"]), "y_px": int(point["y_px"])}
        for point in points
    ]
    return _canonical_digest(raw), _canonical_digest(rounded)


def _validate_reference_outputs(
    gate: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    exclusions: Sequence[Mapping[str, object]],
    root: Path,
) -> str:
    """Cross-check independent private oracles without embedding their IDs."""
    outputs = gate["expected_outputs"]
    oracle = gate["coordinate_oracle"]
    if (outputs is None) != (oracle is None):
        raise FormatError(
            "coordinate_oracle and expected_outputs must both be present or both null"
        )
    if outputs is None:
        if gate["trust_level"] != "archive_split_only":
            raise FormatError(
                "reference_verified trust requires coordinate/output oracles"
            )
        return "UNVERIFIED_COORDINATES"
    if gate["trust_level"] != "reference_verified":
        raise FormatError("complete reference oracles require reference_verified trust")
    if not isinstance(outputs, dict) or not isinstance(oracle, dict):
        raise FormatError("reference output/oracle sections must be objects or null")

    included_ids = sorted(str(row["image_id"]) for row in rows)
    identity_fields = (
        "image_id",
        "source_member",
        "subject_id",
        "split",
        "group",
        "device",
        "capture",
        "finger_position",
        "n_minutiae",
        "n_unique_minutiae",
    )
    identities = [
        {key: str(row[key]) for key in identity_fields}
        for row in sorted(rows, key=lambda item: str(item["image_id"]))
    ]
    hull_identities = []
    for row in sorted(rows, key=lambda item: str(item["image_id"])):
        hull = cv2.imread(
            str(_resolve_private_artifact(root, row["hull_path"])),
            cv2.IMREAD_GRAYSCALE,
        )
        if hull is None:
            raise FormatError(f"cannot hash hull for {row['image_id']!r}")
        binary = (hull > 0).astype(np.uint8)
        hull_identities.append(
            {
                "image_id": str(row["image_id"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "pixels_sha256": hashlib.sha256(binary.tobytes()).hexdigest(),
            }
        )
    actual_outputs = {
        "built_count": len(rows),
        "excluded_count": len(exclusions),
        "included_ids_sha256": _canonical_digest(included_ids),
        "excluded": sorted(
            [
                {"image_id": str(item["image_id"]), "reason": str(item["reason"])}
                for item in exclusions
            ],
            key=lambda item: item["image_id"],
        ),
        "manifest_identity_sha256": _canonical_digest(identities),
        "all_hulls_sha256": _canonical_digest(hull_identities),
    }
    if outputs != actual_outputs:
        raise FormatError("built inclusion/exclusion identities disagree with reference")

    point_digests: Dict[str, Tuple[str, str]] = {
        str(row["image_id"]): _point_digests(root, row) for row in rows
    }
    all_points = [
        {
            "image_id": image_id,
            "raw_points_sha256": point_digests[image_id][0],
            "rounded_points_sha256": point_digests[image_id][1],
        }
        for image_id in sorted(point_digests)
    ]
    if not isinstance(oracle.get("source"), str) or not oracle["source"].strip():
        raise FormatError("coordinate oracle must name its independent source")
    checks = oracle.get("checks")
    if not isinstance(checks, list) or not checks:
        raise FormatError("coordinate oracle requires at least one identity check")
    seen = set()
    for check in checks:
        if not isinstance(check, dict) or set(check) != {
            "image_id",
            "raw_points_sha256",
            "rounded_points_sha256",
        }:
            raise FormatError("coordinate-oracle check has the wrong schema")
        image_id = str(check["image_id"])
        if image_id in seen or image_id not in point_digests:
            raise FormatError(f"invalid coordinate-oracle identity {image_id!r}")
        seen.add(image_id)
        if tuple(check[key] for key in ("raw_points_sha256", "rounded_points_sha256")) != point_digests[image_id]:
            raise FormatError(
                f"coordinate-oracle point digests disagree for {image_id!r}"
            )
    if oracle.get("all_points_sha256") != _canonical_digest(all_points):
        raise FormatError("full coordinate identity digest disagrees with reference")
    return "REFERENCE_VERIFIED"


def _validate_release_outputs(
    rows: Sequence[Mapping[str, object]],
    exclusions: Sequence[Mapping[str, object]],
    splits: Mapping[str, str],
) -> None:
    """Ensure a production output accounts for every official transaction."""
    row_ids = {str(row["image_id"]) for row in rows}
    exclusion_ids = {str(row["image_id"]) for row in exclusions}
    if len(row_ids) != len(rows) or len(exclusion_ids) != len(exclusions):
        raise FormatError("production rows or exclusions contain duplicate image IDs")
    if row_ids & exclusion_ids:
        raise FormatError("an image occurs in both the manifest and exclusions")
    if not rows:
        raise FormatError("production output contains no evaluable minutiae hulls")
    if len(rows) + len(exclusions) != REFERENCE_RECORDS:
        raise FormatError(
            f"production output accounts for {len(rows) + len(exclusions)} records; "
            f"expected {REFERENCE_RECORDS}"
        )
    items = [*rows, *exclusions]
    source_members = [str(item["source_member"]) for item in items]
    if len(source_members) != len(set(source_members)):
        raise FormatError("production output contains duplicate source members")
    metadata = [parse_member_metadata(member) for member in source_members]
    for item, parsed in zip(items, metadata):
        if str(item["image_id"]) != parsed.image_id:
            raise FormatError(
                f"output image_id {item['image_id']!r} disagrees with source "
                f"member identity {parsed.image_id!r}"
            )
    allowed_exclusions = {
        "fewer_than_three_unique_minutiae",
        "degenerate_minutiae_hull",
        "empty_minutiae_hull",
    }
    unexpected_reasons = {
        str(item["reason"]) for item in exclusions
    } - allowed_exclusions
    if unexpected_reasons:
        raise FormatError(
            f"production output has unsupported exclusion reasons: "
            f"{sorted(unexpected_reasons)}"
        )
    groups = Counter(item.group for item in metadata)
    captures = Counter(item.capture for item in metadata)
    if dict(groups) != dict(REFERENCE_GROUP_COUNTS):
        raise FormatError(
            f"production group counts {dict(groups)} != {dict(REFERENCE_GROUP_COUNTS)}"
        )
    if dict(captures) != dict(REFERENCE_CAPTURE_COUNTS):
        raise FormatError(
            f"production capture counts {dict(captures)} != "
            f"{dict(REFERENCE_CAPTURE_COUNTS)}"
        )
    release_subjects = {item.subject_id for item in metadata}
    if release_subjects != set(splits):
        missing = sorted(set(splits) - release_subjects)
        unexpected = sorted(release_subjects - set(splits))
        raise FormatError(
            "frozen split subjects disagree with the release: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    for row in rows:
        parsed = parse_member_metadata(str(row["source_member"]))
        expected = {
            "subject_id": parsed.subject_id,
            "split": splits[parsed.subject_id],
            "group": parsed.group,
            "device": parsed.device,
            "capture": parsed.capture,
            "finger_position": parsed.finger_position,
        }
        observed = {key: str(row[key]) for key in expected}
        if observed != expected:
            raise FormatError(
                f"manifest metadata for {parsed.image_id!r} disagrees with "
                f"source/split metadata: {observed} != {expected}"
            )


def _write_readme(
    path: Path,
    archive: Path,
    rows: Sequence[Mapping[str, object]],
    exclusions: Sequence[Mapping[str, object]],
    split_seed: object,
    verification_status: str,
) -> None:
    groups = Counter(str(row["group"]) for row in rows)
    captures = Counter(str(row["capture"]) for row in rows)
    splits = Counter(str(row["split"]) for row in rows)
    path.write_text(
        f"""# SD302g minutiae-region lower bounds

Generated from `{archive.name}`. This restricted output contains {len(rows):,}
decoded Type-14 images and convex hulls of expert Field 9.331 minutiae.

These hulls mark only a region that is certainly within the fingerprint. They
are not full foreground outlines, must not be used as supervised segmentation
labels, and support only the dedicated lower-bound evaluator.

- `images/`: decoded embedded Type-14 images.
- `hulls/`: 0/255 minutiae convex hulls at native image dimensions.
- `minutiae/`: physical coordinates, continuous pixel coordinates, explicit
  nearest-pixel raster indices, direction, type, and uncertainty per minutia.
- `manifest.csv`: private paths and group/device/capture/subject/split metadata.
- `exclusions.csv`: records with fewer than three unique, non-collinear points.
- `errors.csv`: fatal parse, association, integrity, or decoding failures.
- `provenance.json`: input, manifest, and private reference-gate hashes.

The frozen subject split seed is `{split_seed}`. Built-row counts are groups
{dict(sorted(groups.items()))}, captures {dict(sorted(captures.items()))}, and
splits {dict(sorted(splits.items()))}. There are {len(exclusions):,} hull
exclusions.

Coordinate verification status: **{verification_status}**. Results marked
`UNVERIFIED_COORDINATES` are candidate diagnostics only and are not paper-ready.

Coordinates use `pixel = EFS_lambda * ppi / 2540`. Continuous coordinates are
retained. Raster indices use the explicit implementation rule
`floor(pixel + 0.5)`; coordinates are never flipped, swapped, clipped, or
dilated. Production reference verification requires an independently prepared
private point-coordinate oracle.
""",
        encoding="utf-8",
    )


def _outside_repo(path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    try:
        path.relative_to(repo)
    except ValueError:
        return
    raise ValueError(f"private output must be outside the source repository: {path}")


def _existing_complete(out: Path) -> bool:
    required = (
        out / "images",
        out / "hulls",
        out / "minutiae",
        out / "manifest.csv",
        out / "exclusions.csv",
        out / "errors.csv",
        out / "provenance.json",
        out / "README.md",
    )
    return out.is_dir() and all(path.exists() for path in required)


def _read_manifest(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise FormatError("existing SD302g manifest schema mismatch")
        return [dict(row) for row in reader]


def _read_exclusions(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXCLUSION_COLUMNS:
            raise FormatError("existing SD302g exclusion schema mismatch")
        return [dict(row) for row in reader]


def _read_errors(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ERROR_COLUMNS:
            raise FormatError("existing SD302g error schema mismatch")
        return [dict(row) for row in reader]


def _read_provenance(path: Path) -> Mapping[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise FormatError("existing SD302g provenance is not an object")
    return payload


def _provenance(
    *,
    archive_path: Path,
    split_path: Path,
    reference_path: Optional[Path],
    manifest_path: Path,
    verification_status: str,
    bootstrap: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "schema": "cureblock.sd302g.build-provenance.v1",
        "protocol": "sd302g-minutiae-lower-bound-v2",
        "archive_sha256": _sha256_file(archive_path),
        "split_sha256": _sha256_file(split_path),
        "reference_gate_sha256": (
            None if reference_path is None else _sha256_file(reference_path)
        ),
        "manifest_sha256": _sha256_file(manifest_path),
        "verification_status": verification_status,
        "bootstrap": dict(bootstrap),
    }


def build(args: argparse.Namespace) -> None:
    archive_path = Path(args.archive).resolve()
    split_path = Path(args.splits).resolve()
    out = Path(args.out).resolve()
    reference_path = (
        None if not args.reference_json else Path(args.reference_json).resolve()
    )
    _outside_repo(out)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    split_seed, splits = _load_splits(split_path)
    if args.skip_release_validation:
        if reference_path is not None:
            raise ValueError(
                "--reference-json must not be combined with "
                "--skip-release-validation"
            )
        gate: Optional[Mapping[str, object]] = None
    else:
        if reference_path is None:
            raise ValueError(
                "production builds require --reference-json from restricted, "
                "independent release review"
            )
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        _outside_repo(reference_path)
        gate = _load_reference_gate(
            reference_path,
            archive_path=archive_path,
            split_path=split_path,
            split_seed=split_seed,
            splits=splits,
        )

    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise FormatError(f"ZIP CRC validation failed for {bad_member}")
        members = _select_members(
            archive.namelist(), validate_release=not args.skip_release_validation
        )
    if args.skip_release_validation and len(members) == REFERENCE_RECORDS:
        raise ValueError(
            "--skip-release-validation cannot be used on a release-sized archive"
        )
    if gate is not None:
        _validate_reference_inventory(gate, members)
        bootstrap = dict(gate["bootstrap"])
    else:
        # Synthetic-only runs inherit their fixture split seed. This value is
        # never presented as a production preregistration.
        bootstrap = {
            "seed": split_seed,
            "replicates": 1000,
            "unit": "subject",
            "interval": "percentile_95",
            "paired": True,
        }

    if _existing_complete(out):
        rows = _read_manifest(out / "manifest.csv")
        exclusions = _read_exclusions(out / "exclusions.csv")
        errors = _read_errors(out / "errors.csv")
        if errors:
            raise FormatError(
                f"existing output contains {len(errors)} fatal error row(s)"
            )
        _validate_artifacts(rows, out)
        if not args.skip_release_validation:
            _validate_release_outputs(rows, exclusions, splits)
            if gate is None:
                raise FormatError("production reference gate was not loaded")
            verification_status = _validate_reference_outputs(
                gate, rows, exclusions, out
            )
        else:
            verification_status = "SYNTHETIC_TEST_ONLY"
        expected_provenance = _provenance(
            archive_path=archive_path,
            split_path=split_path,
            reference_path=reference_path,
            manifest_path=out / "manifest.csv",
            verification_status=verification_status,
            bootstrap=bootstrap,
        )
        if _read_provenance(out / "provenance.json") != expected_provenance:
            raise FormatError(
                "existing output provenance disagrees with current inputs"
            )
        print(f"validated output already exists; leaving unchanged: {out}")
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output exists and is not a complete build: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.rmdir()
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))
    rows: List[Mapping[str, object]] = []
    exclusions: List[Mapping[str, object]] = []
    errors: List[Mapping[str, object]] = []
    started = time.monotonic()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for index, member in enumerate(members, start=1):
                metadata: Optional[MemberMetadata] = None
                try:
                    metadata = parse_member_metadata(member)
                    row = dict(
                        process_member(
                            member,
                            archive.read(member),
                            staging,
                            splits,
                            split_seed,
                        )
                    )
                    row.pop("_split_seed", None)
                    rows.append(row)
                except HullUnavailable as exc:
                    exclusions.append(
                        {
                            "image_id": metadata.image_id if metadata else "",
                            "source_member": member,
                            "reason": exc.reason,
                            "detail": str(exc),
                        }
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "source_member": member,
                            "stage": "transaction",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                if index % 250 == 0 or index == len(members):
                    print(
                        f"processed {index:,}/{len(members):,}; built={len(rows):,}, "
                        f"excluded={len(exclusions):,}, fatal={len(errors):,}",
                        flush=True,
                    )
        rows.sort(key=lambda row: str(row["image_id"]))
        exclusions.sort(key=lambda row: str(row["image_id"]))
        _write_csv(staging / "manifest.csv", rows, MANIFEST_COLUMNS)
        _write_csv(staging / "exclusions.csv", exclusions, EXCLUSION_COLUMNS)
        _write_csv(staging / "errors.csv", errors, ERROR_COLUMNS)
        if errors:
            raise FormatError(
                f"build produced {len(errors)} fatal error(s); inspect "
                f"{staging / 'errors.csv'}"
            )
        _validate_artifacts(rows, staging)
        if not args.skip_release_validation:
            _validate_release_outputs(rows, exclusions, splits)
            if gate is None:
                raise FormatError("production reference gate was not loaded")
            verification_status = _validate_reference_outputs(
                gate, rows, exclusions, staging
            )
        else:
            verification_status = "SYNTHETIC_TEST_ONLY"
        _write_json(
            staging / "provenance.json",
            _provenance(
                archive_path=archive_path,
                split_path=split_path,
                reference_path=reference_path,
                manifest_path=staging / "manifest.csv",
                verification_status=verification_status,
                bootstrap=bootstrap,
            ),
        )
        _write_readme(
            staging / "README.md",
            archive_path,
            rows,
            exclusions,
            split_seed,
            verification_status,
        )
        os.replace(staging, out)
        print(
            f"published {len(rows):,} hulls with {len(exclusions):,} exclusions "
            f"to {out} in {time.monotonic() - started:.1f}s"
        )
    except Exception:
        print(f"build failed; retained diagnostic staging at {staging}", file=sys.stderr)
        raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="official SD302g ZIP")
    parser.add_argument("out", help="restricted output directory outside the repository")
    parser.add_argument("--splits", required=True, help="frozen 136/24/40 subject split JSON")
    parser.add_argument(
        "--reference-json",
        help=(
            "restricted independent reference gate; required for production "
            "and forbidden for synthetic fixtures"
        ),
    )
    parser.add_argument(
        "--skip-release-validation",
        action="store_true",
        help="synthetic fixtures only; skip official 2,380-record aggregate checks",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    build(parse_args(argv))


if __name__ == "__main__":
    main()
