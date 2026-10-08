#!/usr/bin/env python3
"""Build SD302h examiner-ROI and ridge-quality binary masks.

This parser intentionally implements only the traditional ANSI/NIST-ITL fields
needed for SD302h.  It is fail-closed: malformed records, unsupported units,
ambiguous image associations, and unexpected reference statistics are reported
instead of being guessed around.

Normative format references:

* NIST SP 500-290e3, ANSI/NIST-ITL 1-2011 Update:2015 field definitions and
  the applicable EFS annex.
* NIST TN 2367, sections 2.1--2.2 and 3.1--3.3 (SD302h release facts).

Real images, generated masks, summaries, and overlays must be written only to
the restricted output directory supplied by ``--out``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2
import numpy as np


FS = b"\x1c"
GS = b"\x1d"
RS = b"\x1e"
US = b"\x1f"

REFERENCE_COUNTS: Mapping[str, int] = {
    "NOVALUE": 4636,
    "VALUE": 3815,
    "LIMITED": 1639,
    "NONPRINT": 2,
}
REFERENCE_ROI_MASKS = 10_092
REFERENCE_QUALITY_MASKS = 10_024
REFERENCE_QUALITY_EXCLUSIONS: Mapping[str, int] = {
    "missing_quality_map": 67,
    "ragged_quality_map": 1,
}
EAA_CODES = frozenset(REFERENCE_COUNTS)
ERROR_COLUMNS = ("member", "record_index", "stage", "error")
QUALITY_EXCLUSION_COLUMNS = ("id", "member", "record_index", "reason", "error")
SUMMARY_COLUMNS = (
    "id",
    "image_name",
    "subject_id",
    "examiner_value",
    "ppi",
    "width",
    "height",
    "roi_area_px",
    "quality_area_px",
    "roi_quality_iou",
)

_RECORD_PREFIX = re.compile(br"(?P<type>[1-9][0-9]?)\.001:(?P<length>[0-9]+)")
_FIELD_PREFIX = re.compile(br"^(?P<type>[1-9][0-9]?)\.(?P<number>[0-9]{3}):", re.DOTALL)
_PATH_VERTEX = re.compile(r"(?P<x>[0-9]+),(?P<y>[0-9]+)")
_PPI_IN_NAME = re.compile(r"(?:^|_)(?P<ppi>[0-9]+)PPI(?:_|$)", re.IGNORECASE)
_SUBJECT_IN_NAME = re.compile(r"^(?P<subject>[0-9]{8})(?:_|$)")


class FormatError(ValueError):
    """Raised when an ANSI/NIST value cannot be interpreted unambiguously."""


class MissingQualityMapError(FormatError):
    """Raised when an otherwise usable impression has no ridge-quality fields."""


class RaggedQualityMapError(FormatError):
    """Raised when the release supplies a non-rectangular ridge-quality grid."""


@dataclass(frozen=True)
class TaggedRecord:
    """One length-delimited traditional-encoding logical record."""

    record_type: int
    raw: bytes
    fields: Mapping[str, Tuple[bytes, ...]]

    def one(self, tag: str, *, required: bool = True) -> Optional[bytes]:
        """Return a field that must occur at most once."""
        values = self.fields.get(tag, ())
        if not values:
            if required:
                raise FormatError(f"missing required field {tag}")
            return None
        if len(values) != 1:
            raise FormatError(f"field {tag} occurs {len(values)} times; expected one")
        return values[0]


@dataclass(frozen=True)
class ImageRecord:
    """Metadata and compressed payload from a Type-13 or Type-14 record."""

    record_type: int
    idc: str
    width: int
    height: int
    ppi: float
    compression: str
    payload: bytes


@dataclass(frozen=True)
class ROI:
    """EFS region-of-interest geometry in lambda (0.01 mm) units."""

    width_units: int
    height_units: int
    x_offset_units: int
    y_offset_units: int
    vertices: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class QualityMap:
    """Decoded EFS ridge-quality grid."""

    grid_size_units: int
    values: np.ndarray


@dataclass(frozen=True)
class BuiltImpression:
    """Internal provenance needed to create deterministic QA overlays."""

    impression_id: str
    member: str
    record_index: int
    idc: str
    roi_path: Path
    quality_path: Path


def _ascii(value: bytes, label: str) -> str:
    """Decode a normative ASCII field or fail with context."""
    try:
        return value.decode("ascii")
    except UnicodeDecodeError as exc:
        raise FormatError(f"{label} is not ASCII") from exc


def _positive_int(value: bytes, label: str, *, allow_zero: bool = False) -> int:
    """Parse a canonical non-negative decimal integer."""
    text = _ascii(value, label)
    if not re.fullmatch(r"[0-9]+", text):
        raise FormatError(f"{label} is not a decimal integer: {text!r}")
    number = int(text)
    if number < 0 or (number == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise FormatError(f"{label} must be {qualifier}, got {number}")
    return number


def iter_logical_records(transaction: bytes) -> Iterator[Tuple[int, bytes]]:
    """Yield ANSI/NIST logical records using each record's declared byte length.

    Splitting a complete transaction on FS is unsafe because binary image data may
    contain that byte.  The ``*.001`` logical-record-length is therefore the only
    boundary used here; the final FS is then independently checked.
    """
    offset = 0
    index = 0
    while offset < len(transaction):
        match = _RECORD_PREFIX.match(transaction, offset)
        if match is None:
            preview = transaction[offset : offset + 32]
            raise FormatError(f"record {index}: invalid prefix at byte {offset}: {preview!r}")
        declared = int(match.group("length"))
        if declared <= 0:
            raise FormatError(f"record {index}: non-positive declared length {declared}")
        end = offset + declared
        if end > len(transaction):
            raise FormatError(
                f"record {index}: declared length {declared} exceeds remaining "
                f"{len(transaction) - offset} bytes"
            )
        raw = transaction[offset:end]
        if not raw.endswith(FS):
            raise FormatError(f"record {index}: declared record does not end in FS")
        yield int(match.group("type")), raw
        offset = end
        index += 1
    if offset != len(transaction):  # defensive; loop construction should guarantee this
        raise FormatError(f"transaction has {len(transaction) - offset} trailing bytes")


def parse_tagged_record(record_type: int, raw: bytes) -> TaggedRecord:
    """Parse the tagged fields of one traditional-encoding logical record.

    Type-13/14 image payloads are terminal ``*.999`` fields.  They are isolated
    before GS splitting so delimiter-looking bytes inside compressed imagery cannot
    be misinterpreted as fields.
    """
    if not raw.endswith(FS):
        raise FormatError("logical record is missing terminal FS")
    body = raw[:-1]
    terminal_tag = f"{record_type}.999:".encode("ascii")
    terminal_marker = GS + terminal_tag
    terminal_pos = body.find(terminal_marker)
    terminal_value: Optional[bytes] = None
    if terminal_pos >= 0:
        textual = body[:terminal_pos]
        terminal_value = body[terminal_pos + len(terminal_marker) :]
    elif body.startswith(terminal_tag):
        textual = b""
        terminal_value = body[len(terminal_tag) :]
    else:
        textual = body

    fields: MutableMapping[str, List[bytes]] = defaultdict(list)
    if textual:
        for encoded in textual.split(GS):
            match = _FIELD_PREFIX.match(encoded)
            if match is None:
                raise FormatError(f"record type {record_type}: malformed tagged field {encoded[:40]!r}")
            actual_type = int(match.group("type"))
            if actual_type != record_type:
                raise FormatError(
                    f"record type {record_type}: contains field for record type {actual_type}"
                )
            tag = f"{actual_type}.{int(match.group('number')):03d}"
            fields[tag].append(encoded[match.end() :])
    if terminal_value is not None:
        fields[f"{record_type}.999"].append(terminal_value)

    parsed = TaggedRecord(
        record_type=record_type,
        raw=raw,
        fields={key: tuple(values) for key, values in fields.items()},
    )
    declared = _positive_int(parsed.one(f"{record_type}.001"), f"{record_type}.001")
    if declared != len(raw):
        raise FormatError(
            f"record type {record_type}: declared length {declared} != actual {len(raw)}"
        )
    return parsed


def parse_transaction(transaction: bytes) -> Tuple[TaggedRecord, ...]:
    """Parse every logical record in one LFFS transaction."""
    return tuple(parse_tagged_record(kind, raw) for kind, raw in iter_logical_records(transaction))


def _field_items(value: bytes) -> List[bytes]:
    """Return information items, retaining empty optional positions."""
    return value.split(US)


def parse_image_record(record: TaggedRecord, image_name: str) -> ImageRecord:
    """Extract dimensions, sampling rate, compression, and bytes from Type 13/14."""
    if record.record_type not in (13, 14):
        raise FormatError(f"expected Type-13/14 image record, got Type-{record.record_type}")
    prefix = str(record.record_type)
    idc = _ascii(record.one(f"{prefix}.002"), f"{prefix}.002")
    width = _positive_int(record.one(f"{prefix}.006"), f"{prefix}.006")
    height = _positive_int(record.one(f"{prefix}.007"), f"{prefix}.007")
    scale_units = _positive_int(record.one(f"{prefix}.008"), f"{prefix}.008")
    horizontal_scale = _positive_int(record.one(f"{prefix}.009"), f"{prefix}.009")
    vertical_scale = _positive_int(record.one(f"{prefix}.010"), f"{prefix}.010")
    if horizontal_scale != vertical_scale:
        raise FormatError(
            f"{image_name}: unequal horizontal/vertical sampling rates "
            f"{horizontal_scale}/{vertical_scale}"
        )
    if scale_units == 1:  # pixels per inch
        ppi = float(horizontal_scale)
    elif scale_units == 2:  # pixels per centimetre
        ppi = float(horizontal_scale) * 2.54
    else:
        raise FormatError(f"{image_name}: unsupported scale-units code {scale_units}")

    name_match = _PPI_IN_NAME.search(image_name)
    if name_match is not None:
        named_ppi = float(name_match.group("ppi"))
        # A pixels/cm field is integer-valued and can therefore differ slightly.
        if not math.isclose(ppi, named_ppi, rel_tol=0.0, abs_tol=1.0):
            raise FormatError(
                f"{image_name}: metadata sampling rate {ppi:g} ppi disagrees "
                f"with filename {named_ppi:g} ppi"
            )

    return ImageRecord(
        record_type=record.record_type,
        idc=idc,
        width=width,
        height=height,
        ppi=ppi,
        compression=_ascii(record.one(f"{prefix}.011"), f"{prefix}.011"),
        payload=record.one(f"{prefix}.999"),
    )


def parse_roi(value: bytes) -> ROI:
    """Parse Field 9.300 (EWI, EHI, optional EHO, EVO, ROP)."""
    items = _field_items(value)
    if len(items) < 2 or len(items) > 5:
        raise FormatError(f"9.300 must contain 2--5 information items, got {len(items)}")
    width = _positive_int(items[0], "9.300 EWI")
    height = _positive_int(items[1], "9.300 EHI")
    x_offset = _positive_int(items[2], "9.300 EHO", allow_zero=True) if len(items) >= 3 and items[2] else 0
    y_offset = _positive_int(items[3], "9.300 EVO", allow_zero=True) if len(items) >= 4 and items[3] else 0
    vertices: Tuple[Tuple[int, int], ...] = ()
    if len(items) == 5 and items[4]:
        path = _ascii(items[4], "9.300 ROP")
        parsed_vertices = tuple((int(m.group("x")), int(m.group("y"))) for m in _PATH_VERTEX.finditer(path))
        canonical = "-".join(f"{x},{y}" for x, y in parsed_vertices)
        if canonical != path:
            raise FormatError(f"9.300 ROP has unsupported path syntax: {path!r}")
        if len(parsed_vertices) < 3:
            raise FormatError("9.300 ROP polygon has fewer than three vertices")
        for x, y in parsed_vertices:
            if x > width or y > height:
                raise FormatError(
                    f"9.300 ROP vertex {(x, y)} exceeds ROI dimensions {(width, height)}"
                )
        vertices = parsed_vertices
    return ROI(width, height, x_offset, y_offset, vertices)


def parse_measurement_units(record: TaggedRecord) -> str:
    """Validate EFS measurement units.

    SD302h follows the legacy lambda convention.  In SP 500-290e3 the field is
    normally absent; omission means 0.01 mm.  Any explicit value is rejected here
    rather than assigned an undocumented interpretation.
    """
    value = record.one("9.304", required=False)
    if value is None or value == b"":
        return "lambda_0.01mm"
    raise FormatError(
        f"explicit Field 9.304 measurement units are unsupported by this "
        f"SD302h-specific parser: {_ascii(value, '9.304')!r}"
    )


def _decode_rle_row(text: str) -> List[int]:
    """Decode the RLE representation specified by SP 500-290e3 Table 46."""
    values: List[int] = []
    position = 0
    while position < len(text):
        start = position
        while position < len(text) and text[position].isdigit():
            position += 1
        count_text = text[start:position]
        if position >= len(text) or text[position] not in "ABCDEF":
            raise FormatError(f"invalid RLE ridge-quality row near {text[start:]!r}")
        count = int(count_text) if count_text else 1
        if count <= 0:
            raise FormatError("RLE ridge-quality count must be positive")
        values.extend([ord(text[position]) - ord("A")] * count)
        position += 1
    return values


def _validate_rqf_pair(value: bytes, label: str = "9.309") -> Tuple[bytes, bytes]:
    """Validate and return one GSZ/RDF pair without interpreting map geometry."""
    items = _field_items(value)
    if len(items) != 2:
        raise FormatError(f"{label} must contain GSZ and RDF, got {len(items)} items")
    _positive_int(items[0], f"{label} GSZ")
    data_format = _ascii(items[1], f"{label} RDF")
    if data_format not in {"UNC", "RLE"}:
        raise FormatError(f"unsupported {label} RDF {data_format!r}")
    return items[0], items[1]


def resolve_quality_fields(record: TaggedRecord) -> Tuple[bytes, bytes]:
    """Resolve the measured SD302h release encodings of Fields 9.308/9.309.

    Most records contain one field of each type. Two deterministic release
    defects are also present and are normalized narrowly:

    * Some 9.309 values concatenate the exact same GSZ/RDF pair twice.
    * In four records, a second field tagged 9.308 contains more map rows and
      then an RS followed by the GSZ/RDF pair that should have been 9.309.

    Any non-identical duplicate, different field cardinality, malformed suffix,
    or genuinely absent quality map remains an error. In particular, records
    without 9.308/9.309 are never converted to all-background masks.
    """
    rqm_values = record.fields.get("9.308", ())
    rqf_values = record.fields.get("9.309", ())

    if len(rqm_values) == 1 and len(rqf_values) == 1:
        rqf_items = _field_items(rqf_values[0])
        if len(rqf_items) == 2:
            _validate_rqf_pair(rqf_values[0])
            return rqm_values[0], rqf_values[0]
        if len(rqf_items) == 4:
            first = tuple(rqf_items[:2])
            second = tuple(rqf_items[2:])
            if first != second:
                raise FormatError("9.309 contains two non-identical GSZ/RDF pairs")
            normalized = US.join(first)
            _validate_rqf_pair(normalized, "normalized 9.309")
            return rqm_values[0], normalized
        raise FormatError(f"9.309 must contain 2 items or an exact duplicated pair, got {len(rqf_items)}")

    if len(rqm_values) == 2 and len(rqf_values) == 0:
        second_map_field = rqm_values[1]
        if RS not in second_map_field:
            raise FormatError("second 9.308 has no RS-delimited RQF suffix")
        additional_rows, rqf_suffix = second_map_field.rsplit(RS, 1)
        if not additional_rows:
            raise FormatError("second 9.308 has no map rows before its RQF suffix")
        _validate_rqf_pair(rqf_suffix, "9.308-carried RQF")
        # The same four release records also inject that exact RQF pair, plus
        # one US delimiter, into one row of the first map field. Require one
        # and only one byte-exact marker at an RS row boundary, then remove
        # only its metadata bytes. This is deterministic repair of duplicated
        # metadata, not heuristic recovery of arbitrary malformed content.
        embedded_marker = rqf_suffix + US
        first_map_field = rqm_values[0]
        if first_map_field.count(embedded_marker) != 1:
            raise FormatError(
                "first 9.308 must contain exactly one embedded RQF marker "
                "matching the second-field suffix"
            )
        marker_offset = first_map_field.index(embedded_marker)
        if marker_offset == 0 or first_map_field[marker_offset - 1 : marker_offset] != RS:
            raise FormatError("embedded RQF marker is not at a 9.308 row boundary")
        after_marker = marker_offset + len(embedded_marker)
        if after_marker >= len(first_map_field) or first_map_field[after_marker : after_marker + 1] == RS:
            raise FormatError("embedded RQF marker is not followed by map-row data")
        normalized_first = (
            first_map_field[:marker_offset] + first_map_field[after_marker:]
        )
        return normalized_first + RS + additional_rows, rqf_suffix

    if not rqm_values and not rqf_values:
        raise MissingQualityMapError(
            "record has no 9.308 ridge-quality map and no 9.309 format"
        )
    raise FormatError(
        f"unsupported ridge-quality field cardinality: "
        f"9.308={len(rqm_values)}, 9.309={len(rqf_values)}"
    )


def parse_quality_map(rqm_value: bytes, rqf_value: bytes, roi: ROI) -> QualityMap:
    """Decode Fields 9.308/9.309 into a rectangular uint8 grid.

    SP 500-290e3 describes a nominal row/column count of ``ceil(EHI/GSZ)``
    and ``ceil(EWI/GSZ)``.  The released SD302h transactions do not follow
    that count: measured grids include, for example, 9x9 and 13x13 maps for
    650x650-lambda ROIs with GSZ=20.  The authoritative information in those
    records is therefore the rectangular row data itself.  Geometry is applied
    later by nearest-neighbour expansion across the complete ROI, which also
    reproduces the independently validated reference IoUs.
    """
    rqf = _field_items(rqf_value)
    _validate_rqf_pair(rqf_value)
    grid_size = _positive_int(rqf[0], "9.309 GSZ")
    data_format = _ascii(rqf[1], "9.309 RDF")
    if data_format not in {"UNC", "RLE"}:
        raise FormatError(f"unsupported 9.309 RDF {data_format!r}")
    rows = rqm_value.split(RS)
    if not rows or any(not encoded for encoded in rows):
        raise FormatError("9.308 must contain one or more non-empty rows")
    decoded: List[List[int]] = []
    expected_cols: Optional[int] = None
    for row_index, encoded in enumerate(rows):
        text = _ascii(encoded, f"9.308 row {row_index}")
        if data_format == "UNC":
            if not re.fullmatch(r"[0-5]+", text):
                raise FormatError(f"9.308 UNC row {row_index} contains a non-quality code")
            values = [int(ch) for ch in text]
        else:
            values = _decode_rle_row(text)
        if not values:
            raise FormatError(f"9.308 row {row_index} is empty")
        if expected_cols is None:
            expected_cols = len(values)
        elif len(values) != expected_cols:
            raise RaggedQualityMapError(
                f"9.308 is ragged: row {row_index} has {len(values)} cells, "
                f"first row has {expected_cols}"
            )
        decoded.append(values)
    return QualityMap(grid_size_units=grid_size, values=np.asarray(decoded, dtype=np.uint8))


def parse_examiner_value(value: bytes) -> str:
    """Return the first information item of Field 9.353 without retaining PII."""
    items = _field_items(value)
    code = _ascii(items[0], "9.353 AAV") if items else ""
    if code not in EAA_CODES:
        raise FormatError(f"unsupported 9.353 examiner assessment code {code!r}")
    return code


def _unit_to_pixel(value: int, ppi: float) -> float:
    """Convert lambda units (0.01 mm) to continuous image-pixel coordinates."""
    return float(value) * 0.01 * ppi / 25.4


def rasterize_roi(roi: ROI, width: int, height: int, ppi: float) -> np.ndarray:
    """Rasterise the ROI rectangle or optional polygon at native image size."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if roi.vertices:
        vertices = np.asarray(
            [
                (
                    round(_unit_to_pixel(roi.x_offset_units + x, ppi)),
                    round(_unit_to_pixel(roi.y_offset_units + y, ppi)),
                )
                for x, y in roi.vertices
            ],
            dtype=np.int32,
        )
        vertices[:, 0] = np.clip(vertices[:, 0], 0, width - 1)
        vertices[:, 1] = np.clip(vertices[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [vertices], 255)
    else:
        x0 = math.floor(_unit_to_pixel(roi.x_offset_units, ppi))
        y0 = math.floor(_unit_to_pixel(roi.y_offset_units, ppi))
        x1 = math.ceil(_unit_to_pixel(roi.x_offset_units + roi.width_units, ppi))
        y1 = math.ceil(_unit_to_pixel(roi.y_offset_units + roi.height_units, ppi))
        x0, x1 = max(0, x0), min(width, x1)
        y0, y1 = max(0, y0), min(height, y1)
        if x0 >= x1 or y0 >= y1:
            raise FormatError("9.300 ROI rectangle lies outside the image")
        mask[y0:y1, x0:x1] = 255
    if not np.any(mask):
        raise FormatError("9.300 ROI rasterised to an empty mask")
    return mask


def rasterize_quality(
    quality: QualityMap,
    roi: ROI,
    width: int,
    height: int,
    ppi: float,
    threshold: int,
) -> np.ndarray:
    """Expand a quality grid over its complete ROI using nearest neighbour.

    This is the measured SD302h convention. It deliberately does not use GSZ
    to place fixed-size cells because doing so leaves large parts of some
    released 256-pixel images uncovered and disagrees with the supplied laptop
    validation. The ROI offset is still applied, so the implementation remains
    correct for non-zero-offset records even though all measured SD302h offsets
    are zero.
    """
    if threshold < 0 or threshold >= 5:
        raise ValueError("quality threshold must be in [0, 4]")
    x0 = round(_unit_to_pixel(roi.x_offset_units, ppi))
    y0 = round(_unit_to_pixel(roi.y_offset_units, ppi))
    roi_width_px = round(_unit_to_pixel(roi.width_units, ppi))
    roi_height_px = round(_unit_to_pixel(roi.height_units, ppi))
    if roi_width_px <= 0 or roi_height_px <= 0:
        raise FormatError("9.300 ROI has no pixel extent for the ridge-quality map")
    expanded = cv2.resize(
        (quality.values > threshold).astype(np.uint8) * 255,
        (roi_width_px, roi_height_px),
        interpolation=cv2.INTER_NEAREST,
    )
    x_start, y_start = max(0, x0), max(0, y0)
    x_end, y_end = min(width, x0 + roi_width_px), min(height, y0 + roi_height_px)
    if x_start >= x_end or y_start >= y_end:
        raise FormatError("9.308 quality-map ROI lies outside the image")
    source_x = x_start - x0
    source_y = y_start - y0
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y_start:y_end, x_start:x_end] = expanded[
        source_y : source_y + (y_end - y_start),
        source_x : source_x + (x_end - x_start),
    ]
    return mask


def binary_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Compute foreground IoU, treating two empty masks as 1."""
    a, b = left > 0, right > 0
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def _write_png(path: Path, image: np.ndarray) -> None:
    """Write a PNG atomically and verify that OpenCV accepted it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.png")
    if not cv2.imwrite(str(temporary), image):
        raise OSError(f"OpenCV failed to write {temporary}")
    os.replace(temporary, path)


def _select_archive_members(names: Sequence[str], variant: str) -> List[str]:
    """Select one documented SD302h media branch, never duplicate variants."""
    requested = tuple(variant.lower().split("_"))
    candidates: List[str] = []
    for name in names:
        if not name.lower().endswith(".lffs"):
            continue
        components = {part.lower() for part in PurePosixPath(name).parts}
        if all(token in components for token in requested):
            candidates.append(name)
    if not candidates:
        raise FormatError(
            f"archive has no .lffs members in documented variant {variant!r}; "
            "refusing to choose another branch implicitly"
        )
    duplicate_stems = [stem for stem, count in Counter(PurePosixPath(n).stem for n in candidates).items() if count > 1]
    if duplicate_stems:
        raise FormatError(
            f"variant {variant!r} contains duplicate LFFS stems, e.g. {duplicate_stems[:5]}"
        )
    return sorted(candidates)


def _subject_from_name(image_name: str) -> str:
    match = _SUBJECT_IN_NAME.match(image_name)
    if match is None:
        raise FormatError(f"cannot derive SD302 subject id from {image_name!r}")
    return match.group("subject")


def _image_records(records: Sequence[TaggedRecord], image_name: str) -> Dict[str, ImageRecord]:
    images: Dict[str, ImageRecord] = {}
    for record in records:
        if record.record_type not in (13, 14):
            continue
        parsed = parse_image_record(record, image_name)
        if parsed.idc in images:
            raise FormatError(f"multiple embedded images have IDC {parsed.idc!r}")
        images[parsed.idc] = parsed
    if not images:
        raise FormatError("transaction contains no Type-13/14 image record")
    return images


def process_member(
    member: str,
    transaction: bytes,
    staging: Path,
    quality_threshold: int,
) -> Tuple[List[Dict[str, object]], List[BuiltImpression], List[Dict[str, str]]]:
    """Build every ROI and every available quality mask in one transaction.

    Field 9.300 and its associated image are mandatory. A missing or malformed
    ridge-quality map does not invalidate that examiner ROI: it is recorded in
    ``quality_exclusions.csv`` and omitted only from Table A-prime.
    """
    image_name = PurePosixPath(member).stem
    records = parse_transaction(transaction)
    images = _image_records(records, image_name)
    type9 = [record for record in records if record.record_type == 9]
    if not type9:
        raise FormatError("transaction contains no Type-9 EFS record")
    summary_rows: List[Dict[str, object]] = []
    built: List[BuiltImpression] = []
    quality_exclusions: List[Dict[str, str]] = []
    for occurrence, record in enumerate(type9, start=1):
        parse_measurement_units(record)
        idc = _ascii(record.one("9.002"), "9.002")
        if idc not in images:
            raise FormatError(f"Type-9 IDC {idc!r} has no associated Type-13/14 image")
        image = images[idc]
        roi = parse_roi(record.one("9.300"))
        examiner_value = parse_examiner_value(record.one("9.353"))
        subject_id = _subject_from_name(image_name)
        # Preserve the release filename as the identifier for the common
        # one-impression case. Only disambiguate files that contain multiple
        # Type-9 feature sets, as required by the SD302h naming contract.
        impression_id = image_name if len(type9) == 1 else f"{image_name}_{occurrence}"
        roi_mask = rasterize_roi(roi, image.width, image.height, image.ppi)
        roi_path = staging / "gt_roi" / f"{impression_id}.png"
        _write_png(roi_path, roi_mask)

        quality_mask: Optional[np.ndarray] = None
        quality_error: Optional[FormatError] = None
        try:
            rqm, rqf = resolve_quality_fields(record)
            quality = parse_quality_map(rqm, rqf, roi)
            quality_mask = rasterize_quality(
                quality,
                roi,
                image.width,
                image.height,
                image.ppi,
                quality_threshold,
            )
        except FormatError as exc:
            quality_error = exc

        quality_area: object = ""
        roi_quality_iou: object = ""
        if quality_error is None:
            assert quality_mask is not None
            quality_path = staging / "gt_quality" / f"{impression_id}.png"
            _write_png(quality_path, quality_mask)
            quality_area = int(np.count_nonzero(quality_mask))
            roi_quality_iou = binary_iou(roi_mask, quality_mask)
            built.append(
                BuiltImpression(
                    impression_id, member, occurrence, idc, roi_path, quality_path
                )
            )
        else:
            if isinstance(quality_error, MissingQualityMapError):
                reason = "missing_quality_map"
            elif isinstance(quality_error, RaggedQualityMapError):
                reason = "ragged_quality_map"
            else:
                reason = "invalid_quality_map"
            quality_exclusions.append(
                {
                    "id": impression_id,
                    "member": member,
                    "record_index": str(occurrence),
                    "reason": reason,
                    "error": f"{type(quality_error).__name__}: {quality_error}",
                }
            )

        summary_rows.append(
            {
                "id": impression_id,
                "image_name": image_name,
                "subject_id": subject_id,
                "examiner_value": examiner_value,
                "ppi": f"{image.ppi:g}",
                "width": image.width,
                "height": image.height,
                "roi_area_px": int(np.count_nonzero(roi_mask)),
                "quality_area_px": quality_area,
                "roi_quality_iou": roi_quality_iou,
            }
        )
    return summary_rows, built, quality_exclusions


def _decode_associated_image(transaction: bytes, idc: str, image_name: str) -> np.ndarray:
    records = parse_transaction(transaction)
    images = _image_records(records, image_name)
    if idc not in images:
        raise FormatError(f"overlay IDC {idc!r} has no image")
    image_record = images[idc]
    image = cv2.imdecode(np.frombuffer(image_record.payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FormatError(
            f"OpenCV cannot decode embedded image (compression {image_record.compression!r})"
        )
    if image.shape != (image_record.height, image_record.width):
        raise FormatError(
            f"decoded image shape {image.shape} != metadata "
            f"{(image_record.height, image_record.width)}"
        )
    return image


def _contour_overlay(image: np.ndarray, roi: np.ndarray, quality: np.ndarray) -> np.ndarray:
    """Draw ROI (green) and usable-quality (magenta) contours for private QA."""
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    roi_contours, _ = cv2.findContours((roi > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quality_contours, _ = cv2.findContours(
        (quality > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(canvas, roi_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(canvas, quality_contours, -1, (255, 0, 255), 2)
    return canvas


def _write_overlays(
    archive: zipfile.ZipFile,
    built: Sequence[BuiltImpression],
    staging: Path,
    count: int,
) -> List[str]:
    """Write a best-effort deterministic private QA sample.

    Overlays are review aids, not ground truth. Failure to decode an embedded
    image for this optional sample must not discard otherwise validated masks.
    """
    ranked = sorted(
        built,
        key=lambda item: hashlib.sha256(item.impression_id.encode("utf-8")).digest(),
    )[: min(count, len(built))]
    warnings: List[str] = []
    transaction_cache: Dict[str, bytes] = {}
    for item in ranked:
        try:
            transaction = transaction_cache.setdefault(item.member, archive.read(item.member))
            image_name = PurePosixPath(item.member).stem
            image = _decode_associated_image(transaction, item.idc, image_name)
            roi = cv2.imread(str(item.roi_path), cv2.IMREAD_GRAYSCALE)
            quality = cv2.imread(str(item.quality_path), cv2.IMREAD_GRAYSCALE)
            if roi is None or quality is None:
                raise OSError("could not reread a generated mask")
            _write_png(staging / "overlays" / f"{item.impression_id}.png", _contour_overlay(image, roi, quality))
        except Exception as exc:
            warnings.append(
                f"{item.member} record {item.record_index}: "
                f"{type(exc).__name__}: {exc}"
            )
    return warnings


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _load_reference_low_iou(path: Path) -> Mapping[str, float]:
    """Load restricted per-image validation values without embedding them in git."""
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise FormatError("low-IoU reference JSON must be a non-empty object")
    parsed: Dict[str, float] = {}
    for impression_id, value in raw.items():
        if not isinstance(impression_id, str) or not impression_id:
            raise FormatError("low-IoU reference IDs must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FormatError(f"low-IoU reference for {impression_id!r} is not numeric")
        parsed[impression_id] = float(value)
    return parsed


def _load_quality_exclusion_reference(path: Path) -> Mapping[str, str]:
    """Load the restricted expected exclusion IDs and reason codes."""
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise FormatError("quality-exclusion reference JSON must be a non-empty object")
    parsed: Dict[str, str] = {}
    allowed = frozenset(REFERENCE_QUALITY_EXCLUSIONS)
    for impression_id, reason in raw.items():
        if not isinstance(impression_id, str) or not impression_id:
            raise FormatError("quality-exclusion reference IDs must be non-empty strings")
        if not isinstance(reason, str) or reason not in allowed:
            raise FormatError(
                f"quality-exclusion reason for {impression_id!r} must be one of "
                f"{sorted(allowed)}, got {reason!r}"
            )
        parsed[impression_id] = reason
    return parsed


def _validate_artifacts(
    rows: Sequence[Mapping[str, object]],
    quality_exclusions: Sequence[Mapping[str, object]],
    staging: Path,
    overlay_count: int,
) -> None:
    """Validate mask/CSV identity invariants for production and synthetic runs."""
    row_ids = [str(row["id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise FormatError("summary.csv contains duplicate impression IDs")
    exclusion_ids = [str(row["id"]) for row in quality_exclusions]
    if len(exclusion_ids) != len(set(exclusion_ids)):
        raise FormatError("quality_exclusions.csv contains duplicate impression IDs")
    unknown_exclusions = set(exclusion_ids) - set(row_ids)
    if unknown_exclusions:
        raise FormatError(
            f"quality exclusions are absent from summary.csv: {sorted(unknown_exclusions)[:5]}"
        )

    excluded = set(exclusion_ids)
    expected_quality = set(row_ids) - excluded
    for row in rows:
        impression_id = str(row["id"])
        quality_values = (row["quality_area_px"], row["roi_quality_iou"])
        if impression_id in excluded and any(value not in ("", None) for value in quality_values):
            raise FormatError(f"excluded impression {impression_id!r} has quality metrics")
        if impression_id not in excluded and any(value in ("", None) for value in quality_values):
            raise FormatError(f"quality-valid impression {impression_id!r} lacks quality metrics")

    roi_paths = list((staging / "gt_roi").glob("*.png"))
    quality_paths = list((staging / "gt_quality").glob("*.png"))
    roi_ids = {path.stem for path in roi_paths}
    quality_ids = {path.stem for path in quality_paths}
    if len(roi_paths) != len(roi_ids) or roi_ids != set(row_ids):
        raise FormatError(
            f"ROI mask identities disagree with summary: files={len(roi_paths)}, "
            f"unique={len(roi_ids)}, rows={len(row_ids)}"
        )
    if len(quality_paths) != len(quality_ids) or quality_ids != expected_quality:
        raise FormatError(
            f"quality mask identities disagree with summary/exclusions: "
            f"files={len(quality_paths)}, unique={len(quality_ids)}, "
            f"expected={len(expected_quality)}"
        )
    actual_overlays = sum(1 for _ in (staging / "overlays").glob("*.png"))
    expected_overlays = min(overlay_count, len(expected_quality))
    if actual_overlays > expected_overlays:
        raise FormatError(
            f"overlay count exceeds requested sample: got {actual_overlays}, "
            f"maximum {expected_overlays}"
        )


def _validate_reference(
    rows: Sequence[Mapping[str, object]],
    quality_exclusions: Sequence[Mapping[str, object]],
    reference_low_iou: Mapping[str, float],
    reference_quality_exclusions: Mapping[str, str],
) -> None:
    """Enforce the predeclared release aggregates and restricted ID references."""
    if len(rows) != REFERENCE_ROI_MASKS:
        raise FormatError(
            f"ROI impression count mismatch: got {len(rows)}, expected {REFERENCE_ROI_MASKS}"
        )
    counts = Counter(str(row["examiner_value"]) for row in rows)
    if dict(counts) != dict(REFERENCE_COUNTS):
        raise FormatError(
            f"examiner-value counts disagree with NIST TN 2367: got {dict(counts)}, "
            f"expected {dict(REFERENCE_COUNTS)}"
        )
    exclusion_counts = Counter(str(row["reason"]) for row in quality_exclusions)
    if dict(exclusion_counts) != dict(REFERENCE_QUALITY_EXCLUSIONS):
        raise FormatError(
            f"quality-exclusion counts disagree with the verified release: "
            f"got {dict(exclusion_counts)}, expected {dict(REFERENCE_QUALITY_EXCLUSIONS)}"
        )
    if len(rows) - len(quality_exclusions) != REFERENCE_QUALITY_MASKS:
        raise FormatError(
            f"quality-valid count mismatch: got {len(rows) - len(quality_exclusions)}, "
            f"expected {REFERENCE_QUALITY_MASKS}"
        )
    actual_exclusions = {
        str(row["id"]): str(row["reason"]) for row in quality_exclusions
    }
    if actual_exclusions != dict(reference_quality_exclusions):
        missing = sorted(set(reference_quality_exclusions) - set(actual_exclusions))
        unexpected = sorted(set(actual_exclusions) - set(reference_quality_exclusions))
        changed = sorted(
            impression_id
            for impression_id in set(actual_exclusions) & set(reference_quality_exclusions)
            if actual_exclusions[impression_id] != reference_quality_exclusions[impression_id]
        )
        raise FormatError(
            "quality exclusions disagree with restricted reference: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, changed={changed[:5]}"
        )

    by_id = {
        str(row["id"]): float(row["roi_quality_iou"])
        for row in rows
        if row["roi_quality_iou"] not in ("", None)
    }
    differences = []
    for impression_id, expected in reference_low_iou.items():
        if impression_id not in by_id:
            differences.append(f"{impression_id}: missing")
        elif abs(by_id[impression_id] - expected) > 0.01:
            differences.append(
                f"{impression_id}: got {by_id[impression_id]:.4f}, expected {expected:.2f}±0.01"
            )
    if differences:
        raise FormatError("low-IoU validation disagrees with laptop reference: " + "; ".join(differences))


def _write_readme(
    path: Path,
    archive_path: Path,
    variant: str,
    rows: Sequence[Mapping[str, object]],
    quality_exclusions: Sequence[Mapping[str, object]],
    quality_threshold: int,
    overlay_count: int,
) -> None:
    counts = Counter(str(row["examiner_value"]) for row in rows)
    quality_count = len(rows) - len(quality_exclusions)
    exclusion_counts = Counter(str(row["reason"]) for row in quality_exclusions)
    content = f"""# SD302h ground-truth masks

This restricted directory was generated from `{archive_path.name}` using the
`{variant}` SD302h transaction branch. It contains {len(rows):,} examiner-annotated
ROI impressions and {quality_count:,} independently usable ridge-quality maps.

## Mask definitions

- `gt_roi/`: the examiner Field 9.300 ROI polygon (or its ROI rectangle when no
  polygon is present), rasterised at the native embedded-image dimensions.
- `gt_quality/`: the {quality_count:,} impressions with a valid Field 9.308,
  using cells with ridge-quality code greater than
  `{quality_threshold}`. With the default threshold 0, codes 1--5 are usable ridge
  detail and code 0 is background, following NIST SP 500-290e3.
- `quality_exclusions.csv`: the {len(quality_exclusions):,} impressions whose
  quality maps are unavailable. They are omitted only from Table A-prime; their
  ROI masks remain valid for Table A. Reason counts are:
  {dict(sorted(exclusion_counts.items()))}.
- `overlays/`: a best-effort deterministic private QA sample of up to
  {min(overlay_count, quality_count)} images. Green is the ROI contour and magenta
  is the quality-mask contour. An undecodable optional overlay does not invalidate
  its independently rasterised masks.

Coordinates use the EFS default lambda unit (0.01 mm) and are converted with
`pixels = units * 0.01 * ppi / 25.4`. ROI path coordinates are relative to the
ROI origin; offsets are applied before rasterisation. Quality cells begin at the
ROI's top-left and run left-to-right, top-to-bottom. The SD302h grids do not obey
the nominal `ceil(ROI / GSZ)` dimensions in SP 500-290e3, so the decoded rectangular
grid is expanded with nearest-neighbour interpolation over the complete ROI. This
mapping reproduces the independently validated low-IoU reference cases; fixed-size
GSZ placement does not.

## Examiner-value counts

| Assessment | Count |
|---|---:|
"""
    for assessment in ("VALUE", "LIMITED", "NOVALUE", "NONPRINT"):
        content += f"| {assessment} | {counts.get(assessment, 0):,} |\n"
    content += """

## Sources

- NIST SP 500-290e3, ANSI/NIST-ITL 1-2011 Update:2015.
- NIST TN 2367, *NIST Special Database 302* (2026).

`summary.csv` contains one row for every ROI-valid impression; quality-area and
ROI/quality-IoU cells are blank for the impressions in `quality_exclusions.csv`.
`errors.csv` is reserved for fatal mandatory-field, image-association, or ROI
failures. A published production build has no such errors. No method-generated
segmentation was used as ground truth.
"""
    path.write_text(content, encoding="utf-8")


def _existing_output_is_complete(out: Path) -> bool:
    """Recognize an already-published output without mutating it."""
    required = (
        out / "summary.csv",
        out / "quality_exclusions.csv",
        out / "errors.csv",
        out / "README.md",
        out / "gt_roi",
        out / "gt_quality",
    )
    return out.is_dir() and all(path.exists() for path in required)


def _read_summary(path: Path) -> List[Dict[str, object]]:
    """Read a generated summary while enforcing its exact public schema."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SUMMARY_COLUMNS:
            raise FormatError(
                f"existing summary schema {reader.fieldnames!r} != {list(SUMMARY_COLUMNS)!r}"
            )
        return [dict(row) for row in reader]


def _read_quality_exclusions(path: Path) -> List[Dict[str, object]]:
    """Read the restricted Table A-prime exclusion manifest."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != QUALITY_EXCLUSION_COLUMNS:
            raise FormatError(
                f"existing quality-exclusion schema {reader.fieldnames!r} != "
                f"{list(QUALITY_EXCLUSION_COLUMNS)!r}"
            )
        return [dict(row) for row in reader]


def _validate_existing_output(
    out: Path,
    overlay_count: int,
    validate_reference: bool,
    reference_low_iou: Mapping[str, float],
    reference_quality_exclusions: Mapping[str, str],
) -> None:
    """Never trust an idempotent-build shortcut without validating its artifacts."""
    rows = _read_summary(out / "summary.csv")
    quality_exclusions = _read_quality_exclusions(out / "quality_exclusions.csv")
    with (out / "errors.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ERROR_COLUMNS:
            raise FormatError(
                f"existing error schema {reader.fieldnames!r} != {list(ERROR_COLUMNS)!r}"
            )
        errors = list(reader)
    if errors:
        raise FormatError(f"existing output records {len(errors)} error(s)")
    _validate_artifacts(rows, quality_exclusions, out, overlay_count)
    if validate_reference:
        _validate_reference(
            rows,
            quality_exclusions,
            reference_low_iou,
            reference_quality_exclusions,
        )


def build(args: argparse.Namespace) -> None:
    """Build into a sibling staging directory and publish only after validation."""
    archive_path = Path(args.archive).resolve()
    out = Path(args.out).resolve()
    reference_low_iou = (
        _load_reference_low_iou(Path(args.reference_low_iou_json).resolve())
        if args.reference_low_iou_json
        else {}
    )
    reference_quality_exclusions = (
        _load_quality_exclusion_reference(
            Path(args.reference_quality_exclusions_json).resolve()
        )
        if args.reference_quality_exclusions_json
        else {}
    )
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if _existing_output_is_complete(out):
        _validate_existing_output(
            out,
            args.overlays,
            not args.skip_reference_validation,
            reference_low_iou,
            reference_quality_exclusions,
        )
        print(f"validated output already exists; leaving unchanged: {out}")
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output exists and is not a recognized complete build: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.rmdir()  # known empty directory only
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))
    started = time.monotonic()
    all_rows: List[Dict[str, object]] = []
    built: List[BuiltImpression] = []
    quality_exclusions: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise FormatError(f"ZIP CRC validation failed for {bad_member}")
            members = _select_archive_members(archive.namelist(), args.variant)
            print(f"selected {len(members):,} LFFS transactions ({args.variant})", flush=True)
            for member_index, member in enumerate(members, start=1):
                try:
                    rows, impression_refs, member_quality_exclusions = process_member(
                        member,
                        archive.read(member),
                        staging,
                        args.quality_threshold,
                    )
                    all_rows.extend(rows)
                    built.extend(impression_refs)
                    quality_exclusions.extend(member_quality_exclusions)
                except Exception as exc:
                    errors.append(
                        {
                            "member": member,
                            "record_index": "",
                            "stage": "transaction",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                if member_index % 250 == 0 or member_index == len(members):
                    print(
                        f"processed {member_index:,}/{len(members):,} transactions; "
                        f"ROI={len(all_rows):,}, quality={len(built):,}, "
                        f"quality exclusions={len(quality_exclusions):,}, "
                        f"fatal errors={len(errors):,}",
                        flush=True,
                    )
            overlay_warnings = _write_overlays(archive, built, staging, args.overlays)
            for warning in overlay_warnings:
                print(f"overlay warning: {warning}", file=sys.stderr)

        all_rows.sort(key=lambda row: str(row["id"]))
        quality_exclusions.sort(key=lambda row: str(row["id"]))
        _write_csv(staging / "summary.csv", all_rows, SUMMARY_COLUMNS)
        _write_csv(
            staging / "quality_exclusions.csv",
            quality_exclusions,
            QUALITY_EXCLUSION_COLUMNS,
        )
        _write_csv(staging / "errors.csv", errors, ERROR_COLUMNS)
        _write_readme(
            staging / "README.md",
            archive_path,
            args.variant,
            all_rows,
            quality_exclusions,
            args.quality_threshold,
            args.overlays,
        )
        if errors:
            raise FormatError(
                f"build produced {len(errors)} error(s); inspect {staging / 'errors.csv'}"
            )
        _validate_artifacts(all_rows, quality_exclusions, staging, args.overlays)
        if not args.skip_reference_validation:
            _validate_reference(
                all_rows,
                quality_exclusions,
                reference_low_iou,
                reference_quality_exclusions,
            )
        os.replace(staging, out)
        elapsed = time.monotonic() - started
        counts = Counter(str(row["examiner_value"]) for row in all_rows)
        print(
            f"published ROI={len(all_rows):,}, quality={len(built):,}, "
            f"quality exclusions={len(quality_exclusions):,} to {out} in {elapsed:.1f}s"
        )
        print("examiner values: " + json.dumps(dict(sorted(counts.items())), sort_keys=True))
        print("20 lowest ROI-quality IoU:")
        quality_rows = [
            row for row in all_rows if row["roi_quality_iou"] not in ("", None)
        ]
        for row in sorted(
            quality_rows,
            key=lambda item: (float(item["roi_quality_iou"]), str(item["id"])),
        )[:20]:
            print(f"  {float(row['roi_quality_iou']):.4f} {row['id']}")
    except Exception:
        print(f"build failed; diagnostic staging output retained at {staging}", file=sys.stderr)
        raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="path to the official sd302h.zip")
    parser.add_argument("out", help="restricted output directory")
    parser.add_argument(
        "--variant",
        choices=("original_unmasked", "original_masked", "enhanced_unmasked", "enhanced_masked"),
        default="original_unmasked",
        help="documented SD302h media branch to parse (default: original_unmasked)",
    )
    parser.add_argument(
        "--quality-threshold",
        type=int,
        default=0,
        choices=range(0, 5),
        metavar="0..4",
        help="quality mask includes codes strictly above this value (default: 0)",
    )
    parser.add_argument("--overlays", type=int, default=200)
    parser.add_argument(
        "--reference-low-iou-json",
        help="restricted JSON object mapping validation image IDs to expected IoUs",
    )
    parser.add_argument(
        "--reference-quality-exclusions-json",
        help=(
            "restricted JSON object mapping each approved quality-exclusion ID "
            "to missing_quality_map or ragged_quality_map"
        ),
    )
    parser.add_argument(
        "--skip-reference-validation",
        action="store_true",
        help="for synthetic unit fixtures only; never use for the production SD302h build",
    )
    args = parser.parse_args(argv)
    if args.overlays < 0:
        parser.error("--overlays must be non-negative")
    if not args.skip_reference_validation and (
        not args.reference_low_iou_json or not args.reference_quality_exclusions_json
    ):
        parser.error(
            "--reference-low-iou-json and --reference-quality-exclusions-json "
            "are required unless --skip-reference-validation is used"
        )
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    build(parse_args(argv))


if __name__ == "__main__":
    main()
