"""Manifest loading, image reading and resolution normalisation."""
from pathlib import Path
from typing import Mapping, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["image_path", "mask_path", "subject_id", "subset", "ppi"]


def load_manifest(path):
    # Preserve anonymous NIST identifiers exactly; integer inference would drop
    # their leading zeroes before the later string conversion.
    df = pd.read_csv(path, dtype={"subject_id": "string"})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")
    if "image_id" not in df.columns:
        df["image_id"] = df["image_path"].map(lambda p: Path(p).stem)
    if df["image_id"].duplicated().any():
        dups = df.loc[df["image_id"].duplicated(), "image_id"].head().tolist()
        raise ValueError(f"image_id must be unique, duplicates e.g. {dups}")
    df["mask_path"] = df["mask_path"].fillna("").astype(str)
    df["subject_id"] = df["subject_id"].astype(str)
    df["ppi"] = df["ppi"].astype(float)
    for column in ("mask_ppi", "quality_mask_ppi", "target_ppi",
                   "target_width", "target_height"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def mask_ppi_for(row: Mapping[str, object], mask_column: str) -> float:
    """Return a mask's declared PPI, falling back only for legacy manifests.

    A column named ``quality_mask_path`` uses ``quality_mask_ppi``; the primary
    ``mask_path`` uses ``mask_ppi``. Older synthetic manifests have neither and
    therefore retain the historical assumption that image and mask PPI match.
    """
    candidate = mask_column[:-5] + "_ppi" if mask_column.endswith("_path") else ""
    keys = [candidate] if candidate else []
    if mask_column == "mask_path":
        keys.append("mask_ppi")
    for key in keys:
        value = row.get(key)
        if value is not None and not pd.isna(value):
            ppi = float(value)
            if ppi <= 0:
                raise ValueError(f"{row.get('image_id', '<unknown>')}: invalid {key}={value}")
            return ppi
    return float(row["ppi"])


def manifest_target_shape(
    row: Mapping[str, object], target_ppi: float
) -> Optional[Tuple[int, int]]:
    """Return a pre-audited target ``(height, width)`` when declared.

    Legacy manifests may omit all three target-grid fields.  A partially
    declared or incompatible grid is an error: silently falling back to
    independent image/mask scaling can misregister ground truth.
    """
    required = ("target_ppi", "target_height", "target_width")
    present = [
        row.get(key) is not None
        and not pd.isna(row.get(key))
        and str(row.get(key)).strip() != ""
        for key in required
    ]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            f"{row.get('image_id', '<unknown>')}: target_ppi, target_height and "
            "target_width must be declared together"
        )
    declared_ppi = float(row["target_ppi"])
    requested_ppi = float(target_ppi)
    if (
        not np.isfinite(declared_ppi)
        or declared_ppi <= 0
        or not np.isfinite(requested_ppi)
        or requested_ppi <= 0
    ):
        raise ValueError(f"{row.get('image_id', '<unknown>')}: invalid target PPI")
    if abs(declared_ppi - requested_ppi) > 1e-6:
        raise ValueError(
            f"{row.get('image_id', '<unknown>')}: manifest target PPI "
            f"{declared_ppi} differs from requested {requested_ppi}"
        )
    dimensions = [float(row["target_height"]), float(row["target_width"])]
    if any(
        not np.isfinite(value)
        or value <= 0
        or not float(value).is_integer()
        for value in dimensions
    ):
        raise ValueError(f"{row.get('image_id', '<unknown>')}: invalid target shape")
    height, width = (int(value) for value in dimensions)
    return height, width


def attach_split(df, splits):
    """splits: dict subject_id -> 'train' | 'val' | 'test'."""
    df = df.copy()
    df["split"] = df["subject_id"].map(splits)
    if df["split"].isna().any():
        n = int(df["split"].isna().sum())
        raise ValueError(f"{n} manifest rows have a subject_id missing from the split file")
    return df


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"could not read mask: {path}")
    return m > 0


def to_target_ppi(arr, ppi, target_ppi, is_mask=False):
    """Resample an image or mask so every method is scored at one resolution."""
    if abs(ppi - target_ppi) < 1e-6:
        return arr
    scale = target_ppi / ppi
    h, w = arr.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    if is_mask:
        out = cv2.resize(arr.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        return out.astype(bool)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(arr, size, interpolation=interp)


def match_shape(mask, shape):
    """Nearest-neighbour resize of a boolean mask to a given (h, w)."""
    if mask.shape == tuple(shape):
        return mask
    out = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return out.astype(bool)
