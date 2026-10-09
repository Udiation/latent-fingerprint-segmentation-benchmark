from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from segbench.public_figures import (
    FigureManifestError,
    build_manifest,
    validate_figure_manifest,
)


def _case(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    figure = tmp_path / "figures" / "dice_boxplot.png"
    figure.parent.mkdir()
    figure.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic aggregate plot")
    digest = "a" * 64
    manifest = build_manifest(tmp_path, [figure], source_receipt_sha256=digest)
    path = figure.parent / "figure_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return figure, path, manifest


def test_figure_manifest_accepts_only_hash_bound_declared_aggregate_plot(tmp_path: Path) -> None:
    _, path, _ = _case(tmp_path)
    approved = validate_figure_manifest(
        tmp_path, path, expected_source_receipt_sha256="a" * 64
    )
    assert set(approved) == {"figures/dice_boxplot.png"}


def test_figure_manifest_rejects_byte_tampering(tmp_path: Path) -> None:
    figure, path, _ = _case(tmp_path)
    figure.write_bytes(figure.read_bytes() + b"changed")
    with pytest.raises(FigureManifestError, match="bytes changed"):
        validate_figure_manifest(tmp_path, path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"path": "figures/../outside.png"}, "unsafe aggregate figure path"),
        ({"kind": "qualitative_biometric_gallery"}, "kind differs"),
        ({"source_receipt_sha256": "b" * 64}, "source receipt changed"),
    ],
)
def test_figure_manifest_rejects_path_kind_and_receipt_mutations(
    tmp_path: Path, mutation: dict[str, str], message: str
) -> None:
    _, path, manifest = _case(tmp_path)
    changed = copy.deepcopy(manifest)
    changed["figures"][0].update(mutation)
    path.write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n")
    with pytest.raises(FigureManifestError, match=message):
        validate_figure_manifest(tmp_path, path)


def test_figure_manifest_rejects_independent_receipt_mismatch(tmp_path: Path) -> None:
    _, path, _ = _case(tmp_path)
    with pytest.raises(FigureManifestError, match="independently pinned"):
        validate_figure_manifest(tmp_path, path, expected_source_receipt_sha256="b" * 64)
