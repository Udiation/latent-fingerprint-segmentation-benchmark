from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "site" / "build.py"


def _module():
    spec = importlib.util.spec_from_file_location("site_build_signed_stats", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_signed_difference_intervals_are_narrowly_allowed(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "statistics.csv"
    path.write_text(
        "estimate,ci_lower,ci_upper\n-0.04,-0.06,-0.02\n",
        encoding="utf-8",
    )
    rows = module.read_csv(
        path,
        {"estimate", "ci_lower", "ci_upper"},
        signed_columns={"estimate", "ci_lower", "ci_upper"},
    )
    assert rows[0]["ci_lower"] == "-0.06"


def test_result_intervals_remain_proportions(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "results.csv"
    path.write_text("ci_lower,ci_upper\n-0.01,0.50\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        module.read_csv(path, {"ci_lower", "ci_upper"})
