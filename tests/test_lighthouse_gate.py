from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "lighthouse_gate_test", ROOT / "scripts/check_lighthouse.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


URL = "https://example.invalid/benchmark/"


def _site(tmp_path: Path) -> tuple[Path, Path]:
    index = tmp_path / "index.html"
    manifest = tmp_path / "build_manifest.json"
    index.write_text("<html><body>synthetic</body></html>", encoding="utf-8")
    manifest.write_text('{"schema_version":1,"status":"PASS"}\n', encoding="utf-8")
    return index, manifest


def _report(
    tmp_path: Path, score: object, *, url: str = URL,
    fetch_time: datetime | None = None,
) -> Path:
    path = tmp_path / "lighthouse.json"
    fetched = fetch_time or datetime.now(timezone.utc) + timedelta(seconds=2)
    path.write_text(json.dumps({
        "requestedUrl": url, "finalUrl": url,
        "fetchTime": fetched.isoformat().replace("+00:00", "Z"),
        "categories": {"accessibility": {"score": score}},
    }))
    return path


def _check(tmp_path: Path, score: object, **kwargs):
    index, manifest = _site(tmp_path)
    return _module().check_report(
        _report(tmp_path, score, **kwargs), expected_url=URL,
        site_index=index, build_manifest=manifest,
    )


def test_lighthouse_gate_accepts_score_at_release_floor(tmp_path: Path) -> None:
    result = _check(tmp_path, 0.90)
    assert result["status"] == "PASS"
    assert result["minimum"] == 0.90


def test_lighthouse_gate_rejects_score_below_release_floor(tmp_path: Path) -> None:
    result = _check(tmp_path, 0.899)
    assert result["status"] == "FAIL"


@pytest.mark.parametrize("score", [None, True, "0.95", 1.1])
def test_lighthouse_gate_rejects_invalid_report_score(tmp_path: Path, score: object) -> None:
    with pytest.raises(ValueError, match="score"):
        _check(tmp_path, score)


def test_lighthouse_gate_rejects_wrong_site(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="URL differs"):
        _check(tmp_path, 0.95, url="https://example.invalid/other/")


def test_lighthouse_gate_rejects_report_older_than_built_site(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stale"):
        _check(tmp_path, 0.95, fetch_time=datetime(2000, 1, 1, tzinfo=timezone.utc))


def test_lighthouse_gate_rejects_future_dated_report(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="future"):
        _check(tmp_path, 0.95, fetch_time=datetime.now(timezone.utc) + timedelta(days=1))
