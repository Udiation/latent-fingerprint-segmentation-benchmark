"""Synthetic-only structural tests for the public repository and project page."""
from __future__ import annotations

from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


class CardCounter(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.methods = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        classes = set((dict(attrs).get("class") or "").split())
        if tag == "article" and {"card", "method"}.issubset(classes):
            self.methods += 1


def _load_builder():
    path = ROOT / "site" / "build.py"
    spec = importlib.util.spec_from_file_location("release_site_build_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_requirement_checker_passes() -> None:
    subprocess.run(
        [sys.executable, "site/build.py"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    completed = subprocess.run(
        [sys.executable, "scripts/check_release_requirements.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["status"] == "PASS"
    assert report["method_cards"] == 12
    assert 120 <= report["abstract_words"] <= 150


def test_method_catalog_and_generated_cards_match_review_entries() -> None:
    builder = _load_builder()
    assert len(builder.METHOD_CATALOG) == 12
    assert len({row["entry"] for row in builder.METHOD_CATALOG}) == 12
    builder.main()
    counter = CardCounter()
    counter.feed((ROOT / "site" / "index.html").read_text(encoding="utf-8"))
    assert counter.methods == 12


def test_browser_features_are_wired_to_embedded_aggregate_payload() -> None:
    html = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "site" / "assets" / "app.js").read_text(encoding="utf-8")
    for element_id in (
        "metric", "group-filter", "sort-toggle", "result-body", "pareto",
        "significance-body", "synthetic-select", "synthetic-method", "public-data",
    ):
        assert f'id="{element_id}"' in html
    for function in (
        "renderTable", "renderPareto", "renderSignificance", "renderGroupFilter",
        "renderFindings",
    ):
        assert function in javascript
    assert "payload.results" in javascript
    assert "payload.statistics" in javascript
    assert "sig-marker" in javascript
    assert 'aria-label="High to low"' in html
    assert 'setAttribute("aria-label", label)' in javascript


def test_synthetic_notebook_is_output_free_and_clearly_non_benchmark() -> None:
    notebook = json.loads(
        (ROOT / "examples" / "synthetic" / "quickstart.ipynb").read_text(encoding="utf-8")
    )
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell["outputs"] == [] and cell["execution_count"] is None for cell in code_cells)
    text = json.dumps(notebook)
    assert "synthetic" in text.lower()
    assert "not paper results" in text.lower()


def test_method_prediction_contract_is_fail_closed() -> None:
    builder = _load_builder()
    source = (ROOT / "site" / "build.py").read_text(encoding="utf-8")
    assert "method_prediction" in source
    assert "source_stem" in source
    examples = builder.synthetic_examples()
    assert examples
    assert all("predictions" in example for example in examples)


def test_synthetic_receipt_rejects_path_traversal() -> None:
    builder = _load_builder()
    with pytest.raises(ValueError, match="unsafe asset path"):
        builder._approved_assets(
            {"assets": [{"path": "../outside.png", "role": "method_prediction"}]}
        )
