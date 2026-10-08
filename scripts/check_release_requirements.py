#!/usr/bin/env python3
"""Audit result-independent public-repository and project-site requirements."""
from __future__ import annotations

import csv
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import re
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


class SiteAudit(HTMLParser):
    """Collect semantic site structure without executing browser code."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.method_cards = 0
        self.abstract_depth = 0
        self.abstract_parts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        element_id = values.get("id")
        if element_id:
            self.ids.add(element_id)
        classes = set((values.get("class") or "").split())
        if tag == "article" and {"card", "method"}.issubset(classes):
            self.method_cards += 1
        if element_id == "abstract":
            self.abstract_depth = 1
        elif self.abstract_depth:
            self.abstract_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self.abstract_depth:
            self.abstract_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.abstract_depth:
            self.abstract_parts.append(data)


def _load_builder() -> ModuleType:
    path = ROOT / "site" / "build.py"
    spec = importlib.util.spec_from_file_location("public_site_build_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load site builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_paths() -> tuple[str, ...]:
    return (
        "README.md", "LICENSE", "CITATION.cff", "CHANGELOG.md",
        "segbench", "configs", "scripts", "splits/sd302_subject_split.json",
        "results", "figures", "docs/METHODOLOGY.md", "docs/PROTOCOL.md",
        "docs/DATASETS.md", "docs/METHODS.md", "docs/RESULTS.md",
        "docs/LIMITATIONS.md", "docs/FAQ.md", "examples/synthetic/quickstart.ipynb",
        "tests", "environment.yml", "requirements.txt", "pyproject.toml",
        ".github/workflows/ci.yml", ".github/workflows/pages.yml",
    )


def main() -> None:
    checks: dict[str, bool] = {}
    checks["required_repository_structure"] = all(
        (ROOT / relative).exists() for relative in _required_paths()
    )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_sections = (
        "TL;DR", "Headline results", "Overview", "Methods mapped from the review",
        "Datasets", "Quick start", "Protocol summary", "Project links", "Citation",
        "Licence, acknowledgements, and contact",
    )
    checks["readme_ten_sections"] = all(
        re.search(rf"^## {re.escape(section)}$", readme, re.M)
        for section in readme_sections
    )
    checks["readme_badges_and_links"] = all(
        token in readme for token in ("License: MIT", "actions/workflows/ci.yml", "DOI", "Project website")
    )
    checks["readme_no_claimed_partial_result"] = "Headline finding:** pending" in readme
    headline_match = re.search(
        r"<!-- BEGIN GENERATED HEADLINE RESULTS -->(.*?)<!-- END GENERATED HEADLINE RESULTS -->",
        readme,
        re.S,
    )
    with (ROOT / "results" / "public_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        public_rows = [
            row
            for row in csv.DictReader(handle)
            if row["table"] == "A" and row["view"] == "overall" and row["group"] == "overall"
        ]
    headline = headline_match.group(1) if headline_match else ""
    checks["readme_headline_from_aggregate_rows"] = bool(headline_match) and all(
        f"| {row['family']} | {row['approach']} |" in headline for row in public_rows
    ) and headline.count("\n|") == len(public_rows) + 2

    builder = _load_builder()
    catalog = tuple(builder.METHOD_CATALOG)
    checks["twelve_review_method_entries"] = len(catalog) == 12 and len(
        {row["entry"] for row in catalog}
    ) == 12
    checks["method_cards_have_required_fields"] = all(
        all(row.get(field) for field in ("entry", "family", "source", "implementation", "deviation"))
        for row in catalog
    )

    html = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    audit = SiteAudit()
    audit.feed(html)
    required_ids = {
        "abstract", "problem", "pipeline", "datasets", "methods", "results", "demo",
        "findings", "reproduce", "limitations", "citation", "metric", "group-filter",
        "sort-toggle", "pareto", "significance-body", "synthetic-select",
        "synthetic-method", "public-data",
    }
    checks["site_sections_and_controls"] = required_ids.issubset(audit.ids)
    checks["site_exactly_twelve_method_cards"] = audit.method_cards == 12
    abstract_words = re.findall(r"\b[\w–-]+\b", " ".join(audit.abstract_parts))
    checks["abstract_120_to_150_words"] = 120 <= len(abstract_words) <= 150
    checks["site_no_tracking_or_external_analytics"] = not any(
        token in html.casefold()
        for token in ("google-analytics", "googletagmanager", "gtag(", "mixpanel", "segment.io")
    )
    checks["site_hull_explainer"] = (ROOT / "site" / "assets" / "hull-explainer.svg").is_file()

    app = (ROOT / "site" / "assets" / "app.js").read_text(encoding="utf-8")
    checks["interactive_results_features"] = all(
        token in app
        for token in (
            "renderTable", "renderPareto", "renderSignificance", "renderGroupFilter",
            "metricCell", "sortAscending", "activeGroup",
        )
    )
    checks["synthetic_demo_is_receipt_gated"] = all(
        token in (ROOT / "site" / "build.py").read_text(encoding="utf-8")
        for token in ("method_prediction", "source_stem", "synthetic receipt mismatch")
    )

    receipt = json.loads(
        (ROOT / "examples" / "synthetic" / "synthetic_receipt.json").read_text(encoding="utf-8")
    )
    checks["synthetic_assets_have_receipt"] = receipt.get("kind") == "segbench_synthetic_dataset" and bool(
        receipt.get("assets")
    )

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    checks["ci_runs_synthetic_and_public_gates"] = all(
        token in workflow
        for token in (
            "smoke_synthetic.py", "check_site_numbers.py", "check_accessibility.py",
            "check_release_requirements.py", "scan_release.py",
        )
    )
    checks["license_and_citation"] = "MIT License" in (ROOT / "LICENSE").read_text(
        encoding="utf-8"
    ) and all(
        token in (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        for token in ("Udit", "TODO: co-authors", "license: MIT")
    )
    checks["classical_and_zero_shot_configs"] = all(
        (ROOT / "configs" / name).is_file()
        for name in ("svm.yaml", "rf.yaml", "welm.yaml", "gmfs.yaml", "sufs_zero.yaml", "fingernet.yaml")
    )

    report = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "method_cards": audit.method_cards,
        "abstract_words": len(abstract_words),
        "note": "Result-dependent metrics, model-derived synthetic predictions, and publication remain gated.",
    }
    (ROOT / "release_requirements.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
