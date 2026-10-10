#!/usr/bin/env python3
"""Require the generated site's embedded result rows to match the public CSV."""
from __future__ import annotations

import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PayloadParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.capture = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "script" and values.get("id") == "public-data" and values.get("type") == "application/json":
            self.capture = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self.capture:
            self.capture = False

    def handle_data(self, data: str) -> None:
        if self.capture:
            self.parts.append(data)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    index = ROOT / "site" / "index.html"
    parser = PayloadParser()
    parser.feed(index.read_text(encoding="utf-8"))
    if not parser.parts:
        raise SystemExit("site has no aggregate JSON payload")
    payload = json.loads("".join(parser.parts))
    sources = {
        "public_results.csv": ROOT / "results" / "public_results.csv",
        "table_status.csv": ROOT / "results" / "table_status.csv",
        "statistics.csv": ROOT / "results" / "statistics.csv",
    }
    supplemental = {
        "sd302b_results.csv": ROOT / "results" / "sd302b_results.csv",
        "sd302b_table_status.csv": ROOT / "results" / "sd302b_table_status.csv",
        "sd302b_release_receipt.json": ROOT / "results" / "sd302b_release_receipt.json",
    }
    if any(path.exists() for path in supplemental.values()):
        if not all(path.is_file() for path in supplemental.values()):
            raise SystemExit("SD302b supplement is incomplete")
        sources.update(supplemental)
    result_rows = read_rows(sources["public_results.csv"])
    table_rows = read_rows(sources["table_status.csv"])
    if "sd302b_results.csv" in sources:
        amendments = read_rows(sources["sd302b_results.csv"])
        result_key = ("table", "view", "group", "method")
        by_result = {tuple(row[key] for key in result_key): row for row in result_rows}
        by_result.update({tuple(row[key] for key in result_key): row for row in amendments})
        result_rows = list(by_result.values())
        amendments = read_rows(sources["sd302b_table_status.csv"])
        by_table = {row["table"]: row for row in table_rows}
        by_table.update({row["table"]: row for row in amendments})
        order = ["A", "A_value", "A_prime", "B1", "B_rolled", "B_plain", "D"]
        table_rows = sorted(
            by_table.values(),
            key=lambda row: (order.index(row["table"]) if row["table"] in order else 99, row["table"]),
        )
    statistics = read_rows(sources["statistics.csv"])
    statistics = [
        row for row in statistics
        if not (
            row["status"] == "n/a"
            and "ra_runet_aligned" in row["comparison"]
        )
    ]
    expected = {
        "results": result_rows,
        "tables": table_rows,
        "statistics": statistics,
    }
    for key, rows in expected.items():
        if payload.get(key) != rows:
            raise SystemExit(f"site payload differs from {key} source rows")
    manifest = json.loads((ROOT / "site" / "build_manifest.json").read_text(encoding="utf-8"))
    for name, path in sources.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if manifest.get("source_sha256", {}).get(name) != digest:
            raise SystemExit(f"site build digest differs from {name}")
    print(json.dumps({"status": "PASS", "result_rows": len(expected["results"]), "checked_sources": len(sources)}, sort_keys=True))


if __name__ == "__main__":
    main()
