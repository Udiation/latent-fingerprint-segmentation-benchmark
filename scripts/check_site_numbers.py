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
    expected = {
        "results": read_rows(sources["public_results.csv"]),
        "tables": read_rows(sources["table_status.csv"]),
        "statistics": read_rows(sources["statistics.csv"]),
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

