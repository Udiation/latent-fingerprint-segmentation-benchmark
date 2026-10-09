#!/usr/bin/env python3
"""Fast structural accessibility gate; Lighthouse remains the release audit."""
from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Audit(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.lang = ""
        self.title = False
        self.main = 0
        self.h1 = 0
        self.images = 0
        self.images_with_alt = 0
        self.buttons = 0
        self.buttons_named = 0
        self.current_button: dict[str, str] | None = None
        self.button_labels_match = True

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "html": self.lang = values.get("lang") or ""
        if tag == "title": self.title = True
        if tag == "main": self.main += 1
        if tag == "h1": self.h1 += 1
        if tag == "img":
            self.images += 1
            if "alt" in values and values.get("alt") is not None: self.images_with_alt += 1
        if tag == "button":
            self.buttons += 1
            if values.get("aria-label") or values.get("id") == "copy-cite" or "tab" in (values.get("class") or "").split(): self.buttons_named += 1
            self.current_button = {
                "accessible": values.get("aria-label") or "",
                "visible": "",
            }

    def handle_data(self, data: str) -> None:
        if self.current_button is not None:
            self.current_button["visible"] += data

    def handle_endtag(self, tag: str) -> None:
        if tag != "button" or self.current_button is None:
            return
        visible = " ".join(self.current_button["visible"].split()).casefold()
        accessible = " ".join(self.current_button["accessible"].split()).casefold()
        if visible and accessible and visible not in accessible:
            self.button_labels_match = False
        self.current_button = None


def main() -> None:
    audit = Audit()
    audit.feed((ROOT / "site" / "index.html").read_text(encoding="utf-8"))
    checks = {
        "document_language": bool(audit.lang),
        "document_title": audit.title,
        "single_main": audit.main == 1,
        "single_h1": audit.h1 == 1,
        "all_images_have_alt": audit.images == audit.images_with_alt,
        "all_buttons_named": audit.buttons == audit.buttons_named,
        "visible_button_labels_match_accessible_names": audit.button_labels_match,
    }
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
