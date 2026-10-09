#!/usr/bin/env python3
"""Fail unless a Lighthouse accessibility report meets the release floor."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]


def _url(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Lighthouse report lacks {label}")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"Lighthouse {label} is not an expected HTTP(S) URL")
    path = parsed.path.rstrip("/") + "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))


def check_report(
    path: Path,
    *,
    expected_url: str,
    site_index: Path,
    build_manifest: Path,
    minimum: float = 0.90,
    maximum_age_hours: float = 24.0,
) -> dict[str, object]:
    if not 0 <= minimum <= 1:
        raise ValueError("minimum accessibility score must be in [0, 1]")
    data = path.read_bytes()
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Lighthouse input is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("categories"), dict):
        raise ValueError("Lighthouse report lacks categories")
    expected = _url(expected_url, "expected URL")
    requested = _url(payload.get("requestedUrl"), "requestedUrl")
    final = _url(payload.get("finalUrl"), "finalUrl")
    if requested != expected or final != expected:
        raise ValueError("Lighthouse report URL differs from the expected publication URL")
    if not site_index.is_file() or site_index.is_symlink() or not build_manifest.is_file() or build_manifest.is_symlink():
        raise ValueError("current site artifacts are absent or symlinked")
    try:
        manifest_payload = json.loads(build_manifest.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("site build manifest is invalid") from exc
    if not isinstance(manifest_payload, dict) or manifest_payload.get("status") != "PASS":
        raise ValueError("site build manifest did not pass")
    fetch_time = payload.get("fetchTime")
    if not isinstance(fetch_time, str):
        raise ValueError("Lighthouse report lacks fetchTime")
    try:
        fetched = datetime.fromisoformat(fetch_time.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Lighthouse fetchTime is invalid") from exc
    if fetched.tzinfo is None:
        raise ValueError("Lighthouse fetchTime lacks a timezone")
    now = datetime.now(timezone.utc)
    fetched_utc = fetched.astimezone(timezone.utc)
    if fetched_utc > now + timedelta(minutes=5):
        raise ValueError("Lighthouse fetchTime is implausibly in the future")
    if maximum_age_hours <= 0 or (now - fetched_utc).total_seconds() > maximum_age_hours * 3600:
        raise ValueError("Lighthouse report is stale")
    built_at = max(site_index.stat().st_mtime, build_manifest.stat().st_mtime)
    if fetched_utc.timestamp() + 1 < built_at:
        raise ValueError("Lighthouse report predates the current built site")
    accessibility = payload["categories"].get("accessibility")
    if not isinstance(accessibility, dict):
        raise ValueError("Lighthouse report lacks the accessibility category")
    score = accessibility.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
        raise ValueError("Lighthouse accessibility score is invalid")
    score = float(score)
    if not 0 <= score <= 1:
        raise ValueError("Lighthouse accessibility score is outside [0, 1]")
    return {
        "schema_version": 1,
        "kind": "lighthouse_accessibility_gate_v1",
        "status": "PASS" if score >= minimum else "FAIL",
        "score": score,
        "minimum": minimum,
        "expected_url": expected,
        "requested_url": requested,
        "final_url": final,
        "fetch_time": fetch_time,
        "maximum_age_hours": maximum_age_hours,
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "site_index_sha256": hashlib.sha256(site_index.read_bytes()).hexdigest(),
        "site_build_manifest_sha256": hashlib.sha256(build_manifest.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--minimum", type=float, default=0.90)
    parser.add_argument("--maximum-age-hours", type=float, default=24.0)
    parser.add_argument("--expected-url", required=True)
    parser.add_argument("--site-index", type=Path, default=ROOT / "site/index.html")
    parser.add_argument("--build-manifest", type=Path, default=ROOT / "site/build_manifest.json")
    args = parser.parse_args()
    report = check_report(
        args.input, expected_url=args.expected_url, site_index=args.site_index,
        build_manifest=args.build_manifest, minimum=args.minimum,
        maximum_age_hours=args.maximum_age_hours,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
