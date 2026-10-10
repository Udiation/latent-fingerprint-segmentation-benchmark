#!/usr/bin/env python3
"""Fail-closed privacy and hygiene scan for the public release tree."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from segbench.synthetic_demo import validate_synthetic_demo_receipt
from segbench.public_figures import (
    FIGURE_KINDS, MIME_BY_SUFFIX, FigureManifestError, validate_figure_manifest,
)
from import_stage5_public import (  # type: ignore  # script-directory import
    Stage5ImportError, _regular_bytes, _sha256, _validate_import_receipt, run_import,
)

MAX_BYTES = 10 * 1024 * 1024
BANNED_ARTIFACT_SUFFIXES = {
    ".bin", ".ckpt", ".h5", ".joblib", ".npy", ".npz", ".onnx", ".pt",
    ".pth", ".safetensors",
}
RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
AGGREGATE_FIGURE_SUFFIXES = {".png", ".pdf"}
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".toml",
    ".cff", ".html", ".css", ".js", ".svg", ".sh", ".ipynb",
}
ALLOWED_NON_TEXT_NAMES = {"LICENSE", ".gitignore"}
PROVISIONAL_RECEIPT_KIND = "provisional_public_bundle_receipt_v1"
PROVISIONAL_OUTPUTS = ("public_results.csv", "statistics.csv", "table_status.csv")
PROVISIONAL_B1_COMPLETE = {
    "all_background", "all_foreground", "gmfs_default", "hull_as_prediction", "sufs_zero",
}
PROVISIONAL_B1_PENDING = {
    "alexnet", "cnn_ae", "drunet", "fingernet", "gmfs_tuned", "lenet",
    "r_unet", "ra_runet_aligned", "rf", "segformer_b2", "sufs_ft", "svm",
    "unet", "unetpp", "welm",
}
_OPAQUE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{25,80}(?![A-Za-z0-9_-])"
)
_DATA_IMAGE_URI = re.compile(
    re.escape("data:" + "image/")
    + r"[A-Za-z0-9.+-]+(?:;[A-Za-z0-9.+_=-]+)*,"
    + r"(?:[A-Za-z0-9+/=_.,;:%#?@!$&()*~-][ \t\r\n]*){16,}",
    re.IGNORECASE,
)


def patterns() -> list[tuple[str, re.Pattern[str]]]:
    return [
        ("absolute private path", re.compile("/" + "(?:home|mnt|data|restricted)/")),
        ("cloud-drive URL", re.compile("(?:drive\\." + "google\\.com|docs\\." + "google\\.com)", re.I)),
        ("IPv4 address", re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")),
        ("SD302-like image identifier", re.compile(r"(?<![A-Za-z0-9])\d{8}_[A-Za-z0-9]")),
        ("source-resolution filename token", re.compile("1000" + "PPI", re.I)),
        ("latent-position filename token", re.compile(r"\bLP\d{2}\b", re.I)),
        (
            "local infrastructure hostname",
            re.compile(
                r"(?:\b[a-z0-9-]+\.local\b|"
                r"\b(?:ssh|scp)\s+(?:[^\s@]+@)?[A-Za-z0-9._-]+)",
                re.I,
            ),
        ),
    ]


def opaque_identifier(value: str) -> str | None:
    """Return a cloud-drive-like token without embedding any private denylist.

    Cryptographic digests are uniformly hexadecimal, while repository symbols
    tend to contain several word separators.  A bounded mixed-case token with
    a digit and no more than two separators is therefore treated as an opaque
    external identifier and rejected.  This covers raw cloud file IDs without
    confusing SHA-1/SHA-256 values or descriptive Python identifiers for one.
    """
    for token in _OPAQUE_TOKEN.findall(value):
        if (
            any(character.islower() for character in token)
            and any(character.isupper() for character in token)
            and any(character.isdigit() for character in token)
            and sum(character in "_-" for character in token) <= 2
        ):
            return token
    return None


def long_encoded_token(value: str) -> bool:
    """Reject raw base64/high-entropy payloads that evade bounded token scans."""
    return re.search(
        r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/=_-]{81,}(?![A-Za-z0-9+/=_-])",
        value,
    ) is not None


def files() -> Iterable[Path]:
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if ".git" in path.parts or any(
            part in {".smoke", ".venv", ".pytest_cache", "__pycache__"}
            for part in relative.parts
        ):
            continue
        # ``Path.is_file()`` follows symlinks and returns false for directory
        # symlinks. Yield every symlink first so the caller rejects it rather
        # than silently omitting a Git-tracked link from the privacy scan.
        if path.is_symlink():
            yield path
            continue
        if not path.is_file():
            continue
        yield path


def approved_synthetic_hashes() -> set[str]:
    receipt = ROOT / "examples" / "synthetic" / "synthetic_receipt.json"
    if not receipt.is_file():
        return set()
    data = json.loads(receipt.read_text(encoding="utf-8"))
    approved_roles = {"example_image", "ground_truth_mask", "method_prediction"}
    return {
        str(item.get("sha256"))
        for item in data.get("assets", [])
        if item.get("role") in approved_roles
    }


def _scan_text(
    value: str,
    context: str,
    errors: list[str],
    *,
    allow_identifier_explanation: bool = False,
) -> None:
    """Scan semantic text or a repository path with one shared policy."""
    for label, pattern in patterns():
        if allow_identifier_explanation and label in {
            "SD302-like image identifier",
            "source-resolution filename token",
            "latent-position filename token",
        }:
            continue
        if pattern.search(value):
            errors.append(f"{context}: contains {label}")
    token = opaque_identifier(value)
    if token is not None:
        errors.append(f"{context}: contains opaque cloud-drive-like identifier")
    if long_encoded_token(value):
        errors.append(f"{context}: contains long encoded token")
    if _DATA_IMAGE_URI.search(value) or re.search(
        r"<image\b[^>]*(?:href|xlink:href)\s*=\s*['\"][^'\"]+['\"][^>]*>",
        value,
        re.IGNORECASE,
    ):
        errors.append(f"{context}: contains embedded image payload")


def scan_history(
    errors: list[str], root: Path = ROOT,
    approved_binary_hashes: set[str] | None = None,
) -> None:
    """Scan every committed patch, excluding commit metadata such as author mail.

    The current-tree scan alone is insufficient: a secret or biometric ID that
    was deleted in a later commit remains public after a push.  We track each
    diff's path so the narrowly scoped naming explanation remains the sole
    exception for SD302-style example tokens.
    """
    probe = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if probe.returncode or probe.stdout.strip() != "true":
        return
    messages = subprocess.run(
        ["git", "log", "--format=%B%x00", "--all"],
        cwd=root, capture_output=True, text=True,
    )
    if messages.returncode:
        errors.append("could not inspect git commit messages")
    else:
        for line_number, line in enumerate(messages.stdout.splitlines(), start=1):
            _scan_text(line, f"git history commit message line {line_number}", errors)
    result = subprocess.run(
        ["git", "log", "--format=", "-p", "--all", "--no-ext-diff"],
        cwd=root,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.returncode:
        errors.append("could not inspect git history")
        return
    current_path = ""
    for line_number, line in enumerate(result.stdout.splitlines(), start=1):
        if line.startswith("diff --git a/"):
            match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            current_path = match.group(2) if match else ""
            _scan_text(
                current_path,
                f"git history path at patch line {line_number}",
                errors,
                allow_identifier_explanation=current_path == "docs/DATASETS.md",
            )
            continue
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        _scan_text(
            line[1:],
            f"git history {current_path or 'unknown path'} at patch line {line_number}",
            errors,
            allow_identifier_explanation=current_path == "docs/DATASETS.md",
        )
    objects = subprocess.run(
        ["git", "rev-list", "--objects", "--all"], cwd=root,
        capture_output=True, text=True,
    )
    if objects.returncode:
        errors.append("could not inspect git object inventory")
        return
    approved = approved_binary_hashes or set()
    seen: set[tuple[str, str]] = set()
    for line in objects.stdout.splitlines():
        if " " not in line:
            continue
        object_id, historical_path = line.split(" ", 1)
        suffix = Path(historical_path).suffix.lower()
        relevant = suffix in (RASTER_SUFFIXES | AGGREGATE_FIGURE_SUFFIXES | BANNED_ARTIFACT_SUFFIXES)
        if not relevant or (object_id, historical_path) in seen:
            continue
        seen.add((object_id, historical_path))
        kind = subprocess.run(
            ["git", "cat-file", "-t", object_id], cwd=root,
            capture_output=True, text=True,
        )
        if kind.returncode or kind.stdout.strip() != "blob":
            continue
        size = subprocess.run(
            ["git", "cat-file", "-s", object_id], cwd=root,
            capture_output=True, text=True,
        )
        if size.returncode or not size.stdout.strip().isdigit():
            errors.append(f"could not inspect historical blob: {historical_path}")
            continue
        if int(size.stdout.strip()) > MAX_BYTES:
            errors.append(f"historical file over 10 MB: {historical_path}")
            continue
        if suffix in BANNED_ARTIFACT_SUFFIXES:
            errors.append(f"historical model/data artifact: {historical_path}")
            continue
        blob = subprocess.run(
            ["git", "cat-file", "blob", object_id], cwd=root,
            capture_output=True,
        )
        if blob.returncode:
            errors.append(f"could not read historical binary blob: {historical_path}")
            continue
        digest = hashlib.sha256(blob.stdout).hexdigest()
        if digest not in approved:
            errors.append(f"historical raster/PDF lacks approved provenance: {historical_path}")


def verify_imported_release(root: Path, package: Path, expected_receipt_sha256: str) -> dict[str, object]:
    """Rebuild expected public outputs from the independently pinned package."""
    expected = run_import(
        package, expected_receipt_sha256, repo_root=root, dry_run=True,
        require_release_metadata=True,
    )
    receipt_path = root / "results" / "stage5_import_receipt.json"
    receipt_bytes = _regular_bytes(receipt_path, "Stage-5 import receipt")
    if _sha256(receipt_bytes) != expected["import_receipt_sha256"]:
        raise Stage5ImportError("Stage-5 import receipt differs from source-derived expectation")
    receipt = _validate_import_receipt(root, receipt_bytes)
    if receipt["source_receipt"]["sha256"] != expected_receipt_sha256:
        raise Stage5ImportError("Stage-5 source receipt identity changed")
    if receipt["outputs"] != expected["output_sha256"]:
        raise Stage5ImportError("Stage-5 imported output set differs from source-derived expectation")
    approved = validate_figure_manifest(
        root, expected_source_receipt_sha256=expected_receipt_sha256,
    )
    required = {
        f"figures/{stem}{suffix}" for stem in FIGURE_KINDS for suffix in MIME_BY_SUFFIX
    }
    if set(approved) != required:
        raise Stage5ImportError("aggregate figure manifest is incomplete")
    return approved


def verify_provisional_release(root: Path, expected_receipt_sha256: str) -> dict[str, object]:
    """Verify the separately authorised partial release without weakening final.

    The independently supplied digest binds a tracked receipt.  That receipt in
    turn binds every public aggregate file and the exact pending universes.  A
    provisional scan never treats validation output as a missing test result and
    never approves aggregate figures.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", expected_receipt_sha256):
        raise Stage5ImportError("provisional receipt digest must be lowercase SHA-256")
    receipt_path = root / "results" / "provisional_release_receipt.json"
    receipt_bytes = _regular_bytes(receipt_path, "provisional release receipt")
    if _sha256(receipt_bytes) != expected_receipt_sha256:
        raise Stage5ImportError("provisional release receipt differs from independently pinned digest")
    try:
        receipt = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage5ImportError("provisional release receipt is not valid JSON") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != 1
        or receipt.get("kind") != PROVISIONAL_RECEIPT_KIND
        or receipt.get("status") != "PASS"
        or receipt.get("contains_per_image_rows") is not False
        or receipt.get("contains_biometric_media") is not False
    ):
        raise Stage5ImportError("provisional release receipt contract mismatch")
    if receipt.get("pending_table_a_methods") != ["segformer_b2"]:
        raise Stage5ImportError("provisional Dataset-1 pending universe changed")
    if receipt.get("complete_b1_rows") != sorted(PROVISIONAL_B1_COMPLETE):
        raise Stage5ImportError("provisional B1 complete universe changed")
    if receipt.get("pending_b1_rows") != sorted(PROVISIONAL_B1_PENDING):
        raise Stage5ImportError("provisional B1 pending universe changed")

    output_files = receipt.get("output_files")
    if not isinstance(output_files, dict) or set(output_files) != set(PROVISIONAL_OUTPUTS):
        raise Stage5ImportError("provisional output inventory changed")
    for name in PROVISIONAL_OUTPUTS:
        path = root / "results" / name
        if output_files[name] != _sha256(_regular_bytes(path, f"provisional {name}")):
            raise Stage5ImportError(f"provisional output is not receipt-bound: {name}")
    manifest = {
        "schema_version": 1,
        "kind": "stage4_public_aggregate_bundle",
        "files": output_files,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if receipt.get("output_manifest_sha256") != _sha256(manifest_bytes):
        raise Stage5ImportError("provisional aggregate manifest identity changed")

    with (root / "results" / "public_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    forbidden = {"image_id", "subject_id", "image_path", "mask_path", "filename"}
    if not rows or forbidden.intersection(rows[0]):
        raise Stage5ImportError("provisional results contain per-image identifiers")
    table_a = [
        row for row in rows
        if row.get("table") == "A" and row.get("view") == "overall"
        and row.get("group") == "overall"
    ]
    pending_a = [row for row in table_a if row.get("status") == "pending"]
    if (
        len(table_a) != 18
        or len(pending_a) != 1
        or pending_a[0].get("method") != "segformer_b2"
        or pending_a[0].get("seeds_completed") != "2"
        or "training — pending" not in pending_a[0].get("comment", "")
    ):
        raise Stage5ImportError("provisional Dataset-1 row state differs from 2/3 contract")
    b1 = [
        row for row in rows
        if row.get("table") == "B1" and row.get("view") == "overall"
        and row.get("group") == "overall"
    ]
    b1_complete = {row.get("method") for row in b1 if row.get("status") == "complete"}
    b1_pending = {row.get("method") for row in b1 if row.get("status") == "pending"}
    if b1_complete != PROVISIONAL_B1_COMPLETE or b1_pending != PROVISIONAL_B1_PENDING:
        raise Stage5ImportError("provisional B1 result rows differ from authenticated universes")
    if any(
        row.get(key, "")
        for row in b1
        for key in ("accuracy", "error_rate", "dice", "jaccard", "miou", "mdr", "fdr")
    ):
        raise Stage5ImportError("provisional B1 rows incorrectly contain full-mask metrics")
    return receipt


def git_tree_status(root: Path) -> tuple[bool | None, str]:
    """Return clean state and immutable HEAD; ``None`` means not a Git tree."""
    probe = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], cwd=root,
        capture_output=True, text=True,
    )
    if probe.returncode or probe.stdout.strip() != "true":
        return None, ""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root,
        capture_output=True, text=True,
    )
    if head.returncode or status.returncode:
        return False, ""
    return not status.stdout.strip(), head.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        help="optional JSON report; final-phase reports must be outside the repository",
    )
    parser.add_argument(
        "--phase", choices=("working", "final", "provisional-final"), default="working",
        help=(
            "working permits the deliberate dirty tree created by a controlled import; "
            "final requires the complete Stage-5 package; provisional-final is the "
            "separate receipt-bound partial-release path. Both final modes require an "
            "unchanged committed tree and an external report path"
        ),
    )
    parser.add_argument(
        "--allow-limited-synthetic",
        action="store_true",
        help="CI-only: accept a receipt-valid subset build with at least one completed method",
    )
    parser.add_argument(
        "--stage5-package", type=Path,
        help="canonical package used to re-derive imported outputs for a publication scan",
    )
    parser.add_argument(
        "--expected-stage5-receipt-sha256",
        help="independently supplied SHA-256 of the canonical package receipt",
    )
    parser.add_argument(
        "--expected-provisional-receipt-sha256",
        help="independently supplied SHA-256 of the tracked provisional release receipt",
    )
    args = parser.parse_args()
    errors: list[str] = []
    clean_at_start, head = git_tree_status(ROOT)
    if args.phase in {"final", "provisional-final"}:
        try:
            (args.out or ROOT / "release_scan.json").resolve().relative_to(ROOT.resolve())
        except ValueError:
            pass
        else:
            errors.append("final scan report must be written outside the release repository")
        if clean_at_start is not True:
            errors.append("final scan requires a clean committed Git worktree")
    approved_hashes = approved_synthetic_hashes()
    try:
        manifest_payload = json.loads(
            (ROOT / "figures" / "figure_manifest.json").read_text(encoding="utf-8")
        )
        complete_figures = manifest_payload.get("status") == "complete"
        provenance_args = bool(args.stage5_package) and bool(args.expected_stage5_receipt_sha256)
        if bool(args.stage5_package) != bool(args.expected_stage5_receipt_sha256):
            raise Stage5ImportError("both Stage-5 provenance arguments are required together")
        if args.phase == "provisional-final":
            if complete_figures or provenance_args:
                raise Stage5ImportError(
                    "provisional-final cannot claim the complete Stage-5 figure package"
                )
            if not args.expected_provisional_receipt_sha256:
                raise Stage5ImportError(
                    "provisional-final requires an independently pinned provisional receipt"
                )
            provisional_receipt = verify_provisional_release(
                ROOT, args.expected_provisional_receipt_sha256,
            )
            approved_figures = validate_figure_manifest(ROOT)
        else:
            if args.expected_provisional_receipt_sha256:
                raise Stage5ImportError(
                    "the normal final/working path cannot claim provisional provenance"
                )
            if args.phase == "final" and not complete_figures:
                raise Stage5ImportError("final publication scan requires imported aggregate figures")
            if complete_figures:
                if not provenance_args:
                    raise Stage5ImportError("imported figures require independent Stage-5 source verification")
                approved_figures = verify_imported_release(
                    ROOT, args.stage5_package, args.expected_stage5_receipt_sha256,
                )
            else:
                if provenance_args:
                    raise Stage5ImportError("pending release cannot claim a Stage-5 publication source")
                approved_figures = validate_figure_manifest(ROOT)
    except (OSError, FigureManifestError, json.JSONDecodeError) as exc:
        approved_figures = {}
        errors.append(f"aggregate figure manifest failed validation: {type(exc).__name__}")
    except Stage5ImportError as exc:
        approved_figures = {}
        errors.append(f"Stage-5 publication provenance failed validation: {exc}")
    synthetic_root = ROOT / "examples" / "synthetic"
    synthetic_receipt = synthetic_root / "synthetic_receipt.json"
    try:
        payload = json.loads(synthetic_receipt.read_text(encoding="utf-8"))
        index = validate_synthetic_demo_receipt(synthetic_root, payload)
        statuses = [row["status"] for row in index["methods"].values()]
        expected_complete = 14
        if statuses.count("unavailable") != 2:
            errors.append("synthetic demo does not declare both external-weight methods unavailable")
        if args.allow_limited_synthetic:
            if statuses.count("complete") < 1:
                errors.append("limited synthetic demo executes no method")
        elif statuses.count("complete") != expected_complete:
            errors.append(
                f"synthetic demo has {statuses.count('complete')} complete methods; "
                f"release requires {expected_complete}"
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"synthetic method-demo receipt failed validation: {type(exc).__name__}")
    checked = 0
    for path in files():
        checked += 1
        rel = path.relative_to(ROOT)
        _scan_text(
            rel.as_posix(),
            f"repository path {rel}",
            errors,
            allow_identifier_explanation=rel == Path("docs/DATASETS.md"),
        )
        if path.is_symlink():
            errors.append(f"symlink: {rel}")
            continue
        size = path.stat().st_size
        if size > MAX_BYTES:
            errors.append(f"file over 10 MB: {rel}")
        if path.suffix.lower() in BANNED_ARTIFACT_SUFFIXES:
            errors.append(f"model/data artifact extension: {rel}")
        if path.suffix.lower() in RASTER_SUFFIXES:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if rel.parts[:2] == ("examples", "synthetic") or rel.parts[:3] == ("site", "assets", "synthetic"):
                if digest not in approved_hashes:
                    errors.append(f"raster lacks synthetic receipt provenance: {rel}")
            elif rel.parts[:1] == ("figures",):
                record = approved_figures.get(rel.as_posix())
                if record is None or record.get("sha256") != digest:
                    errors.append(f"aggregate raster is absent from the figure manifest: {rel}")
            else:
                errors.append(f"raster outside approved synthetic/figure directories: {rel}")
        if path.suffix.lower() == ".pdf":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            record = approved_figures.get(rel.as_posix())
            if rel.parts[:1] != ("figures",) or record is None or record.get("sha256") != digest:
                errors.append(f"aggregate PDF is absent from the figure manifest: {rel}")
        if (
            path.suffix.lower() not in TEXT_SUFFIXES | RASTER_SUFFIXES | AGGREGATE_FIGURE_SUFFIXES
            and path.name not in ALLOWED_NON_TEXT_NAMES
        ):
            errors.append(f"unapproved binary or file type: {rel}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in ALLOWED_NON_TEXT_NAMES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF8 text artifact: {rel}")
            continue
        _scan_text(
            text,
            str(rel),
            errors,
            allow_identifier_explanation=path == ROOT / "docs" / "DATASETS.md",
        )
    scan_history(
        errors, approved_binary_hashes=(
            approved_hashes | {str(row.get("sha256")) for row in approved_figures.values()}
        ),
    )
    clean_at_end, end_head = git_tree_status(ROOT)
    if args.phase in {"final", "provisional-final"} and (
        clean_at_end is not True or end_head != head
    ):
        errors.append("Git worktree or HEAD changed during the final scan")
    report = {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "files_checked": checked,
        "phase": args.phase,
        "git_tree_clean": clean_at_end,
        "git_commit": end_head,
        "approved_synthetic_hashes": len(approved_hashes),
        "approved_aggregate_figures": len(approved_figures),
        "stage5_source_verified": bool(approved_figures),
        "stage5_source_receipt_sha256": (
            args.expected_stage5_receipt_sha256 if approved_figures else None
        ),
        "provisional_source_verified": bool(
            args.phase == "provisional-final" and 'provisional_receipt' in locals()
        ),
        "provisional_receipt_sha256": (
            args.expected_provisional_receipt_sha256
            if args.phase == "provisional-final" and 'provisional_receipt' in locals()
            else None
        ),
        "checks": [
            "no biometric filename patterns outside the naming explanation",
            "no private absolute paths, cloud-drive references, or IP addresses",
            "no local infrastructure hostnames or opaque cloud-drive-like identifiers",
            "no trained-weight or data-array artifacts",
            "all raster assets are receipt-bound synthetic images or manifest-bound aggregate plots",
            "all aggregate PDFs are manifest-bound and source-receipt-bound",
            (
                "provisional aggregate outputs are bound to an independently pinned receipt"
                if args.phase == "provisional-final"
                else "final aggregate outputs are re-derived from the independently pinned Stage-5 package"
            ),
            "synthetic method outputs pass provenance validation and release completeness",
            "no unapproved binary file types",
            "no file exceeds 10 MB",
            "git patch history has no private paths, identifiers, infrastructure, or cloud-drive tokens",
            "final phase requires one clean immutable Git tree and an external scan report",
        ],
        "errors": errors,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
