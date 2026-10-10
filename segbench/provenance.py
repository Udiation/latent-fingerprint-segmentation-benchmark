"""Lightweight run provenance for CPU-only entry points.

This module deliberately avoids importing the optional deep-learning stack so
classical-method reproduction and the synthetic CI can run with the base
dependencies alone.  Package versions are read from installed distribution
metadata; no package is imported merely to report its version.
"""
from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Sequence


def _distribution_version(name: str) -> str:
    """Return an installed distribution version, or ``unavailable``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def run_provenance(
    command: Sequence[str],
    *,
    capture_timing: str = "function_entry",
    repository: Path | str | None = None,
) -> Dict[str, Any]:
    """Snapshot reproducibility metadata without importing optional Torch."""
    if capture_timing not in {"process_start", "function_entry"}:
        raise ValueError("capture_timing must be process_start or function_entry")
    repository_path = (
        Path(repository) if repository is not None else Path(__file__).resolve().parents[1]
    )
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        git_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=repository_path,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_commit, git_dirty = "unavailable", None
    return {
        "command_argv": list(command),
        "capture_timing": capture_timing,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "versions": {
            "python": platform.python_version(),
            "torch": _distribution_version("torch"),
            "numpy": _distribution_version("numpy"),
            "opencv": _distribution_version("opencv-python-headless"),
            "pandas": _distribution_version("pandas"),
            "pyyaml": _distribution_version("PyYAML"),
        },
    }
