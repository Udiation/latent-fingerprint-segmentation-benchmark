"""Shared validation for SD302g cross-domain inference authorizations.

The authorization preserves the Dataset-1 validation selector verbatim while
permitting that already-selected model to consume a ground-truth-blind SD302g
manifest.  It is deliberately narrower than a general manifest override.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from .artifact_identity import sha256_file


AUTHORIZATION_KIND = "sd302g_b1_crossdomain_inference_authorization"
AUTHORIZATION_SCHEMA_VERSION = 1
PROTOCOL = "sd302g-minutiae-lower-bound-v2"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def load_json_object(path: Path, label: str) -> Mapping[str, Any]:
    """Load one JSON object with a useful fail-closed error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def validate_for_inference(
    path: Path,
    *,
    method: str,
    task: str,
    seed: int,
    config_path: Path,
    checkpoint_path: Path,
    selector_path: Path,
    amendment_path: Path,
    target_manifest_path: Path,
    split_path: Path,
    subset: str,
    split: str,
    limit: int,
) -> tuple[Mapping[str, Any], str]:
    """Authenticate a production B1 authorization before any image is decoded."""
    authorization = load_json_object(path, "B1 cross-domain authorization")
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported B1 cross-domain authorization schema")
    if authorization.get("kind") != AUTHORIZATION_KIND:
        raise ValueError("wrong B1 cross-domain authorization kind")
    if authorization.get("protocol") != PROTOCOL:
        raise ValueError("wrong B1 cross-domain protocol")
    if authorization.get("status") != "AUTHORIZED":
        raise ValueError("B1 cross-domain inference is not authorized")
    if split != "test" or subset != "finger_exemplar" or limit != 0:
        raise ValueError(
            "B1 cross-domain authorization permits only the complete exemplar test split"
        )

    model = authorization.get("model")
    target = authorization.get("target_domain")
    constraints = authorization.get("constraints")
    if not isinstance(model, dict) or not isinstance(target, dict):
        raise ValueError("B1 authorization lacks model/target bindings")
    if not isinstance(constraints, dict):
        raise ValueError("B1 authorization lacks protocol constraints")
    expected_model = {
        "method": method,
        "task": task,
        "seed": int(seed),
        "source_config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selector_receipt_sha256": sha256_file(selector_path),
        "threshold_amendment_sha256": sha256_file(amendment_path),
    }
    disagreements = {
        key: (model.get(key), value)
        for key, value in expected_model.items()
        if model.get(key) != value
    }
    if disagreements:
        raise ValueError(f"B1 authorization model binding changed: {disagreements}")
    expected_target = {
        "manifest_sha256": sha256_file(target_manifest_path),
        "split_sha256": sha256_file(split_path),
        "subset": "finger_exemplar",
        "split": "test",
        "mask_paths_empty": True,
    }
    disagreements = {
        key: (target.get(key), value)
        for key, value in expected_target.items()
        if target.get(key) != value
    }
    if disagreements:
        raise ValueError(f"B1 authorization target binding changed: {disagreements}")
    required_constraints = {
        "validation_selector_reused_without_change": True,
        "target_labels_opened": False,
        "sd302g_supervised_fitting": False,
        "hulls_used_for_inference": False,
        "minutiae_used_for_inference": False,
        "probability_maps_required": True,
        "score_once_required": True,
    }
    disagreements = {
        key: (constraints.get(key), value)
        for key, value in required_constraints.items()
        if constraints.get(key) is not value
    }
    if disagreements:
        raise ValueError(f"B1 authorization constraint changed: {disagreements}")
    source_manifest_sha256 = model.get("selector_source_manifest_sha256")
    if not isinstance(source_manifest_sha256, str) or not HEX64.fullmatch(
        source_manifest_sha256
    ):
        raise ValueError("B1 authorization lacks the Dataset-1 selector manifest identity")
    return authorization, sha256_file(path)
