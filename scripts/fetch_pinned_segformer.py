#!/usr/bin/env python3
"""Fetch the one registered SegFormer snapshot and prove its local identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

import _path  # noqa: F401
from segbench.artifact_identity import atomic_json_once
from segbench.deep_models import verify_pinned_pretrained_snapshot

MODEL_ID = "nvidia/mit-b2"
REVISION = "3bb39e8739149c3777d0325349b2a6c32c6413db"
FILES = {
    "config.json": {"git_blob_sha1": "1e5025b00cf3728234caf34ba1d0f9b55c8fb79a"},
    "preprocessor_config.json": {
        "git_blob_sha1": "b454cd05f17438fae28f9f201e2b4f7ef7fc5dda"
    },
    "pytorch_model.bin": {
        "sha256": "4500b5665471b593e6757e15bcca5034f433fe3902fe8ec2b7230774a57f264f",
        "bytes": 98978917,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--card-out", required=True, type=Path)
    args = parser.parse_args()
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            cache_dir=args.cache_dir,
            allow_patterns=list(FILES),
        )
    ).resolve()
    observed = verify_pinned_pretrained_snapshot(snapshot, FILES)
    card = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "snapshot": str(snapshot),
        "files": observed,
        "classification_head_use": "discarded",
        "redistribution": "not_authorized_by_this_record",
    }
    atomic_json_once(args.card_out, card)
    print(json.dumps({"snapshot": str(snapshot), "card": str(args.card_out)}))


if __name__ == "__main__":
    main()
