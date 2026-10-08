"""Synthetic tests for the isolated DRUnet MC-dropout auxiliary wrapper."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from segbench.artifact_identity import canonical_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "infer_drunet_mc_dropout.py"
    spec = importlib.util.spec_from_file_location("drunet_aux_test", path)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class _FakeModel:
    def load_state_dict(self, *_args, **_kwargs):
        return None

    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self


def test_stale_shape_valid_maps_are_recomputed_and_card_has_no_masks(
    tmp_path: Path, monkeypatch,
) -> None:
    module = _module()
    config = tmp_path / "drunet.yaml"
    checkpoint = tmp_path / "best.pt"
    manifest = tmp_path / "manifest.csv"
    splits = tmp_path / "splits.json"
    for path, content in (
        (config, b"synthetic config"),
        (checkpoint, b"synthetic checkpoint"),
        (manifest, b"synthetic manifest"),
        (splits, b"synthetic splits"),
    ):
        path.write_bytes(content)
    ids = [f"img{index:02d}" for index in range(20)]
    source_records = [
        {"image_id": image_id, "source_sha256": "a" * 64}
        for image_id in ids
    ]
    commit = "b" * 40
    variance = tmp_path / "variance"
    work = tmp_path / "work"
    card = tmp_path / "card.json"
    plan_path = tmp_path / "plan.json"
    plan = {
        "schema_version": 1,
        "kind": "stage3_drunet_mc_dropout_plan",
        "ground_truth_blind": True,
        "test_inputs_hashed": True,
        "test_images_decoded": False,
        "test_labels_opened": False,
        "timing_excluded": True,
        "mask_outputs_forbidden": True,
        "mc_dropout_passes": 20,
        "test_image_ids": ids,
        "image_id_set_sha256": canonical_sha256(ids),
        "ordered_source_images_sha256": canonical_sha256(source_records),
        "expected_shapes": {image_id: [2, 3] for image_id in ids},
        "run": {
            "method": "drunet",
            "seed": 0,
            "checkpoint_sha256": sha256_file(checkpoint),
            "expected_inference_git_commit": commit,
        },
        "identities": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in {
                "config": config,
                "checkpoint": checkpoint,
                "manifest": manifest,
                "splits": splits,
            }.items()
        },
        "variance_out": str(variance),
        "card_out": str(card),
        "work_dir": str(work),
    }
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    variance.mkdir()
    for image_id in ids:
        np.save(variance / f"{image_id}_variance.npy", np.full((2, 3), 99, np.float32))
    rows = pd.DataFrame(
        [
            {
                "image_id": image_id,
                "source_sha256": "a" * 64,
                "image_path": str(tmp_path / f"{image_id}.png"),
            }
            for image_id in ids
        ]
    )
    monkeypatch.setattr(
        module,
        "load_config",
        lambda _path: {
            "method": "drunet",
            "task": "pixel",
            "seed": 0,
            "model": {},
            "data": {"mask_column": "mask_path", "target_ppi": 500},
            "inference": {"pad_multiple": 32, "tile_size": 0, "tile_overlap": 0},
            "runtime": {"device": "cpu", "memory_fraction": 0.3, "deterministic": True},
        },
    )
    monkeypatch.setattr(
        module.torch,
        "load",
        lambda *_args, **_kwargs: {
            "config": {"seed": 0},
            "resume_config_sha256": "resume",
            "method": "drunet",
            "task": "pixel",
            "normalization": {"source_split": "train", "mean": 0.5, "std": 0.2},
            "model_state": {},
        },
    )
    monkeypatch.setattr(module, "resume_config_digest", lambda _config: "resume")
    monkeypatch.setattr(module, "load_split_rows", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(module, "verify_source_image", lambda row: row["source_sha256"])
    monkeypatch.setattr(module, "read_normalized_image", lambda *_args, **_kwargs: np.zeros((2, 3), np.float32))
    monkeypatch.setattr(module, "resolve_device", lambda _value: module.torch.device("cpu"))
    monkeypatch.setattr(module, "configure_memory_fraction", lambda *_args: None)
    monkeypatch.setattr(module, "seed_everything", lambda *_args: None)
    monkeypatch.setattr(module, "build_model", lambda *_args: _FakeModel())
    calls = []

    def recompute(*_args, **_kwargs):
        calls.append(True)
        return np.full((2, 3), len(calls), np.float32)

    monkeypatch.setattr(module.infer_masks, "_mc_dropout_variance", recompute)
    args = SimpleNamespace(
        plan=plan_path,
        expected_plan_sha256=sha256_file(plan_path),
        config=config,
        checkpoint=checkpoint,
        manifest=manifest,
        splits=splits,
        variance_out=variance,
        card_out=card,
        work_dir=work,
        expected_git_commit=commit,
        expected_checkpoint_sha256=sha256_file(checkpoint),
        seed=0,
        device="cpu",
        memory_fraction=0.3,
    )
    provenance = {
        "capture_timing": "process_start",
        "git_commit": commit,
        "git_dirty": False,
    }
    result = module.run(args, provenance)
    assert len(calls) == 20
    assert result["mask_outputs_written"] is False
    assert result["timing_excluded"] is True
    assert result["variance_set"]["count"] == 20
    assert all(record["shape"] == [2, 3] for record in result["variance_set"]["records"])
    assert all(record["dtype"] == "float32" for record in result["variance_set"]["records"])
    assert not list(variance.rglob("*.png"))

    calls.clear()
    replay = module.run(args, provenance)
    assert calls == []
    assert replay == result
