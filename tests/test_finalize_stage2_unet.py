"""Data-free tests for the one-time Stage 2 U-Net delivery barrier."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _module():
    path = ROOT / "scripts" / "finalize_stage2_unet.py"
    spec = importlib.util.spec_from_file_location("finalize_stage2_test", path)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def _rows() -> pd.DataFrame:
    values = ["VALUE"] * 8 + ["LIMITED"] * 7 + ["NOVALUE"] * 7
    return pd.DataFrame(
        {
            "image_id": [f"synthetic_{index:02d}" for index in range(len(values))],
            "examiner_value": values,
        }
    )


def test_qa_selection_is_deterministic_stratified_and_score_free() -> None:
    module = _module()
    frame = _rows()
    first = module.deterministic_overlay_rows(frame)
    second = module.deterministic_overlay_rows(frame.sample(frac=1, random_state=9))
    assert first.image_id.tolist() == second.image_id.tolist()
    assert first.groupby("examiner_value").size().to_dict() == {
        "LIMITED": 3,
        "NOVALUE": 3,
        "VALUE": 4,
    }
    assert len(first) == 10
    assert not any("dice" in column.lower() for column in frame.columns)


def test_qa_selection_fails_when_registered_stratum_is_missing() -> None:
    module = _module()
    with pytest.raises(ValueError, match="not enough NOVALUE"):
        module.deterministic_overlay_rows(_rows().query("examiner_value != 'NOVALUE'"))


def test_gate_must_be_unqualified_pass(tmp_path: Path) -> None:
    module = _module()
    parser = type("Args", (), {})()
    parser.gate = tmp_path / "gate.json"
    parser.gate.write_text(
        '{"schema_version":1,"gate":"stage2_unet_seed0_validation_health",'
        '"status":"FAIL","failures":["synthetic"]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unqualified PASS"):
        module.validate_gate(parser)


def test_aggregate_nan_policy_matches_empty_class_denominators() -> None:
    module = _module()
    no_foreground = pd.DataFrame(
        {"tp": [0, 0], "fp": [2, 0], "fn": [0, 0], "tn": [8, 10], "mdr": [float("nan"), float("nan")], "fdr": [0.2, 0.0]}
    )
    assert module._aggregate_nan_expected(no_foreground, "mdr", "_mean")
    assert module._aggregate_nan_expected(no_foreground, "mdr", "_pooled")
    assert not module._aggregate_nan_expected(no_foreground, "fdr", "_mean")
    assert not module._aggregate_nan_expected(no_foreground, "fdr", "_pooled")


def _scoring_case(tmp_path: Path):
    module = _module()
    from segbench.metrics import metrics_from_confusion

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best.pt").write_bytes(b"synthetic-best-checkpoint")
    groups = ["VALUE", "LIMITED", "NOVALUE", "MIXED"] * 2
    manifest_rows = []
    per_rows = []
    splits = {}
    for index, group in enumerate(groups):
        image_id = f"synthetic_{index:02d}"
        subject_id = f"subject_{index:02d}"
        splits[subject_id] = "test"
        manifest_rows.append(
            {
                "image_id": image_id,
                "image_path": f"/synthetic/{image_id}.png",
                "mask_path": f"/synthetic/{image_id}_roi.png",
                "quality_mask_path": f"/synthetic/{image_id}_quality.png",
                "subject_id": subject_id,
                "subset": "finger_latent",
                "ppi": 1000,
                "examiner_value": group,
            }
        )
        tp, fp, fn, tn = 80 + index, 5 + index % 2, 7, 90
        record = metrics_from_confusion(tp, fp, fn, tn)
        record.update(
            {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "image_id": image_id,
                "subject_id": subject_id,
                "missing": False,
                "examiner_value": group,
            }
        )
        per_rows.append(record)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps({"seed": 42, "splits": splits}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    per = pd.DataFrame(per_rows)
    checkpoint_sha = module.sha256_file(run_dir / "best.pt")
    summary_rows = []
    for table, gt_column in (("A", "mask_path"), ("A_prime", "quality_mask_path")):
        per_path = stage / (
            f"per_image_unet_finger_latent_{table}_seed0_identity-synthetic.csv"
        )
        per.to_csv(per_path, index=False)
        for group_value, group_frame in [
            ("overall", per),
            *[(value, part) for value, part in per.groupby("examiner_value")],
        ]:
            row = module.summarise(group_frame)
            row.update(
                {
                    "method": "unet",
                    "display_name": "U-Net",
                    "subset": "finger_latent",
                    "split": "test",
                    "ground_truth": gt_column,
                    "table": table,
                    "group_by": "examiner_value",
                    "group_value": group_value,
                    "seed": 0,
                    "training_data": "SD302-Latent train split",
                    "device": "GPU",
                    "run_id": checkpoint_sha,
                    "n_missing": 0,
                }
            )
            summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(stage / "summary.csv", index=False)
    args = SimpleNamespace(
        manifest=manifest,
        splits=split_path,
        run_dir=run_dir,
        device="cuda",
    )
    return SimpleNamespace(module=module, stage=stage, args=args)


def test_scoring_validator_accepts_exact_manifest_derived_mixed_group(
    tmp_path: Path,
) -> None:
    case = _scoring_case(tmp_path)
    evidence = case.module.validate_scoring(case.stage, case.args)
    assert evidence["A"]["groups"] == {
        "overall": 8,
        "LIMITED": 2,
        "MIXED": 2,
        "NOVALUE": 2,
        "VALUE": 2,
    }
    assert evidence["A_prime"]["groups"] == evidence["A"]["groups"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_mixed", "exact manifest universe"),
        ("extra_group", "exact manifest universe"),
        ("duplicate_group", "duplicate table/group"),
        ("wrong_count", "image counts differ"),
        ("swapped_ground_truth", "ground_truth identity changed"),
        ("aggregate_drift", "does not reproduce"),
    ],
)
def test_scoring_validator_rejects_group_or_aggregate_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    case = _scoring_case(tmp_path)
    path = case.stage / "summary.csv"
    summary = pd.read_csv(path)
    selector = (summary.table == "A") & (summary.group_value == "MIXED")
    if mutation == "missing_mixed":
        summary = summary[~selector]
    elif mutation == "extra_group":
        extra = summary[selector].copy()
        extra["group_value"] = "UNREGISTERED"
        summary = pd.concat([summary, extra], ignore_index=True)
    elif mutation == "duplicate_group":
        summary = pd.concat([summary, summary[selector]], ignore_index=True)
    elif mutation == "wrong_count":
        summary.loc[selector, "n_images"] += 1
    elif mutation == "swapped_ground_truth":
        summary.loc[summary.table == "A", "ground_truth"] = "quality_mask_path"
    elif mutation == "aggregate_drift":
        summary.loc[selector, "dice_mean"] -= 0.01
    else:  # pragma: no cover - test declaration controls this value
        raise AssertionError(mutation)
    summary.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        case.module.validate_scoring(case.stage, case.args)


def test_recovery_binds_both_common_evaluator_per_image_files(
    tmp_path: Path,
) -> None:
    module = _module()
    results = tmp_path / "results"
    results.mkdir()
    evidence = {}
    for table in ("A", "A_prime"):
        name = f"per_image_{table}.csv"
        value = f"synthetic-{table}\n".encode()
        (results / name).write_bytes(value)
        evidence[table] = {
            "per_image_file": name,
            "per_image_sha256": hashlib.sha256(value).hexdigest(),
        }
    outputs = {"scoring_evidence": evidence}
    module._validate_per_image_output_hashes(tmp_path, outputs)
    (results / "per_image_A_prime.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="A_prime per-image evidence changed"):
        module._validate_per_image_output_hashes(tmp_path, outputs)


def test_journal_updates_preserve_original_command_and_provenance(tmp_path: Path) -> None:
    module = _module()
    journal = tmp_path / "journal.json"
    module._publish_journal(
        journal,
        "started",
        command=["synthetic", "--fixed"],
        provenance={"git_commit": "a" * 40},
        test_scoring_started=False,
    )
    module._publish_journal(
        journal,
        "inference_started",
        checkpoint_sha256="b" * 64,
        test_inference_started=True,
    )
    value = module._read_json(journal, "journal")
    assert value["command"] == ["synthetic", "--fixed"]
    assert value["provenance"]["git_commit"] == "a" * 40
    assert value["checkpoint_sha256"] == "b" * 64
    assert value["test_scoring_started"] is False


def _recovery_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _module()
    old_commit = "a" * 40
    new_commit = "b" * 40
    journal = tmp_path / ".delivery.journal.json"
    original = {
        "schema_version": 1,
        "status": "failed",
        "command": ["legacy-finalizer", "--fixed"],
        "provenance": {
            "capture_timing": "process_start",
            "git_commit": old_commit,
            "git_dirty": False,
        },
        "test_inference_started": True,
        "test_scoring_started": False,
    }
    journal.write_text(json.dumps(original, sort_keys=True) + "\n", encoding="utf-8")
    original_bytes = journal.read_bytes()

    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "queue_sha256": module.EXPECTED_FAILED_QUEUE_SHA256,
                "jobs": {
                    "stage2.unet.delivery": {
                        "attempts": 2,
                        "status": "failed",
                        "exit_code": 1,
                        "identity_sha256": module.EXPECTED_FAILED_JOB_IDENTITY_SHA256,
                        "oom_retries": 0,
                        "ordinary_retries": 1,
                        "preemptions": 0,
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    queue_log = tmp_path / "queue.log"
    queue_log.write_text("first failure\nsecond refusal\n", encoding="utf-8")
    rerun_log = tmp_path / "RERUN_LOG.md"
    rerun_log.write_text("approved bug-forced rerun\n", encoding="utf-8")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("synthetic: config\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    roi_gt = tmp_path / "roi.png"
    roi_gt.write_bytes(b"synthetic-roi-gt")
    quality_gt = tmp_path / "quality.png"
    quality_gt.write_bytes(b"synthetic-quality-gt")
    pd.DataFrame(
        [
            {
                "image_id": "synthetic_source",
                "image_path": "/synthetic/source.png",
                "mask_path": str(roi_gt),
                "quality_mask_path": str(quality_gt),
                "subject_id": "synthetic_subject",
                "subset": "finger_latent",
                "ppi": 1000,
                "roi_union_sha256": module.sha256_file(roi_gt),
                "quality_union_sha256": module.sha256_file(quality_gt),
            }
        ]
    ).to_csv(manifest, index=False)
    splits = tmp_path / "splits.json"
    splits.write_text('{"splits":{}}\n', encoding="utf-8")
    prediction_sha = hashlib.sha256(b"synthetic-prediction-set").hexdigest()
    inference_card = tmp_path / "inference.json"
    card = {
        "prediction_set": {
            "sha256": prediction_sha,
            "count": module.EXPECTED_PREDICTION_COUNT,
        },
        "identities": {
            "prediction_set_sha256": prediction_sha,
            "prediction_count": module.EXPECTED_PREDICTION_COUNT,
        },
        "provenance": {
            "capture_timing": "process_start",
            "git_commit": old_commit,
            "git_dirty": False,
        },
    }
    inference_card.write_text(
        json.dumps(card, sort_keys=True) + "\n", encoding="utf-8"
    )
    gate = tmp_path / "gate.json"
    gate.write_text('{"status":"PASS"}\n', encoding="utf-8")
    recovery_authorization = tmp_path / "recovery.json"
    args = SimpleNamespace(
        bundle_out=tmp_path / "delivery",
        recovery_authorization=recovery_authorization,
        prior_recovery_authorization=None,
        recovery_ledger=ledger,
        recovery_log=queue_log,
        rerun_log=rerun_log,
        inference_card=inference_card,
        config=config,
        manifest=manifest,
        splits=splits,
        expected_inference_git_commit=old_commit,
        expected_finalizer_git_commit=new_commit,
        device="cuda",
    )
    gate_sha = module.sha256_file(gate)
    checkpoint_sha = module.sha256_file(checkpoint)
    gate_authorization = {
        "gate_sha256": gate_sha,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint": checkpoint,
        "run": {"best_epoch": 4, "elapsed_seconds": 12.5},
        "threshold": 0.3,
    }
    payload = {
        "schema_version": 1,
        "kind": "stage2_unet_failed_delivery_recovery",
        "action": module.RECOVERY_ACTION,
        "reason_code": module.RECOVERY_REASON_CODE,
        "authorization_basis": module.RECOVERY_AUTHORIZATION_BASIS,
        "approved": True,
        "failed_journal_sha256": module.sha256_file(journal),
        "failed_ledger_sha256": module.sha256_file(ledger),
        "failed_queue_sha256": module.EXPECTED_FAILED_QUEUE_SHA256,
        "failed_queue_log_sha256": module.sha256_file(queue_log),
        "rerun_log_sha256": module.sha256_file(rerun_log),
        "inference_card_sha256": module.sha256_file(inference_card),
        "gate_sha256": gate_sha,
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": module.sha256_file(config),
        "manifest_sha256": module.sha256_file(manifest),
        "split_sha256": module.sha256_file(splits),
        "prediction_set_sha256": prediction_sha,
        "prediction_count": module.EXPECTED_PREDICTION_COUNT,
        "old_inference_git_commit": old_commit,
        "new_finalizer_git_commit": new_commit,
        "test_scoring_started": False,
        "rerun_inference": False,
        "bundle_absent": True,
    }
    calls = []
    monkeypatch.setattr(
        module,
        "_validate_inference_card",
        lambda observed, observed_args, observed_authorization: calls.append(
            (observed, observed_args, observed_authorization)
        ),
    )
    provenance = {
        "capture_timing": "process_start",
        "git_commit": new_commit,
        "git_dirty": False,
    }
    return SimpleNamespace(
        module=module,
        args=args,
        journal=journal,
        original=original,
        original_bytes=original_bytes,
        payload=payload,
        authorization=gate_authorization,
        provenance=provenance,
        card=card,
        calls=calls,
    )


def _write_recovery_authorization(case) -> None:
    case.args.recovery_authorization.write_text(
        json.dumps(case.payload, sort_keys=True) + "\n", encoding="utf-8"
    )


def _postscore_recovery_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _module()
    old_commit = "a" * 40
    failed_commit = "b" * 40
    new_commit = "c" * 40
    bundle = tmp_path / "delivery"
    journal = module._journal_path(bundle)

    first_failed_bytes = b'{"synthetic":"first failed journal"}\n'
    first_failed_sha = hashlib.sha256(first_failed_bytes).hexdigest()
    first_archive = journal.with_name(
        f"{journal.name}.failed-{first_failed_sha}.archive"
    )
    first_archive.write_bytes(first_failed_bytes)
    first_authorization = tmp_path / "first-recovery.json"
    first_authorization.write_text(
        '{"kind":"synthetic-first-recovery"}\n', encoding="utf-8"
    )
    first_authorization_sha = module.sha256_file(first_authorization)
    original = {
        "schema_version": 1,
        "status": "failed",
        "phase": "scoring",
        "error": module.POST_SCORE_FAILURE_ERROR,
        "test_inference_started": True,
        "test_scoring_started": True,
        "rerun_inference": False,
        "command": ["legacy-finalizer", "--fixed"],
        "provenance": {"git_commit": old_commit, "git_dirty": False},
        "recovery_provenance": {
            "capture_timing": "process_start",
            "git_commit": failed_commit,
            "git_dirty": False,
        },
        "recovery": {
            "action": module.RECOVERY_ACTION,
            "authorization_sha256": first_authorization_sha,
            "failed_journal_sha256": first_failed_sha,
            "failed_journal_archive": first_archive.name,
            "rerun_inference": False,
        },
    }
    journal.write_text(json.dumps(original, sort_keys=True) + "\n", encoding="utf-8")

    ledger = tmp_path / "failed-recovery-ledger.json"
    job_identity = hashlib.sha256(b"synthetic-postscore-job").hexdigest()
    queue_sha = hashlib.sha256(b"synthetic-postscore-queue").hexdigest()
    ledger.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "queue_sha256": queue_sha,
                "jobs": {
                    "stage2.unet.delivery.recovery": {
                        "attempts": 2,
                        "exit_code": 1,
                        "identity_sha256": job_identity,
                        "oom_retries": 0,
                        "ordinary_retries": 1,
                        "preemptions": 0,
                        "status": "failed",
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    queue_log = tmp_path / "failed-recovery.log"
    queue_log.write_text(
        "unet finger_latent: n=2010 missing=0 Dice 0.7800 Acc 0.7539\n"
        "unet finger_latent: n=2009 missing=0 Dice 0.6846 Acc 0.6629\n"
        f"{module.POST_SCORE_FAILURE_ERROR}\n"
        "failed journal shows that held-out scoring already started\n",
        encoding="utf-8",
    )
    rerun_log = tmp_path / "RERUN_LOG.snapshot.md"
    rerun_log.write_text("authorized bug-forced second score\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("synthetic: config\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    roi_gt = tmp_path / "postscore-roi.png"
    roi_gt.write_bytes(b"synthetic-postscore-roi-gt")
    quality_gt = tmp_path / "postscore-quality.png"
    quality_gt.write_bytes(b"synthetic-postscore-quality-gt")
    pd.DataFrame(
        [
            {
                "image_id": "synthetic_postscore_source",
                "image_path": "/synthetic/postscore-source.png",
                "mask_path": str(roi_gt),
                "quality_mask_path": str(quality_gt),
                "subject_id": "synthetic_postscore_subject",
                "subset": "finger_latent",
                "ppi": 1000,
                "roi_union_sha256": module.sha256_file(roi_gt),
                "quality_union_sha256": module.sha256_file(quality_gt),
            }
        ]
    ).to_csv(manifest, index=False)
    splits = tmp_path / "splits.json"
    splits.write_text('{"splits":{}}\n', encoding="utf-8")
    manifest_audit = manifest.with_suffix(".audit.json")
    ordered_gt_sha = module._rehash_ordered_source_gt(manifest)
    manifest_audit.write_text(
        json.dumps(
            {
                "manifest_sha256": module.sha256_file(manifest),
                "ordered_source_gt_sha256": ordered_gt_sha,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    gate = tmp_path / "gate.json"
    gate.write_text('{"status":"PASS"}\n', encoding="utf-8")
    prediction_sha = hashlib.sha256(b"synthetic-predictions").hexdigest()
    inference_card = tmp_path / "inference.json"
    card = {
        "prediction_set": {
            "sha256": prediction_sha,
            "count": module.EXPECTED_PREDICTION_COUNT,
        },
        "identities": {
            "prediction_set_sha256": prediction_sha,
            "prediction_count": module.EXPECTED_PREDICTION_COUNT,
        },
    }
    inference_card.write_text(
        json.dumps(card, sort_keys=True) + "\n", encoding="utf-8"
    )
    authorization_path = tmp_path / "postscore-recovery.json"
    args = SimpleNamespace(
        gate=gate,
        config=config,
        run_dir=tmp_path / "run",
        manifest=manifest,
        splits=splits,
        masks_out=tmp_path / "masks",
        inference_card=inference_card,
        inference_work_dir=tmp_path / "inference-work",
        bundle_out=bundle,
        expected_inference_git_commit=old_commit,
        expected_finalizer_git_commit=new_commit,
        recovery_authorization=authorization_path,
        prior_recovery_authorization=first_authorization,
        recovery_ledger=ledger,
        recovery_log=queue_log,
        rerun_log=rerun_log,
        device="cuda",
    )
    gate_authorization = {
        "gate_sha256": module.sha256_file(gate),
        "checkpoint_sha256": module.sha256_file(checkpoint),
        "checkpoint": checkpoint,
        "run": {"best_epoch": 4, "elapsed_seconds": 12.5},
        "threshold": 0.3,
    }
    code_hashes = {
        "evaluator_sha256": module.sha256_file(
            ROOT / "scripts" / "evaluate_masks.py"
        ),
        "metrics_sha256": module.sha256_file(ROOT / "segbench" / "metrics.py"),
        "io_sha256": module.sha256_file(ROOT / "segbench" / "io.py"),
        "summary_sha256": module.sha256_file(ROOT / "segbench" / "summary.py"),
    }
    payload = {
        "schema_version": 1,
        "kind": module.POST_SCORE_RECOVERY_KIND,
        "action": module.POST_SCORE_RECOVERY_ACTION,
        "reason_code": module.POST_SCORE_RECOVERY_REASON_CODE,
        "authorization_basis": module.POST_SCORE_RECOVERY_AUTHORIZATION_BASIS,
        "approved": True,
        "failed_journal_sha256": module.sha256_file(journal),
        "failed_ledger_sha256": module.sha256_file(ledger),
        "failed_queue_sha256": queue_sha,
        "failed_queue_log_sha256": module.sha256_file(queue_log),
        "first_recovery_authorization_sha256": first_authorization_sha,
        "first_failed_journal_sha256": first_failed_sha,
        "rerun_log_sha256": module.sha256_file(rerun_log),
        "inference_card_sha256": module.sha256_file(inference_card),
        "gate_sha256": gate_authorization["gate_sha256"],
        "checkpoint_sha256": gate_authorization["checkpoint_sha256"],
        "config_sha256": module.sha256_file(config),
        "manifest_sha256": module.sha256_file(manifest),
        "split_sha256": module.sha256_file(splits),
        "manifest_audit_sha256": module.sha256_file(manifest_audit),
        "ordered_source_gt_sha256": ordered_gt_sha,
        **code_hashes,
        "prediction_set_sha256": prediction_sha,
        "prediction_count": module.EXPECTED_PREDICTION_COUNT,
        "old_inference_git_commit": old_commit,
        "failed_finalizer_git_commit": failed_commit,
        "new_finalizer_git_commit": new_commit,
        "prior_scoring_started": True,
        "prior_score_attempts": 1,
        "prior_evaluator_invocations": 2,
        "prior_outputs_persisted": False,
        "rerun_inference": False,
        "rerun_scoring": True,
        "bundle_absent": True,
        "surviving_score_artifacts_absent": True,
        "score_attempt_ordinal": 2,
    }
    authorization_path.write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )

    constants = {
        "EXPECTED_POST_SCORE_FAILED_JOURNAL_SHA256": payload[
            "failed_journal_sha256"
        ],
        "EXPECTED_POST_SCORE_FAILED_LEDGER_SHA256": payload[
            "failed_ledger_sha256"
        ],
        "EXPECTED_POST_SCORE_FAILED_QUEUE_SHA256": queue_sha,
        "EXPECTED_POST_SCORE_FAILED_QUEUE_LOG_SHA256": payload[
            "failed_queue_log_sha256"
        ],
        "EXPECTED_POST_SCORE_FAILED_JOB_IDENTITY_SHA256": job_identity,
        "EXPECTED_FIRST_RECOVERY_AUTHORIZATION_SHA256": first_authorization_sha,
        "EXPECTED_POST_SCORE_RERUN_LOG_SHA256": payload["rerun_log_sha256"],
        "EXPECTED_INFERENCE_CARD_SHA256": payload["inference_card_sha256"],
        "EXPECTED_PREDICTION_SET_SHA256": prediction_sha,
        "EXPECTED_STAGE2_GATE_SHA256": payload["gate_sha256"],
        "EXPECTED_STAGE2_CHECKPOINT_SHA256": payload["checkpoint_sha256"],
        "EXPECTED_STAGE2_CONFIG_SHA256": payload["config_sha256"],
        "EXPECTED_MANIFEST_SHA256": payload["manifest_sha256"],
        "EXPECTED_SPLIT_SHA256": payload["split_sha256"],
        "EXPECTED_MANIFEST_AUDIT_SHA256": payload["manifest_audit_sha256"],
        "EXPECTED_ORDERED_SOURCE_GT_SHA256": ordered_gt_sha,
        "EXPECTED_FAILED_FINALIZER_COMMIT": failed_commit,
        "EXPECTED_FIRST_FAILED_JOURNAL_SHA256": first_failed_sha,
        "EXPECTED_EVALUATOR_SHA256": code_hashes["evaluator_sha256"],
        "EXPECTED_METRICS_SHA256": code_hashes["metrics_sha256"],
        "EXPECTED_IO_SHA256": code_hashes["io_sha256"],
        "EXPECTED_SUMMARY_SHA256": code_hashes["summary_sha256"],
    }
    for name, value in constants.items():
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(module, "_validate_inference_card", lambda *args: None)
    provenance = {
        "capture_timing": "process_start",
        "git_commit": new_commit,
        "git_dirty": False,
    }
    return SimpleNamespace(
        module=module,
        args=args,
        journal=journal,
        original=original,
        payload=payload,
        authorization=gate_authorization,
        provenance=provenance,
        card=card,
        roi_gt=roi_gt,
        quality_gt=quality_gt,
    )


def test_exact_failed_recovery_reuses_card_and_archives_prior_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _recovery_case(tmp_path, monkeypatch)
    _write_recovery_authorization(case)
    card, evidence = case.module._authorize_failed_recovery(
        case.args, case.journal, case.authorization, case.provenance
    )
    assert card == case.card
    assert len(case.calls) == 1
    assert evidence["action"] == case.module.RECOVERY_ACTION
    assert evidence["test_scoring_started"] is False
    assert evidence["rerun_inference"] is False
    archive = case.module._archive_failed_journal(
        case.journal, evidence["failed_journal_sha256"]
    )
    assert archive.read_bytes() == case.original_bytes

    case.module._publish_journal(
        case.journal,
        "started",
        recovery_command=["new-finalizer", "--recover"],
        recovery_provenance=case.provenance,
        recovery=evidence,
        test_scoring_started=False,
    )
    recovered = case.module._read_json(case.journal, "recovered journal")
    assert recovered["command"] == case.original["command"]
    assert recovered["provenance"] == case.original["provenance"]
    assert recovered["recovery_command"] == ["new-finalizer", "--recover"]
    assert recovered["recovery_provenance"] == case.provenance


@pytest.mark.parametrize(
    "field",
    [
        "failed_journal_sha256",
        "failed_ledger_sha256",
        "failed_queue_sha256",
        "failed_queue_log_sha256",
        "rerun_log_sha256",
        "inference_card_sha256",
        "gate_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "manifest_sha256",
        "split_sha256",
        "prediction_set_sha256",
    ],
)
def test_failed_recovery_rejects_wrong_hashes_without_mutating_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    case = _recovery_case(tmp_path, monkeypatch)
    case.payload[field] = "0" * 64
    _write_recovery_authorization(case)
    with pytest.raises(ValueError):
        case.module._authorize_failed_recovery(
            case.args, case.journal, case.authorization, case.provenance
        )
    assert case.journal.read_bytes() == case.original_bytes


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("attempts", 3),
        ("ordinary_retries", 0),
        ("oom_retries", 1),
        ("preemptions", 1),
        ("identity_sha256", "f" * 64),
    ],
)
def test_failed_recovery_requires_exact_failed_ledger_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    case = _recovery_case(tmp_path, monkeypatch)
    ledger = json.loads(case.args.recovery_ledger.read_text(encoding="utf-8"))
    ledger["jobs"]["stage2.unet.delivery"][field] = replacement
    case.args.recovery_ledger.write_text(
        json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8"
    )
    case.payload["failed_ledger_sha256"] = case.module.sha256_file(
        case.args.recovery_ledger
    )
    _write_recovery_authorization(case)
    with pytest.raises(ValueError, match="exact failed attempt"):
        case.module._authorize_failed_recovery(
            case.args, case.journal, case.authorization, case.provenance
        )


def test_failed_recovery_binds_exact_prediction_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _recovery_case(tmp_path, monkeypatch)
    case.payload["prediction_count"] -= 1
    _write_recovery_authorization(case)
    with pytest.raises(ValueError, match="prediction set differs"):
        case.module._authorize_failed_recovery(
            case.args, case.journal, case.authorization, case.provenance
        )


def test_failed_recovery_requires_absent_delivery_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _recovery_case(tmp_path, monkeypatch)
    case.args.bundle_out.mkdir()
    _write_recovery_authorization(case)
    with pytest.raises(ValueError, match="absent delivery bundle"):
        case.module._authorize_failed_recovery(
            case.args, case.journal, case.authorization, case.provenance
        )


@pytest.mark.parametrize("source", ["authorization", "journal"])
def test_failed_recovery_rejects_scoring_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    case = _recovery_case(tmp_path, monkeypatch)
    if source == "authorization":
        case.payload["test_scoring_started"] = True
    else:
        failed = json.loads(case.journal.read_text(encoding="utf-8"))
        failed["test_scoring_started"] = True
        case.journal.write_text(
            json.dumps(failed, sort_keys=True) + "\n", encoding="utf-8"
        )
        case.payload["failed_journal_sha256"] = case.module.sha256_file(case.journal)
    frozen = case.journal.read_bytes()
    _write_recovery_authorization(case)
    with pytest.raises(ValueError, match="scoring|untouched"):
        case.module._authorize_failed_recovery(
            case.args, case.journal, case.authorization, case.provenance
        )
    assert case.journal.read_bytes() == frozen


def test_failed_recovery_requires_explicit_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _recovery_case(tmp_path, monkeypatch)
    case.args.recovery_authorization = None
    with pytest.raises(RuntimeError, match="explicit recovery authorization"):
        case.module._authorize_failed_recovery(
            case.args, case.journal, case.authorization, case.provenance
        )
    assert case.journal.read_bytes() == case.original_bytes


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("reason_code", "different_bug"),
        ("authorization_basis", "unrecorded_approval"),
    ],
)
def test_failed_recovery_requires_exact_audit_basis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
) -> None:
    case = _recovery_case(tmp_path, monkeypatch)
    case.payload[field] = replacement
    _write_recovery_authorization(case)
    with pytest.raises(ValueError, match="identity or action changed"):
        case.module._authorize_failed_recovery(
            case.args, case.journal, case.authorization, case.provenance
        )


def test_main_refusal_does_not_overwrite_failed_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _recovery_case(tmp_path, monkeypatch)
    case.args.recovery_authorization = None
    monkeypatch.setattr(case.module, "parse_args", lambda: case.args)
    monkeypatch.setattr(
        case.module,
        "run_provenance",
        lambda command, capture_timing: case.provenance,
    )
    monkeypatch.setattr(
        case.module, "validate_gate", lambda args: case.authorization
    )
    with pytest.raises(RuntimeError, match="explicit recovery authorization"):
        case.module.main()
    assert case.journal.read_bytes() == case.original_bytes
    assert not case.module._lock_path(case.args.bundle_out).exists()


@pytest.mark.parametrize("prior_status", [None, "preempted_before_scoring"])
def test_main_rejects_recovery_arguments_without_failed_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prior_status: str | None,
) -> None:
    case = _recovery_case(tmp_path, monkeypatch)
    _write_recovery_authorization(case)
    if prior_status is None:
        case.journal.unlink()
        frozen = None
    else:
        prior = json.loads(case.journal.read_text(encoding="utf-8"))
        prior["status"] = prior_status
        case.journal.write_text(
            json.dumps(prior, sort_keys=True) + "\n", encoding="utf-8"
        )
        frozen = case.journal.read_bytes()
    monkeypatch.setattr(case.module, "parse_args", lambda: case.args)
    monkeypatch.setattr(
        case.module,
        "run_provenance",
        lambda command, capture_timing: case.provenance,
    )
    with pytest.raises(RuntimeError, match="recovery arguments require"):
        case.module.main()
    if frozen is None:
        assert not case.journal.exists()
    else:
        assert case.journal.read_bytes() == frozen
    assert not case.module._lock_path(case.args.bundle_out).exists()


def test_recovery_main_never_calls_inference_child_and_publishes_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _recovery_case(tmp_path, monkeypatch)
    _write_recovery_authorization(case)
    expected_authorization_sha = case.module.sha256_file(
        case.args.recovery_authorization
    )
    monkeypatch.setattr(case.module, "parse_args", lambda: case.args)
    monkeypatch.setattr(
        case.module,
        "run_provenance",
        lambda command, capture_timing: case.provenance,
    )
    monkeypatch.setattr(
        case.module, "validate_gate", lambda args: case.authorization
    )
    monkeypatch.setattr(
        case.module,
        "ensure_inference",
        lambda *args, **kwargs: pytest.fail("recovery reran the inference child"),
    )
    monkeypatch.setattr(
        case.module,
        "_score_command",
        lambda args, stage, card, authorization, *, table, gt_column: [
            "synthetic-score",
            table,
        ],
    )

    def fake_scoring(stage: Path, args) -> dict:
        summary = stage / "summary.csv"
        summary.write_text("synthetic-summary\n", encoding="utf-8")
        result = {"summary_sha256": case.module.sha256_file(summary)}
        for table in ("A", "A_prime"):
            per_image = stage / f"per_image_{table}.csv"
            per_image.write_text(f"synthetic-{table}\n", encoding="utf-8")
            result[table] = {
                "per_image_file": per_image.name,
                "per_image_sha256": case.module.sha256_file(per_image),
            }
        return result

    monkeypatch.setattr(case.module, "validate_scoring", fake_scoring)

    def fake_run(command) -> None:
        if any(str(item).endswith("make_tables.py") for item in command):
            out = Path(command[command.index("--out") + 1])
            (out / "tables.md").write_text("synthetic md\n", encoding="utf-8")
            (out / "tables.tex").write_text("synthetic tex\n", encoding="utf-8")

    monkeypatch.setattr(case.module, "_run", fake_run)

    def fake_qa(args, stage: Path, authorization) -> list[dict]:
        curve = stage / "training_curves.png"
        curve.write_bytes(b"synthetic-curve")
        overlay_dir = stage / "overlays"
        overlay_dir.mkdir()
        records = [{"curve_sha256": case.module.sha256_file(curve)}]
        for ordinal in range(1, 11):
            overlay = overlay_dir / f"overlay_{ordinal:02d}.png"
            overlay.write_bytes(f"overlay-{ordinal}".encode("ascii"))
            records.append(
                {
                    "ordinal": ordinal,
                    "image_id": f"synthetic_{ordinal}",
                    "examiner_value": "VALUE",
                    "sha256": case.module.sha256_file(overlay),
                }
            )
        return records

    monkeypatch.setattr(case.module, "_write_qa", fake_qa)
    case.module.main()

    delivery = case.module._read_json(
        case.args.bundle_out / "delivery.json", "synthetic delivery"
    )
    assert delivery["recovery"]["action"] == case.module.RECOVERY_ACTION
    assert (
        delivery["recovery"]["authorization_sha256"]
        == expected_authorization_sha
    )
    assert delivery["recovery"]["rerun_inference"] is False
    journal = case.module._read_json(case.journal, "completed journal")
    assert journal["status"] == "complete"
    assert journal["rerun_inference"] is False


@pytest.mark.parametrize(
    "field", ["old_inference_git_commit", "new_finalizer_git_commit"]
)
def test_failed_recovery_rejects_commit_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    case = _recovery_case(tmp_path, monkeypatch)
    case.payload[field] = "c" * 40
    _write_recovery_authorization(case)
    with pytest.raises(ValueError, match="commit differs"):
        case.module._authorize_failed_recovery(
            case.args, case.journal, case.authorization, case.provenance
        )
    assert case.journal.read_bytes() == case.original_bytes


def test_device_labels_accept_only_case_insensitive_cpu_and_cuda() -> None:
    module = _module()
    assert module._canonical_device_label("cuda") == "CUDA"
    assert module._canonical_device_label("CUDA") == "CUDA"
    assert module._canonical_device_label("cpu") == "CPU"
    for unsupported in ("GPU", " TPU", "TPU", " CUDA", "CUDA "):
        with pytest.raises(ValueError, match="unsupported device"):
            module._canonical_device_label(unsupported)


def test_exact_postscore_recovery_authorizes_second_score_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _postscore_recovery_case(tmp_path, monkeypatch)
    card, evidence = case.module._authorize_failed_scoring_recovery(
        case.args, case.journal, case.authorization, case.provenance
    )
    assert card == case.card
    assert evidence["kind"] == case.module.POST_SCORE_RECOVERY_KIND
    assert evidence["score_attempts"] == 2
    assert evidence["prior_scoring_started"] is True
    assert evidence["rerun_inference"] is False
    assert evidence["rerun_scoring"] is True


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("prior_evaluator_invocations", 1),
        ("rerun_inference", True),
        ("score_attempt_ordinal", 3),
        ("evaluator_sha256", "0" * 64),
        ("metrics_sha256", "0" * 64),
        ("io_sha256", "0" * 64),
        ("summary_sha256", "0" * 64),
        ("ordered_source_gt_sha256", "0" * 64),
        ("failed_journal_sha256", "0" * 64),
    ],
)
def test_postscore_recovery_rejects_tampered_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    case = _postscore_recovery_case(tmp_path, monkeypatch)
    case.payload[field] = replacement
    case.args.recovery_authorization.write_text(
        json.dumps(case.payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    frozen = case.journal.read_bytes()
    with pytest.raises(ValueError):
        case.module._authorize_failed_scoring_recovery(
            case.args, case.journal, case.authorization, case.provenance
        )
    assert case.journal.read_bytes() == frozen


def test_postscore_recovery_rejects_surviving_score_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _postscore_recovery_case(tmp_path, monkeypatch)
    staging = case.args.bundle_out.parent / f".{case.args.bundle_out.name}-partial"
    staging.mkdir()
    with pytest.raises(ValueError, match="surviving score artifacts"):
        case.module._authorize_failed_scoring_recovery(
            case.args, case.journal, case.authorization, case.provenance
        )


@pytest.mark.parametrize(
    "artifact",
    [
        "journal",
        "ledger",
        "queue_log",
        "rerun_log",
        "first_authorization",
        "inference_card",
        "manifest_audit",
    ],
)
def test_postscore_recovery_rejects_changed_bound_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    case = _postscore_recovery_case(tmp_path, monkeypatch)
    paths = {
        "journal": case.journal,
        "ledger": case.args.recovery_ledger,
        "queue_log": case.args.recovery_log,
        "rerun_log": case.args.rerun_log,
        "first_authorization": case.args.prior_recovery_authorization,
        "inference_card": case.args.inference_card,
        "manifest_audit": case.args.manifest.with_suffix(".audit.json"),
    }
    with paths[artifact].open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        case.module._authorize_failed_scoring_recovery(
            case.args, case.journal, case.authorization, case.provenance
        )


def test_postscore_recovery_rehashes_current_gt_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _postscore_recovery_case(tmp_path, monkeypatch)
    case.roi_gt.write_bytes(b"mutated-roi-gt")
    with pytest.raises(ValueError, match="ROI GT changed after manifest freeze"):
        case.module._authorize_failed_scoring_recovery(
            case.args, case.journal, case.authorization, case.provenance
        )


def _install_postscore_main_fakes(case, monkeypatch: pytest.MonkeyPatch):
    module = case.module
    calls = []
    monkeypatch.setattr(module, "parse_args", lambda: case.args)
    monkeypatch.setattr(
        module,
        "run_provenance",
        lambda command, capture_timing: case.provenance,
    )
    monkeypatch.setattr(module, "validate_gate", lambda args: case.authorization)
    monkeypatch.setattr(
        module,
        "ensure_inference",
        lambda *args, **kwargs: pytest.fail("post-score recovery reran inference"),
    )
    monkeypatch.setattr(
        module,
        "_score_command",
        lambda args, stage, card, authorization, *, table, gt_column: [
            "synthetic-score",
            table,
        ],
    )

    def fake_run(command) -> None:
        calls.append(list(command))
        if any(str(item).endswith("make_tables.py") for item in command):
            out = Path(command[command.index("--out") + 1])
            (out / "tables.md").write_text("synthetic md\n", encoding="utf-8")
            (out / "tables.tex").write_text("synthetic tex\n", encoding="utf-8")

    monkeypatch.setattr(module, "_run", fake_run)

    def fake_scoring(stage: Path, args) -> dict:
        summary = stage / "summary.csv"
        summary.write_text("synthetic-summary\n", encoding="utf-8")
        result = {"summary_sha256": module.sha256_file(summary)}
        for table in ("A", "A_prime"):
            per_image = stage / f"per_image_{table}.csv"
            per_image.write_text(f"synthetic-{table}\n", encoding="utf-8")
            result[table] = {
                "per_image_file": per_image.name,
                "per_image_sha256": module.sha256_file(per_image),
            }
        return result

    monkeypatch.setattr(module, "validate_scoring", fake_scoring)

    def fake_qa(args, stage: Path, authorization) -> list[dict]:
        curve = stage / "training_curves.png"
        curve.write_bytes(b"synthetic-curve")
        overlay_dir = stage / "overlays"
        overlay_dir.mkdir()
        records = [{"curve_sha256": module.sha256_file(curve)}]
        for ordinal in range(1, 11):
            overlay = overlay_dir / f"overlay_{ordinal:02d}.png"
            overlay.write_bytes(f"overlay-{ordinal}".encode("ascii"))
            records.append(
                {
                    "ordinal": ordinal,
                    "image_id": f"synthetic_{ordinal}",
                    "examiner_value": "VALUE",
                    "sha256": module.sha256_file(overlay),
                }
            )
        return records

    monkeypatch.setattr(module, "_write_qa", fake_qa)
    return calls


def test_postscore_main_runs_exactly_two_scorers_and_records_two_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _postscore_recovery_case(tmp_path, monkeypatch)
    calls = _install_postscore_main_fakes(case, monkeypatch)
    case.module.main()
    score_calls = [command for command in calls if command[0] == "synthetic-score"]
    assert score_calls == [
        ["synthetic-score", "A"],
        ["synthetic-score", "A_prime"],
    ]
    delivery = case.module._read_json(
        case.args.bundle_out / "delivery.json", "synthetic delivery"
    )
    assert delivery["score_attempts"] == 2
    assert delivery["postscore_recovery"]["score_attempts"] == 2
    assert delivery["prior_recovery"] == case.original["recovery"]
    assert delivery["recovery"]["rerun_inference"] is False
    journal = case.module._read_json(case.journal, "completed journal")
    assert journal["status"] == "complete"
    assert journal["score_attempts"] == 2
    assert journal["test_scoring_started"] is True
    archive = case.journal.with_name(
        f"{case.journal.name}.failed-{case.payload['failed_journal_sha256']}.archive"
    )
    assert archive.is_file()

    # A normal invocation after atomic publication only revalidates the bundle;
    # it cannot run either scorer or inference again.
    case.args.recovery_authorization = None
    case.args.prior_recovery_authorization = None
    case.args.recovery_ledger = None
    case.args.recovery_log = None
    case.args.rerun_log = None
    calls.clear()
    case.module.main()
    assert calls == []


def test_postscore_failure_retains_staging_and_blocks_third_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _postscore_recovery_case(tmp_path, monkeypatch)
    calls = _install_postscore_main_fakes(case, monkeypatch)

    def fail_score(command) -> None:
        calls.append(list(command))
        raise RuntimeError("synthetic second-score failure")

    monkeypatch.setattr(case.module, "_run", fail_score)
    with pytest.raises(RuntimeError, match="synthetic second-score failure"):
        case.module.main()
    journal = case.module._read_json(case.journal, "failed second-score journal")
    retained = Path(journal["retained_staging_path"])
    assert retained.is_dir()
    assert journal["score_attempts"] == 2
    assert journal["test_scoring_started"] is True
    assert not case.args.bundle_out.exists()
    call_count = len(calls)
    with pytest.raises(ValueError, match="surviving score artifacts"):
        case.module.main()
    assert len(calls) == call_count
