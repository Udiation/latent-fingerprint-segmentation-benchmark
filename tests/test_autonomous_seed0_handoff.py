"""CPU-only tests for the fail-closed full seed-0 handoff controller."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Dict

import pytest
import yaml

from segbench.artifact_identity import canonical_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _module():
    path = ROOT / "scripts/autonomous_seed0_handoff.py"
    spec = importlib.util.spec_from_file_location("test_autonomous_seed0_handoff_module", path)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def _json(path: Path, value: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _yaml(path: Path, value: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _code(root: Path, marker: str) -> Dict[str, Any]:
    return {
        "root": str(root),
        "git_commit": marker * 40,
        "git_tree": ("a" if marker != "a" else "b") * 40,
        "files_sha256": {"synthetic.py": marker * 64},
    }


def _resource_evidence(tmp_path: Path) -> Dict[str, str]:
    path = tmp_path / "private-resource-evidence.md"
    path.write_text("measured fixture\n", encoding="utf-8")
    return {"path": str(path), "sha256": sha256_file(path)}


def _production_runtime(tmp_path: Path) -> Dict[str, Any]:
    return {
        "allowed_gpus": [1, 3],
        "max_jobs_per_gpu": 3,
        "memory_fraction": 0.3,
        "max_cpu_cores": 64,
        "max_project_ram_gib": 512.0,
        "min_disk_free_fraction": 0.15,
        "min_free_mib": 40960,
        "foreign_process_mib": 2048,
        "min_available_ram_gib": 32.0,
        "host_ram_gib_per_job": 80.0,
        "cpu_workers_per_job": 8,
        "reservation_evidence": _resource_evidence(tmp_path),
    }


def _source_fixture(tmp_path: Path) -> Dict[str, Any]:
    restricted = tmp_path / "restricted"
    handoff = restricted / "handoff"
    source_code = tmp_path / "source-code"
    source_plan = {
        "schema_version": 1,
        "kind": "autonomous_validation_handoff_plan",
        "code": _code(source_code, "c"),
        "restricted_root": str(restricted),
        "handoff_root": str(handoff),
        "manifest": str(restricted / "manifest.csv"),
        "splits": str(restricted / "splits.json"),
        "cache_dir": str(restricted / "cache"),
        "training_threshold_amendment": str(restricted / "amendment.json"),
        "runs_root": str(restricted / "runs"),
        "python": str(tmp_path / "venv/python"),
        "selected_full_resources": {
            "host_ram_gib_per_job": 80.0,
            "cpu_workers_per_job": 8,
            "reservation_evidence": _resource_evidence(tmp_path),
        },
        "test_or_heldout_opened": False,
        "gpu_launched": False,
    }
    source_plan_path = _json(tmp_path / "source-plan.json", source_plan)
    source_proof = {
        "schema_version": 1,
        "kind": "autonomous_validation_handoff_preparation",
        "status": "READY_NOT_LAUNCHED",
        "plan": str(source_plan_path),
        "plan_sha256": sha256_file(source_plan_path),
        "code_commit": source_plan["code"]["git_commit"],
        "source_controller_authenticated": True,
        "source_controller_phase": "mixed_recovery_controller",
        "source_controller_argv_sha256": "d" * 64,
        "test_or_heldout_opened": False,
        "gpu_launched": False,
    }
    source_proof_path = _json(tmp_path / "source-proof.json", source_proof)
    return {
        "restricted": restricted,
        "handoff": handoff,
        "source_plan": source_plan,
        "source_plan_path": source_plan_path,
        "source_proof_path": source_proof_path,
    }


def _controller_plan(tmp_path: Path, fixture: Dict[str, Any]) -> Dict[str, Any]:
    module = _module()
    root = fixture["restricted"] / "seed0-controller"
    return {
        "schema_version": 1,
        "kind": module.PLAN_KIND,
        "code": _code(tmp_path / "controller-code", "e"),
        "source_handoff": {
            "plan": str(fixture["source_plan_path"]),
            "plan_sha256": sha256_file(fixture["source_plan_path"]),
            "preparation_proof": str(fixture["source_proof_path"]),
            "preparation_proof_sha256": sha256_file(fixture["source_proof_path"]),
            "completion": str(fixture["handoff"] / "completion.json"),
            "code": fixture["source_plan"]["code"],
        },
        "restricted_root": str(fixture["restricted"]),
        "controller_root": str(root),
        "state": str(root / "state.json"),
        "log": str(root / "queue.log"),
        "launch_authorization": str(root / "authorization.json"),
        "completion": str(root / "completion.json"),
        "source_wait_seconds": 60.0,
        "poll_seconds": 0.01,
        "runtime_policy": _production_runtime(tmp_path),
        "test_or_heldout_opened": False,
        "gpu_launched": False,
    }


def _selected_fixture(tmp_path: Path, fixture: Dict[str, Any]) -> Dict[str, Any]:
    source = fixture["source_plan"]
    handoff = fixture["handoff"]
    config = _yaml(handoff / "selected_full_configs/unetpp.yaml", {"method": "unetpp"})
    entrypoint = Path(source["code"]["root"]) / "segbench/scripts/train_seg.py"
    out = Path(source["runs_root"]) / "full_v3_expanded_grid/unetpp/seed0"
    item = {
        "name": "full.unetpp.seed0",
        "argv": [
            source["python"], str(entrypoint), "--config", str(config),
            "--manifest", source["manifest"], "--splits", source["splits"],
            "--out", str(out), "--seed", "0", "--resume", "auto",
            "--cache-dir", source["cache_dir"], "--threshold-amendment",
            source["training_threshold_amendment"],
        ],
        "env": {},
        "completion_artifact": str(out / "run.json"),
        "resources": {"host_ram_gib": 80.0, "cpu_workers": 8},
    }
    item["job_identity_sha256"] = canonical_sha256(item)
    queue = {
        "schema_version": 1,
        "phase": "selected_full_seed0_training",
        "launch_authorized": True,
        "selection_split": "val",
        "test_inference_authorized": False,
        "resource_reservation": {
            "host_ram_gib_per_job": 80.0,
            "cpu_workers_per_job": 8,
            "reservation_evidence": _resource_evidence(tmp_path),
        },
        "jobs": [item],
    }
    queue_path = _yaml(handoff / "selected_full_seed0_queue.yaml", queue)
    initial = _json(handoff / "initial_selections/summary.json", {"selection_source": "validation_only"})
    late = _json(handoff / "late_selections/summary.json", {"selection_source": "validation_only"})
    proof = {
        "schema_version": 1,
        "kind": "selected_full_seed0_queue_preparation",
        "status": "READY_NOT_LAUNCHED",
        "test_or_heldout_opened": False,
        "selection_summaries": [
            {"path": str(initial), "sha256": sha256_file(initial)},
            {"path": str(late), "sha256": sha256_file(late)},
        ],
        "queue": str(queue_path),
        "queue_sha256": sha256_file(queue_path),
        "jobs": [item["name"]],
        "resource_reservation": queue["resource_reservation"],
        "decisions": [{
            "method": "unetpp",
            "action": "fresh_full_seed0",
            "selected_learning_rate": 0.0003,
            "selected_threshold": 0.2,
            "selection_sha256": "f" * 64,
            "config": str(config),
            "config_sha256": sha256_file(config),
        }],
        "gpu_launched": False,
    }
    proof_path = _json(handoff / "selected_full_seed0_preparation.json", proof)
    completion = {
        "schema_version": 1,
        "kind": "autonomous_validation_handoff_completion",
        "status": "COMPLETE_NO_TEST_OPENED",
        "plan_sha256": sha256_file(fixture["source_plan_path"]),
        "test_or_heldout_opened": False,
        "source_ledger_written": False,
        "source_signalled": False,
        "initial_selection_summary": str(initial),
        "late_selection_summary": str(late),
        "selected_full_seed0_queue": str(queue_path),
        "selected_full_seed0_queue_sha256": sha256_file(queue_path),
        "selected_full_seed0_launched": False,
    }
    completion_path = _json(handoff / "completion.json", completion)
    return {
        "queue": queue,
        "queue_path": queue_path,
        "proof": proof,
        "proof_path": proof_path,
        "completion": completion,
        "completion_path": completion_path,
        "item": item,
    }


def test_prepare_is_no_launch_and_accepts_absent_future_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    fixture = _source_fixture(tmp_path)
    spec = _json(tmp_path / "spec.json", {
        "schema_version": 1,
        "kind": module.SPEC_KIND,
        "source_handoff_plan": str(fixture["source_plan_path"]),
        "source_handoff_plan_sha256": sha256_file(fixture["source_plan_path"]),
        "source_handoff_preparation_proof": str(fixture["source_proof_path"]),
        "source_handoff_preparation_proof_sha256": sha256_file(fixture["source_proof_path"]),
        "controller_root": str(fixture["restricted"] / "seed0-controller"),
        "source_wait_seconds": 60,
        "poll_seconds": 1,
        "runtime_policy": _production_runtime(tmp_path),
    })
    monkeypatch.setattr(module, "_code_identity", lambda root: _code(root, "e"))
    monkeypatch.setattr(module.upstream, "_verify_code", lambda value: Path(value["root"]))
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_a, **_k: pytest.fail("prepare must not launch a subprocess"),
    )
    out, proof = tmp_path / "plan.json", tmp_path / "preparation.json"
    result = module.prepare(spec, tmp_path / "controller-code", out, proof)
    assert result["status"] == "READY_NOT_LAUNCHED"
    assert result["gpu_launched"] is False
    assert result["test_or_heldout_opened"] is False
    assert not (fixture["handoff"] / "completion.json").exists()


def test_selected_queue_authenticates_without_opening_data(tmp_path: Path) -> None:
    module = _module()
    fixture = _source_fixture(tmp_path)
    selected = _selected_fixture(tmp_path, fixture)
    plan = _controller_plan(tmp_path, fixture)
    authorization = module._authenticate_selected(
        plan,
        fixture["source_plan"],
        selected["completion"],
        sha256_file(selected["completion_path"]),
    )
    assert authorization["status"] == "AUTHORIZED_NOT_LAUNCHED"
    assert authorization["test_or_heldout_opened"] is False
    assert authorization["jobs"][0]["output"].endswith(
        "/full_v3_expanded_grid/unetpp/seed0"
    )


def test_upstream_and_seed0_resource_evidence_must_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    fixture = _source_fixture(tmp_path)
    plan = _controller_plan(tmp_path, fixture)
    other = tmp_path / "other-private-evidence.md"
    other.write_text("different measurement\n", encoding="utf-8")
    plan["runtime_policy"]["reservation_evidence"] = {
        "path": str(other), "sha256": sha256_file(other),
    }
    monkeypatch.setattr(module.upstream, "_verify_code", lambda value: Path(value["root"]))
    with pytest.raises(ValueError, match="resource binding changed"):
        module._verify_upstream(plan)


def test_runtime_resource_evidence_mutation_fails_closed(tmp_path: Path) -> None:
    module = _module()
    policy = _production_runtime(tmp_path)
    assert module._validated_runtime_policy(policy) == policy
    Path(policy["reservation_evidence"]["path"]).write_text(
        "mutated\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="evidence identity changed"):
        module._validated_runtime_policy(policy)


@pytest.mark.parametrize("target", ["queue", "proof"])
def test_selected_artifact_mutation_fails_closed(tmp_path: Path, target: str) -> None:
    module = _module()
    fixture = _source_fixture(tmp_path)
    selected = _selected_fixture(tmp_path, fixture)
    plan = _controller_plan(tmp_path, fixture)
    path = selected[f"{target}_path"]
    if target == "queue":
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        changed = json.loads(path.read_text(encoding="utf-8"))
        changed["gpu_launched"] = True
        _json(path, changed)
    with pytest.raises(ValueError, match="changed"):
        module._authenticate_selected(
            plan,
            fixture["source_plan"],
            selected["completion"],
            sha256_file(selected["completion_path"]),
        )


def test_legacy_output_path_is_rejected(tmp_path: Path) -> None:
    module = _module()
    fixture = _source_fixture(tmp_path)
    selected = _selected_fixture(tmp_path, fixture)
    argv = selected["item"]["argv"]
    output_index = argv.index("--out") + 1
    argv[output_index] = str(Path(fixture["source_plan"]["runs_root"]) / "full/unetpp/seed0")
    selected["item"]["completion_artifact"] = argv[output_index] + "/run.json"
    selected["item"]["job_identity_sha256"] = canonical_sha256({
        key: value for key, value in selected["item"].items()
        if key != "job_identity_sha256"
    })
    with pytest.raises(ValueError, match="fresh versioned"):
        module._validate_job(
            selected["item"], fixture["source_plan"],
            selected["proof"]["decisions"][0], _production_runtime(tmp_path)
        )


def test_heldout_option_is_rejected(tmp_path: Path) -> None:
    module = _module()
    fixture = _source_fixture(tmp_path)
    selected = _selected_fixture(tmp_path, fixture)
    selected["item"]["argv"].extend(["--test-split", "test"])
    with pytest.raises(ValueError, match="held-out token"):
        module._validate_job(
            selected["item"], fixture["source_plan"],
            selected["proof"]["decisions"][0], _production_runtime(tmp_path)
        )


def test_upstream_plan_or_code_mutation_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    fixture = _source_fixture(tmp_path)
    plan = _controller_plan(tmp_path, fixture)
    monkeypatch.setattr(module.upstream, "_verify_code", lambda value: Path(value["root"]))
    fixture["source_plan_path"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity changed"):
        module._verify_upstream(plan)
    monkeypatch.setattr(module, "_code_identity", lambda _root: _code(tmp_path, "f"))
    with pytest.raises(ValueError, match="code identity changed"):
        module._verify_code(_code(tmp_path, "e"))


def test_wait_can_start_before_completion_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    fixture = _source_fixture(tmp_path)
    plan = _controller_plan(tmp_path, fixture)
    completion_path = fixture["handoff"] / "completion.json"
    calls = []
    monkeypatch.setattr(
        module,
        "_verify_upstream",
        lambda _plan: (fixture["source_plan"], {}),
    )

    def publish(delay: float) -> None:
        calls.append(delay)
        _json(completion_path, {"kind": "future-completion"})

    source, completion, digest = module._wait_for_completion(plan, sleep=publish)
    assert source == fixture["source_plan"]
    assert completion["kind"] == "future-completion"
    assert digest == sha256_file(completion_path)
    assert calls == [plan["poll_seconds"]]


def test_guarded_command_has_exact_resource_limits(tmp_path: Path) -> None:
    module = _module()
    fixture = _source_fixture(tmp_path)
    plan = _controller_plan(tmp_path, fixture)
    authorization = {"selected_queue": str(fixture["handoff"] / "queue.yaml")}
    argv = module._guarded_argv(plan, authorization)
    assert argv[argv.index("--gpus") + 1] == "1,3"
    assert argv[argv.index("--max-per-gpu") + 1] == "3"
    assert argv[argv.index("--memory-fraction") + 1] == "0.3"
    assert argv[argv.index("--max-cpu-cores") + 1] == "64"
    assert argv[argv.index("--max-project-ram-gib") + 1] == "512"
    assert argv[argv.index("--min-disk-free-fraction") + 1] == "0.15"


def test_private_policy_reserves_three_jobs_per_gpu_within_host_caps(
    tmp_path: Path,
) -> None:
    module = _module()
    policy = module._validated_runtime_policy(_production_runtime(tmp_path))
    slots = len(policy["allowed_gpus"]) * policy["max_jobs_per_gpu"]
    assert slots == 6
    assert slots * policy["host_ram_gib_per_job"] == 480.0
    assert slots * policy["host_ram_gib_per_job"] <= policy["max_project_ram_gib"]
    assert slots * policy["cpu_workers_per_job"] == 48
    assert slots * policy["cpu_workers_per_job"] <= policy["max_cpu_cores"]

    over_reserved = dict(policy, host_ram_gib_per_job=96.0)
    with pytest.raises(ValueError, match="cannot fill every registered GPU slot"):
        module._validated_runtime_policy(over_reserved)


def test_controller_implementation_contains_no_private_hardware_defaults() -> None:
    source = (ROOT / "scripts/autonomous_seed0_handoff.py").read_text(encoding="utf-8")
    for private_literal in (
        '"allowed_gpus": [1, 3]',
        '"max_project_ram_gib": 512',
        '"host_ram_gib_per_job": 80',
        '"memory_fraction": 0.3',
    ):
        assert private_literal not in source
