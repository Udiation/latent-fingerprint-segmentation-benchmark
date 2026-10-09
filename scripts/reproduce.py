#!/usr/bin/env python3
"""Run one frozen benchmark table from a declarative private input plan.

Only audited repository entry points can be dispatched. This command never
downloads data or selects learning rates, checkpoints, or mask thresholds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "segbench.reproduction-plan.v1"
TABLES = {"A", "A_prime", "B1", "D"}
SAFE_KEY = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
HEX256 = re.compile(r"^[0-9a-f]{64}$")


class PlanError(ValueError):
    """The reproduction plan is incomplete or scientifically unsafe."""


def parse_env(path: Path) -> dict[str, str]:
    """Read a small KEY=VALUE file without shell evaluation."""
    if not path.is_file():
        raise PlanError(f"missing local config: {path}")
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise PlanError(f"invalid local config line {number}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise PlanError(f"invalid local config key on line {number}")
        values[key] = value.strip()
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PlanError(f"{context} must be a JSON object")
    return value


def _required_string(values: Mapping[str, Any], key: str, context: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{context} requires non-empty {key!r}")
    return value


def _path(value: object, base: Path, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{context} must be a non-empty path string")
    candidate = Path(value).expanduser()
    return (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def _file(value: object, base: Path, context: str) -> Path:
    path = _path(value, base, context)
    if not path.is_file():
        raise PlanError(f"{context} is not a regular file: {path}")
    return path


def _directory(value: object, base: Path, context: str) -> Path:
    path = _path(value, base, context)
    if not path.is_dir():
        raise PlanError(f"{context} is not a directory: {path}")
    return path


def _outside_public_tree(path: Path, context: str) -> None:
    try:
        path.resolve().relative_to(ROOT)
    except ValueError:
        return
    raise PlanError(f"{context} must be outside the public repository")


def _checked_artifact(
    values: Mapping[str, Any], path_key: str, hash_key: str, base: Path, context: str
) -> Path:
    path = _file(values.get(path_key), base, f"{context}.{path_key}")
    expected = values.get(hash_key)
    if not isinstance(expected, str) or not HEX256.fullmatch(expected.lower()):
        raise PlanError(f"{context}.{hash_key} must be a SHA-256")
    actual = _sha256(path)
    if actual != expected.lower():
        raise PlanError(
            f"{context}.{path_key} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return path


def _option(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _prediction_template(value: object, base: Path, context: str) -> str:
    if not isinstance(value, str) or "{image_id}" not in value:
        raise PlanError(f"{context} must contain {{image_id}}")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return str(candidate.resolve())


def _load_plan(path: Path, expected_table: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PlanError(f"invalid JSON plan: {error}") from error
    plan = _mapping(payload, "plan")
    if plan.get("schema") != SCHEMA:
        raise PlanError(f"plan schema must be {SCHEMA!r}")
    if plan.get("table") != expected_table:
        raise PlanError(
            f"plan table {plan.get('table')!r} does not match entry point {expected_table!r}"
        )
    if not isinstance(plan.get("entries"), list) or not plan["entries"]:
        raise PlanError("plan.entries must be a non-empty list")
    return plan


def _default_evaluation(table: str) -> Mapping[str, Any]:
    if table == "A":
        return {
            "kind": "mask",
            "gt_column": "mask_path",
            "group_by": "examiner_value",
            "exclude": ["examiner_value=NONPRINT"],
        }
    if table == "A_prime":
        return {
            "kind": "mask",
            "gt_column": "quality_mask_path",
            "group_by": "examiner_value",
            "exclude": ["examiner_value=NONPRINT"],
        }
    if table == "B1":
        return {"kind": "hull"}
    return {}


def _resolve_context(
    plan: Mapping[str, Any], entry: Mapping[str, Any], base: Path, table: str
) -> dict[str, Any]:
    source_manifest = _file(
        entry.get("source_manifest", plan.get("source_manifest")),
        base,
        "source_manifest",
    )
    evaluation_manifest = _file(
        entry.get(
            "evaluation_manifest", plan.get("evaluation_manifest", str(source_manifest))
        ),
        base,
        "evaluation_manifest",
    )
    split_value = entry.get("split", plan.get("split", "test"))
    if split_value != "test":
        raise PlanError("paper-table reproduction is restricted to split='test'")
    subset = entry.get("subset", plan.get("subset"))
    if not isinstance(subset, str) or not subset:
        raise PlanError("subset must be declared")
    target_ppi = float(entry.get("target_ppi", plan.get("target_ppi", 500.0)))
    if target_ppi != 500.0:
        raise PlanError("paper-table reproduction requires target_ppi=500")
    splits_value = entry.get("splits", plan.get("splits"))
    splits = None
    if splits_value is not None:
        splits = _file(splits_value, base, "splits")
        try:
            split_map = json.loads(splits.read_text(encoding="utf-8"))["splits"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise PlanError("splits must contain a JSON object named 'splits'") from error
        if not isinstance(split_map, dict) or not split_map:
            raise PlanError("splits.splits must be a non-empty object")
        if set(split_map.values()) - {"train", "val", "test"}:
            raise PlanError("splits contains an unsupported partition label")
    if table in {"A", "A_prime"} and splits is None:
        raise PlanError(f"Table {table} requires a subject split file")
    return {
        "source_manifest": source_manifest,
        "evaluation_manifest": evaluation_manifest,
        "splits": splits,
        "subset": subset,
        "split": split_value,
        "target_ppi": target_ppi,
    }


def _runner_command(
    runner: Mapping[str, Any], context: Mapping[str, Any], run_dir: Path, base: Path
) -> tuple[list[str] | None, str, float]:
    """Build one audited inference command and its prediction template."""
    kind = _required_string(runner, "kind", "runner")
    predictions = run_dir / "predictions"
    probabilities = run_dir / "probabilities"
    cards = run_dir / "cards"
    template = str(predictions / "{image_id}.png")
    common = [
        "--manifest", str(context["source_manifest"]),
        "--splits", str(context["splits"]),
        "--subset", str(context["subset"]),
        "--split", str(context["split"]),
    ]

    if kind == "existing_predictions":
        template = _prediction_template(
            runner.get("pred_template"), base, "runner.pred_template"
        )
        pred_ppi = runner.get("pred_ppi")
        if pred_ppi is None or not math.isfinite(float(pred_ppi)) or float(pred_ppi) <= 0:
            raise PlanError("existing predictions require an explicit positive pred_ppi")
        return None, template, float(pred_ppi)

    if context["splits"] is None:
        raise PlanError(f"runner kind {kind!r} requires splits")

    if kind == "deep_checkpoint":
        checkpoint = _checked_artifact(
            runner, "checkpoint", "checkpoint_sha256", base, "runner"
        )
        if "seed" not in runner:
            raise PlanError("deep_checkpoint runner requires seed")
        command = [
            sys.executable, str(ROOT / "scripts" / "infer_masks.py"),
            "--config", str(_file(runner.get("config"), base, "runner.config")),
            "--checkpoint", str(checkpoint), *common,
            "--out", str(predictions),
            "--probability-out", str(probabilities),
            "--threshold-receipt", str(_file(
                runner.get("threshold_receipt"), base, "runner.threshold_receipt"
            )),
            "--threshold-amendment", str(_file(
                runner.get("threshold_amendment"), base, "runner.threshold_amendment"
            )),
            "--timing-out", str(cards / "timing.json"),
            "--device", _required_string(runner, "device", "runner"),
            "--seed", str(int(runner["seed"])),
            "--skip-valid",
        ]
        if runner.get("cache_dir") is not None:
            command.extend(["--cache-dir", str(_directory(
                runner["cache_dir"], base, "runner.cache_dir"
            ))])
        if runner.get("cache_compatibility_receipt") is not None:
            command.extend(["--cache-compatibility-receipt", str(_file(
                runner["cache_compatibility_receipt"], base,
                "runner.cache_compatibility_receipt",
            ))])
        if runner.get("cross_domain_authorization") is not None:
            command.extend(["--cross-domain-authorization", str(_file(
                runner["cross_domain_authorization"], base,
                "runner.cross_domain_authorization",
            ))])
        _option(command, "--memory-fraction", runner.get("memory_fraction"))
        return command, template, 500.0

    if kind == "sufs_checkpoint":
        checkpoint = _checked_artifact(
            runner, "checkpoint", "checkpoint_sha256", base, "runner"
        )
        command = [
            sys.executable, str(ROOT / "scripts" / "infer_sufs.py"),
            "--config", str(_file(runner.get("config"), base, "runner.config")),
            "--checkpoint", str(checkpoint), *common,
            "--out", str(predictions),
            "--probability-out", str(probabilities),
            "--threshold-receipt", str(_file(
                runner.get("threshold_receipt"), base, "runner.threshold_receipt"
            )),
            "--threshold-amendment", str(_file(
                runner.get("threshold_amendment"), base, "runner.threshold_amendment"
            )),
            "--run-card", str(cards / "inference.json"),
            "--device", _required_string(runner, "device", "runner"),
            "--skip-valid",
        ]
        if runner.get("cache_dir") is not None:
            command.extend(["--cache-dir", str(_directory(
                runner["cache_dir"], base, "runner.cache_dir"
            ))])
        _option(command, "--memory-fraction", runner.get("memory_fraction"))
        return command, template, 500.0

    if kind == "classical_model":
        if "seed" not in runner:
            raise PlanError("classical_model runner requires seed")
        method = _required_string(runner, "method", "runner")
        model = _checked_artifact(runner, "model", "model_sha256", base, "runner")
        command = [
            sys.executable, str(ROOT / "scripts" / "predict_classical.py"),
            *common,
            "--model", str(model),
            "--training-card", str(_file(
                runner.get("training_card"), base, "runner.training_card"
            )),
            "--method", method,
            "--display-name", _required_string(runner, "display_name", "runner"),
            "--target-ppi", "500",
            "--seed", str(int(runner["seed"])),
            "--output-root", str(predictions),
            "--run-card", str(cards / "inference.json"),
        ]
        return command, template, 500.0

    if kind == "pyfing":
        command = [
            sys.executable, str(ROOT / "scripts" / "run_pyfing.py"), "generate",
            "--method", _required_string(runner, "method", "runner"),
            "--display-name", _required_string(runner, "display_name", "runner"),
            *common,
            "--target-ppi", "500",
            "--output-root", str(predictions),
            "--probability-root", str(probabilities),
            "--run-card", str(cards / "inference.json"),
        ]
        if runner.get("parameters_json") is not None:
            command.extend(["--parameters-json", str(_checked_artifact(
                runner, "parameters_json", "parameters_sha256", base, "runner"
            ))])
        if runner.get("cache_dir") is not None:
            command.extend(["--cache-dir", str(_directory(
                runner["cache_dir"], base, "runner.cache_dir"
            ))])
        _option(command, "--memory-fraction", runner.get("memory_fraction"))
        return command, template, 500.0

    if kind == "fingernet":
        weights = _checked_artifact(runner, "weights", "weights_sha256", base, "runner")
        command = [
            sys.executable, str(ROOT / "scripts" / "run_fingernet.py"),
            "--config", str(_file(runner.get("config"), base, "runner.config")),
            *common,
            "--source-repo", str(_directory(
                runner.get("source_repo"), base, "runner.source_repo"
            )),
            "--weights", str(weights),
            "--output-root", str(predictions),
            "--probability-root", str(probabilities),
            "--run-card", str(cards / "inference.json"),
            "--device", _required_string(runner, "device", "runner"),
        ]
        if runner.get("cache_dir") is not None:
            command.extend(["--cache-dir", str(_directory(
                runner["cache_dir"], base, "runner.cache_dir"
            ))])
        return command, template, 500.0

    if kind == "reference":
        reference = runner.get("reference")
        if reference not in {"all-background", "all-foreground"}:
            raise PlanError("reference runner requires all-background or all-foreground")
        return None, str(reference), 500.0

    raise PlanError(f"unsupported runner kind: {kind!r}")


def _evaluation(
    table: str, plan: Mapping[str, Any], entry: Mapping[str, Any]
) -> Mapping[str, Any]:
    defaults = dict(_default_evaluation(table))
    defaults.update(_mapping(plan.get("evaluation", {}), "plan.evaluation"))
    defaults.update(_mapping(entry.get("evaluation", {}), "entry.evaluation"))
    kind = defaults.get("kind")
    if kind not in {"mask", "hull"}:
        raise PlanError("evaluation.kind must be explicit ('mask' or 'hull')")
    if table in {"A", "A_prime"} and kind != "mask":
        raise PlanError(f"Table {table} requires mask evaluation")
    if table in {"A", "A_prime"}:
        expected_gt = "mask_path" if table == "A" else "quality_mask_path"
        expected_exclusion = ["examiner_value=NONPRINT"]
        if (
            defaults.get("gt_column") != expected_gt
            or defaults.get("group_by") != "examiner_value"
            or defaults.get("exclude") != expected_exclusion
        ):
            raise PlanError(
                f"Table {table} ground truth, grouping, and exclusions are frozen"
            )
    if table == "B1" and kind != "hull":
        raise PlanError("Table B1 requires lower-bound hull evaluation")
    return defaults


def _evaluation_command(
    table: str,
    entry: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    context: Mapping[str, Any],
    prediction_template: str,
    pred_ppi: float,
    result_dir: Path,
) -> list[str]:
    method = _required_string(entry, "method", "entry")
    display_name = _required_string(entry, "display_name", "entry")
    if not SAFE_KEY.fullmatch(method):
        raise PlanError("entry.method must be a filesystem-safe lowercase key")
    metadata = _mapping(entry.get("metadata", {}), "entry.metadata")
    if evaluation["kind"] == "hull":
        if prediction_template in {"all-background", "all-foreground"}:
            raise PlanError("B1 references are generated by evaluate_hull; do not add them")
        return [
            sys.executable, str(ROOT / "scripts" / "evaluate_hull.py"),
            "--manifest", str(context["evaluation_manifest"]),
            "--pred-template", prediction_template,
            "--method", method,
            "--display-name", display_name,
            "--out", str(result_dir),
            "--split", str(context["split"]),
            "--pred-ppi", str(pred_ppi),
        ]

    if context["splits"] is None:
        raise PlanError("mask evaluation requires splits")
    command = [
        sys.executable, str(ROOT / "scripts" / "evaluate_masks.py"),
        "--manifest", str(context["evaluation_manifest"]),
        "--splits", str(context["splits"]),
        "--subsets", str(context["subset"]),
        "--method", method,
        "--display-name", display_name,
        "--pred-ppi", str(pred_ppi),
        "--target-ppi", "500",
        "--split", str(context["split"]),
        "--gt-column", str(evaluation.get("gt_column", "mask_path")),
        "--table", table,
        "--out", str(result_dir),
    ]
    if prediction_template in {"all-background", "all-foreground"}:
        command.extend(["--reference", prediction_template])
    else:
        command.extend(["--pred-template", prediction_template])
    if evaluation.get("group_by"):
        command.extend(["--group-by", str(evaluation["group_by"])])
    exclusions = evaluation.get("exclude", [])
    if not isinstance(exclusions, list) or not all(isinstance(item, str) for item in exclusions):
        raise PlanError("evaluation.exclude must be a list of COLUMN=VALUE strings")
    for exclusion in exclusions:
        command.extend(["--exclude", exclusion])
    metadata_flags = {
        "seed": "--seed", "year": "--year", "family": "--family",
        "training_data": "--training-data", "comment": "--comment",
        "params_m": "--params-m", "device": "--device",
        "threshold": "--threshold", "run_id": "--run-id",
        "ms_per_img": "--ms-per-img",
    }
    for key, flag in metadata_flags.items():
        _option(command, flag, metadata.get(key))
    return command


def _covered_rows(
    evaluation: Mapping[str, Any], context: Mapping[str, Any]
) -> list[Mapping[str, str]]:
    with Path(context["evaluation_manifest"]).open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if evaluation["kind"] == "hull":
        return [row for row in rows if row.get("split") == context["split"]]
    split_map = json.loads(Path(context["splits"]).read_text(encoding="utf-8"))["splits"]
    split_map = {str(key): value for key, value in split_map.items()}
    gt_column = str(evaluation.get("gt_column", "mask_path"))
    exclusions: list[tuple[str, str]] = []
    for expression in evaluation.get("exclude", []):
        if "=" not in expression:
            raise PlanError(f"invalid exclusion: {expression!r}")
        column, value = expression.split("=", 1)
        exclusions.append((column, value))
    selected = []
    for row in rows:
        if row.get("subset") != context["subset"]:
            continue
        if split_map.get(str(row.get("subject_id"))) != context["split"]:
            continue
        if not row.get(gt_column):
            continue
        if any(str(row.get(column, "")) == value for column, value in exclusions):
            continue
        selected.append(row)
    return selected


def _assert_prediction_coverage(
    template: str, evaluation: Mapping[str, Any], context: Mapping[str, Any]
) -> None:
    if template in {"all-background", "all-foreground"}:
        return
    rows = _covered_rows(evaluation, context)
    if not rows:
        raise PlanError("evaluation selection contains no rows")
    missing = []
    for row in rows:
        try:
            path = Path(template.format(image_id=row["image_id"], subset=context["subset"]))
        except KeyError as error:
            raise PlanError(f"prediction template references unsupported field: {error}") from error
        if not path.is_file():
            missing.append(path)
    if missing:
        examples = ", ".join(str(path) for path in missing[:3])
        raise PlanError(
            f"prediction coverage is incomplete: {len(missing)}/{len(rows)} missing; "
            f"examples: {examples}"
        )


def _run(command: Sequence[str], record: dict[str, Any], stage: str) -> None:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(list(command), cwd=ROOT, check=False)
    record.setdefault("commands", []).append(
        {"stage": stage, "argv": list(command), "returncode": completed.returncode}
    )
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def execute(
    table: str, plan_path: Path, output: Path, dry_run: bool
) -> Mapping[str, Any]:
    """Validate and optionally execute one complete table plan."""
    plan_path = plan_path.resolve()
    _outside_public_tree(plan_path, "reproduction plan")
    plan = _load_plan(plan_path, table)
    output = output.expanduser().resolve()
    _outside_public_tree(output, "reproduction output")
    base = plan_path.parent
    table_a_sources: dict[str, Mapping[str, Any]] = {}
    if table == "A_prime":
        source_receipt = _checked_artifact(
            plan, "table_a_receipt", "table_a_receipt_sha256", base, "plan"
        )
        source_payload = _mapping(
            json.loads(source_receipt.read_text(encoding="utf-8")),
            "Table A reproduction receipt",
        )
        if (
            source_payload.get("schema") != "segbench.reproduction-receipt.v1"
            or source_payload.get("table") != "A"
            or source_payload.get("status") != "PASS"
        ):
            raise PlanError("table_a_receipt is not a successful Table A receipt")
        source_entries = source_payload.get("entries")
        if not isinstance(source_entries, list):
            raise PlanError("table_a_receipt entries are malformed")
        table_a_sources = {
            str(item.get("run_key")): item
            for item in source_entries
            if isinstance(item, dict) and item.get("run_key")
        }
    receipt: dict[str, Any] = {
        "schema": "segbench.reproduction-receipt.v1",
        "status": "DRY_RUN" if dry_run else "RUNNING",
        "table": table,
        "plan_sha256": _sha256(plan_path),
        "entries": [],
    }
    run_keys: set[str] = set()
    prepared = []
    for index, raw_entry in enumerate(plan["entries"]):
        entry = _mapping(raw_entry, f"entries[{index}]")
        run_key = _required_string(entry, "run_key", f"entries[{index}]")
        if not SAFE_KEY.fullmatch(run_key) or run_key in run_keys:
            raise PlanError(f"run_key must be unique and filesystem-safe: {run_key!r}")
        run_keys.add(run_key)
        context = _resolve_context(plan, entry, base, table)
        if table in {"A", "A_prime"} and context["subset"] != "finger_latent":
            raise PlanError(f"Table {table} requires subset='finger_latent'")
        _outside_public_tree(context["source_manifest"], "source manifest")
        _outside_public_tree(context["evaluation_manifest"], "evaluation manifest")
        evaluation = _evaluation(table, plan, entry)
        runner = _mapping(entry.get("runner"), f"entries[{index}].runner")
        run_dir = output / "runs" / run_key
        infer_command, template, pred_ppi = _runner_command(runner, context, run_dir, base)
        if table == "A_prime":
            if runner.get("kind") != "existing_predictions":
                raise PlanError(
                    "Table A-prime must reuse Table A predictions via existing_predictions"
                )
            source = table_a_sources.get(run_key)
            if source is None:
                raise PlanError(f"Table A receipt has no run_key {run_key!r}")
            if (
                source.get("method") != entry.get("method")
                or source.get("prediction_template") != template
                or float(source.get("pred_ppi", -1)) != pred_ppi
            ):
                raise PlanError(
                    f"Table A-prime run {run_key!r} does not exactly reuse Table A predictions"
                )
        eval_command = _evaluation_command(
            table, entry, evaluation, context, template, pred_ppi,
            output / "results" / table / run_key,
        )
        record: dict[str, Any] = {
            "run_key": run_key,
            "method": entry["method"],
            "runner_kind": runner["kind"],
            "prediction_template": template,
            "pred_ppi": pred_ppi,
            "source_manifest_sha256": _sha256(context["source_manifest"]),
            "evaluation_manifest_sha256": _sha256(context["evaluation_manifest"]),
            "splits_sha256": (
                _sha256(context["splits"]) if context["splits"] is not None else None
            ),
            "commands": [],
        }
        prepared.append((record, infer_command, eval_command, template, evaluation, context))

    if dry_run:
        for record, infer_command, eval_command, _, _, _ in prepared:
            if infer_command is not None:
                record["commands"].append(
                    {"stage": "inference", "argv": infer_command, "returncode": None}
                )
            record["commands"].append(
                {"stage": "evaluation", "argv": eval_command, "returncode": None}
            )
            receipt["entries"].append(record)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return receipt

    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / f"reproduce_{table}.json"
    try:
        for record, infer_command, eval_command, template, evaluation, context in prepared:
            receipt["entries"].append(record)
            if infer_command is not None:
                _run(infer_command, record, "inference")
            _assert_prediction_coverage(template, evaluation, context)
            _run(eval_command, record, "evaluation")
            record["status"] = "PASS"
            _atomic_json(receipt_path, receipt)
    except Exception:
        receipt["status"] = "FAIL"
        _atomic_json(receipt_path, receipt)
        raise
    receipt["status"] = "PASS"
    _atomic_json(receipt_path, receipt)
    print(f"Table {table} reproduction PASS; receipt: {receipt_path}")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", choices=sorted(TABLES), required=True)
    parser.add_argument(
        "--config", type=Path,
        help="optional KEY=VALUE file defining REPRO_PLAN and REPRO_OUTPUT",
    )
    parser.add_argument("--plan", type=Path, help=f"private {SCHEMA} JSON plan")
    parser.add_argument("--output", type=Path, help="private reproduction output root")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    values = parse_env(args.config) if args.config else {}
    plan_value = args.plan or values.get("REPRO_PLAN") or os.environ.get("SEGBENCH_REPRO_PLAN")
    output_value = (
        args.output or values.get("REPRO_OUTPUT") or os.environ.get("SEGBENCH_REPRO_OUTPUT")
    )
    if plan_value is None or output_value is None:
        raise SystemExit(
            "provide --plan and --output, or REPRO_PLAN/REPRO_OUTPUT in --config, "
            "or SEGBENCH_REPRO_PLAN/SEGBENCH_REPRO_OUTPUT in the environment"
        )
    try:
        execute(args.table, Path(plan_value), Path(output_value), args.dry_run)
    except PlanError as error:
        raise SystemExit(f"reproduction plan rejected: {error}") from error


if __name__ == "__main__":
    main()
