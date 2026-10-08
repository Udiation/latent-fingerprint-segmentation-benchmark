"""Shared data, optimisation and inference utilities for neural methods.

All learned methods call these functions so normalisation, augmentation,
validation threshold selection and full-image reconstruction remain identical.
The module does not know any private path and never opens a test split while
training.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import random
import subprocess
import tempfile
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, Sampler
import yaml

from segbench import features as block_features
from segbench.io import (
    attach_split,
    load_manifest,
    manifest_target_shape,
    mask_ppi_for,
    read_gray,
    read_mask,
    to_target_ppi,
)
from segbench.postprocess import postprocess_blocks
from segbench import preprocess_cache
from segbench.artifact_identity import atomic_json_once


PROTOCOL_LEARNING_RATES = (3e-4, 1e-3)
PROTOCOL_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
LEGACY_PROTOCOL_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
MAX_CUDA_MEMORY_FRACTION = 0.3
PROTOCOL_DATALOADER = {
    "num_workers": 8,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 4,
}
QUEUE_BATCH_CONTRACT_KIND = "single_oom_half_batch_double_accumulation"


def _positive_int(value: Any, name: str) -> int:
    """Return a positive protocol integer without accepting booleans/floats."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_queue_batch_contract(training: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the sole queue-authorized OOM batch transition.

    The scientific effective batch is immutable.  A queue may either use the
    declared base pair or, once after an authenticated OOM, halve the micro
    batch and double accumulation.  No other scaling is representable.
    """
    base_batch = _positive_int(training.get("batch_size"), "training.batch_size")
    base_accumulation = _positive_int(
        training.get("gradient_accumulation"),
        "training.gradient_accumulation",
    )
    if base_batch % 2:
        raise ValueError(
            "training.batch_size must be even for the authorized OOM transition"
        )
    effective_batch = base_batch * base_accumulation
    return {
        "schema_version": 1,
        "kind": QUEUE_BATCH_CONTRACT_KIND,
        "base": {
            "oom_retry_count": 0,
            "batch_size": base_batch,
            "gradient_accumulation": base_accumulation,
            "effective_batch_size": effective_batch,
        },
        "authorized_oom_retry": {
            "oom_retry_count": 1,
            "batch_size": base_batch // 2,
            "gradient_accumulation": base_accumulation * 2,
            "effective_batch_size": effective_batch,
        },
        "max_oom_retries": 1,
    }


def apply_queue_batch_contract(
    config: MutableMapping[str, Any],
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Authenticate queue OOM state and apply its exact batch pair in-place."""
    environment = os.environ if environ is None else environ
    training = config.get("training")
    runtime = config.get("runtime")
    if not isinstance(training, MutableMapping) or not isinstance(runtime, MutableMapping):
        raise ValueError("config training/runtime sections must be mutable mappings")

    contract = build_queue_batch_contract(training)
    digest = config_digest(contract)
    declared_contract = runtime.get("queue_batch_contract")
    declared_digest = runtime.get("queue_batch_contract_sha256")
    if declared_contract is not None and declared_contract != contract:
        raise ValueError("configured queue batch contract differs from base batch settings")
    if declared_digest is not None and declared_digest != digest:
        raise ValueError("configured queue batch contract digest changed")

    retry_raw = str(environment.get("SEG_OOM_RETRY_COUNT", "0"))
    batch_raw = str(environment.get("SEG_BATCH_SCALE", "1.0"))
    accumulation_raw = str(environment.get("SEG_GRAD_ACCUM_SCALE", "1.0"))
    authorized_environment = {
        ("0", "1.0", "1.0"): "base",
        ("1", "0.5", "2.0"): "oom_retry",
    }
    state = authorized_environment.get((retry_raw, batch_raw, accumulation_raw))
    if state is None:
        raise ValueError(
            "queue batch environment must be exactly base or the single authorized "
            "half-batch/double-accumulation OOM retry"
        )
    active_pair = contract["base" if state == "base" else "authorized_oom_retry"]
    if active_pair["effective_batch_size"] != contract["base"]["effective_batch_size"]:
        raise RuntimeError("authorized OOM transition changed effective batch size")
    training["batch_size"] = int(active_pair["batch_size"])
    training["gradient_accumulation"] = int(active_pair["gradient_accumulation"])
    runtime["queue_batch_contract"] = contract
    runtime["queue_batch_contract_sha256"] = digest
    active = {
        "state": state,
        "oom_retry_count": int(active_pair["oom_retry_count"]),
        "batch_size": int(active_pair["batch_size"]),
        "gradient_accumulation": int(active_pair["gradient_accumulation"]),
        "effective_batch_size": int(active_pair["effective_batch_size"]),
        "contract_sha256": digest,
    }
    runtime["queue_overrides"] = active
    return active


def queue_batch_metadata(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return authenticated stable and active queue-batch metadata."""
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("config runtime section is absent")
    contract = runtime.get("queue_batch_contract")
    digest = runtime.get("queue_batch_contract_sha256")
    active = runtime.get("queue_overrides")
    if not isinstance(contract, Mapping) or not isinstance(active, Mapping):
        raise ValueError("queue batch contract was not applied")
    if contract.get("kind") != QUEUE_BATCH_CONTRACT_KIND:
        raise ValueError("queue batch contract kind changed")
    if digest != config_digest(contract) or active.get("contract_sha256") != digest:
        raise ValueError("queue batch contract/active digest changed")
    state = active.get("state")
    expected_key = "base" if state == "base" else "authorized_oom_retry"
    if state not in {"base", "oom_retry"}:
        raise ValueError("queue batch active state is invalid")
    expected = contract.get(expected_key)
    if not isinstance(expected, Mapping):
        raise ValueError("queue batch contract active pair is absent")
    for key in (
        "oom_retry_count",
        "batch_size",
        "gradient_accumulation",
        "effective_batch_size",
    ):
        if active.get(key) != expected.get(key):
            raise ValueError(f"queue batch active {key} differs from its contract")
    return {
        "contract": dict(contract),
        "contract_sha256": str(digest),
        "active": dict(active),
    }


def validate_resume_batch_transition(
    checkpoint: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    """Reject batch-contract tampering, reverse retries, and arbitrary jumps."""
    current = queue_batch_metadata(config)
    if checkpoint.get("queue_batch_contract_sha256") != current["contract_sha256"]:
        raise ValueError("resume checkpoint queue batch contract differs")
    previous = checkpoint.get("queue_batch_active_state")
    if not isinstance(previous, Mapping):
        raise ValueError("resume checkpoint lacks queue batch active state")
    previous_config = {
        "runtime": {
            "queue_batch_contract": current["contract"],
            "queue_batch_contract_sha256": current["contract_sha256"],
            "queue_overrides": dict(previous),
        }
    }
    previous_meta = queue_batch_metadata(previous_config)
    transition = (
        str(previous_meta["active"]["state"]),
        str(current["active"]["state"]),
    )
    if transition not in {
        ("base", "base"),
        ("base", "oom_retry"),
        ("oom_retry", "oom_retry"),
    }:
        raise ValueError("queue batch resume attempted a reverse or unauthorized transition")


class EpochRandomSampler(Sampler[int]):
    """Deterministic epoch-keyed permutation compatible with persistent workers."""

    def __init__(self, data_source: Any, seed: int) -> None:
        self.data_source = data_source
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(self.data_source.epoch))
        yield from torch.randperm(len(self), generator=generator).tolist()


def protocol_dataloader_settings(training: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the frozen loader settings, rejecting per-method drift."""
    observed = {
        "num_workers": int(training.get("num_workers", -1)),
        "pin_memory": training.get("pin_memory"),
        "persistent_workers": training.get("persistent_workers"),
        "prefetch_factor": int(training.get("prefetch_factor", -1)),
    }
    if observed != PROTOCOL_DATALOADER:
        raise ValueError(
            "training loader settings must exactly match the shared protocol: "
            f"observed={observed}, expected={PROTOCOL_DATALOADER}"
        )
    return dict(PROTOCOL_DATALOADER)


def acceleration_policy(
    receipt: Path | str | None = None,
    *,
    evaluation_mode: str | None = None,
    epochs: int | None = None,
    expected_identity: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Keep acceleration off unless a two-epoch equivalence receipt is supplied.

    Mixed precision is deliberately fail-closed.  A future accepted receipt must
    be a standalone JSON artifact produced before a production launch and must
    prove an absolute validation-Dice delta below 0.005.
    """
    if receipt is not None and evaluation_mode is not None:
        raise ValueError("acceleration receipt and evaluation mode are mutually exclusive")
    if evaluation_mode is not None:
        if evaluation_mode not in {"fp32", "bf16_channels_last"} or epochs != 2:
            raise ValueError(
                "acceleration evaluation requires fp32 or bf16_channels_last for exactly 2 epochs"
            )
        enabled = evaluation_mode == "bf16_channels_last"
        return {
            "precision": "bf16" if enabled else "fp32",
            "channels_last": enabled,
            "acceptance_receipt_sha256": None,
            "evaluation_only": True,
            "production_eligible": False,
            "reason": "isolated two-epoch acceleration acceptance measurement",
        }
    if receipt is None:
        return {
            "precision": "fp32",
            "channels_last": False,
            "acceptance_receipt_sha256": None,
            "evaluation_only": False,
            "production_eligible": True,
            "reason": "no accepted two-epoch equivalence receipt",
        }
    path = Path(receipt)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": 2,
        "kind": "bf16_channels_last_two_epoch_acceptance",
        "accepted": True,
        "epochs": 2,
    }
    disagreements = {
        key: (payload.get(key), value)
        for key, value in required.items()
        if payload.get(key) != value
    }
    delta = float(payload.get("absolute_validation_dice_delta", float("inf")))
    identity_disagreements = {
        key: (payload.get(key), value)
        for key, value in (expected_identity or {}).items()
        if payload.get(key) != value
    }
    run_card_disagreements = {}
    for mode in ("fp32", "bf16"):
        identity = payload.get(f"{mode}_run_card")
        if not isinstance(identity, Mapping):
            run_card_disagreements[mode] = "missing identity"
            continue
        run_card = Path(str(identity.get("path", "")))
        if (
            not run_card.is_file()
            or run_card.is_symlink()
            or identity.get("sha256") != file_digest(run_card)
        ):
            run_card_disagreements[mode] = "bytes changed"
    if (
        disagreements
        or identity_disagreements
        or run_card_disagreements
        or not math.isfinite(delta)
        or not delta < 0.005
    ):
        raise ValueError(
            "mixed-precision receipt does not prove a two-epoch Dice delta < 0.005"
        )
    return {
        "precision": "bf16",
        "channels_last": True,
        "acceptance_receipt_sha256": file_digest(path),
        "evaluation_only": False,
        "production_eligible": True,
        "reason": "accepted two-epoch fp32-vs-bf16 validation comparison",
    }


class EpochProfiler:
    """Measure loader wait, compute time and generic CUDA utilisation per epoch."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.started = time.perf_counter()
        self.waiting_since = self.started
        self.data_wait_seconds = 0.0
        self.compute_seconds = 0.0
        self.compute_started: Optional[float] = None
        self.utilization: List[float] = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def batch_ready(self) -> None:
        now = time.perf_counter()
        self.data_wait_seconds += now - self.waiting_since
        self.compute_started = now

    def compute_done(self) -> None:
        if self.compute_started is None:
            raise RuntimeError("batch_ready must precede compute_done")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            sampler = getattr(torch.cuda, "utilization", None)
            if callable(sampler):
                try:
                    self.utilization.append(float(sampler(self.device)))
                except (RuntimeError, NotImplementedError, OSError):
                    pass
        now = time.perf_counter()
        self.compute_seconds += now - self.compute_started
        self.waiting_since = now
        self.compute_started = None

    def finish(self) -> Dict[str, Any]:
        if self.compute_started is not None:
            raise RuntimeError("cannot finish while a compute interval is open")
        total = time.perf_counter() - self.started
        measured = self.data_wait_seconds + self.compute_seconds
        result: Dict[str, Any] = {
            "total_seconds": total,
            "data_wait_seconds": self.data_wait_seconds,
            "compute_seconds": self.compute_seconds,
            "other_seconds": max(0.0, total - measured),
            "data_wait_fraction": self.data_wait_seconds / total if total else 0.0,
            "compute_fraction": self.compute_seconds / total if total else 0.0,
            "gpu_utilization_sample_count": len(self.utilization),
            "gpu_utilization_percent_mean": (
                float(np.mean(self.utilization)) if self.utilization else None
            ),
            "gpu_utilization_percent_min": (
                float(np.min(self.utilization)) if self.utilization else None
            ),
            "gpu_utilization_percent_max": (
                float(np.max(self.utilization)) if self.utilization else None
            ),
            "peak_device_memory_bytes": (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else None
            ),
        }
        return result


def load_config(
    path: Path | str, *, allow_amendment_legacy_thresholds: bool = False
) -> Dict[str, Any]:
    """Load and validate a benchmark YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    required = {"schema_version", "method", "task", "data", "model", "training", "runtime"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"configuration is missing keys: {missing}")
    if config["task"] not in {"pixel", "patch"}:
        raise ValueError("task must be 'pixel' or 'patch'")
    if config["data"].get("normalization") != "train_only":
        raise ValueError("normalization must be computed from the training split only")
    if "exclude_examiner_values" in config["data"]:
        raise ValueError(
            "use exclude_examiner_values_from_test; train/val exclusions would make "
            "method-family protocols unequal"
        )
    learning_rate = float(config["training"].get("learning_rate", 0.0))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive and finite")
    if not any(math.isclose(learning_rate, candidate) for candidate in PROTOCOL_LEARNING_RATES):
        raise ValueError(
            "training.learning_rate must be one of the pre-registered sweep values "
            f"{PROTOCOL_LEARNING_RATES}"
        )
    thresholds = [float(item) for item in config["training"].get("threshold_candidates", [])]
    if not thresholds or any(not 0.0 < item < 1.0 for item in thresholds):
        raise ValueError("threshold_candidates must be non-empty values in (0, 1)")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("threshold_candidates contains duplicates")
    allowed_threshold_grids = {PROTOCOL_THRESHOLDS}
    if allow_amendment_legacy_thresholds:
        allowed_threshold_grids.add(LEGACY_PROTOCOL_THRESHOLDS)
    if tuple(thresholds) not in allowed_threshold_grids:
        raise ValueError(
            "threshold_candidates must exactly match a pre-registered validation grid "
            "authorized for this run: "
            f"{sorted(allowed_threshold_grids)}"
        )
    memory_fraction = float(config["runtime"].get("memory_fraction", 1.0))
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError("runtime.memory_fraction must be in (0, 1]")
    return config


def examiner_exclusions_for_split(
    data_config: Mapping[str, Any], split: str
) -> Sequence[str]:
    """Apply examiner-value exclusions only to the declared final test split."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown split {split!r}")
    if split != "test":
        return ()
    values = data_config.get("exclude_examiner_values_from_test", ())
    if not isinstance(values, (list, tuple)):
        raise ValueError("exclude_examiner_values_from_test must be a list")
    return tuple(str(value) for value in values)


def config_digest(config: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest of a loaded configuration."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path: Path | str) -> str:
    """Return a streaming SHA-256 digest without exposing private file contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_provenance(
    command: Sequence[str],
    *,
    capture_timing: str = "function_entry",
    repository: Path | str | None = None,
) -> Dict[str, Any]:
    """Snapshot reproducibility metadata once, before a run performs work.

    ``repository`` exists for synthetic tests and is intentionally not emitted,
    so run cards never disclose a host path.  Callers retain the returned mapping
    rather than querying Git again when a long-running job eventually finishes.
    """
    if capture_timing not in {"process_start", "function_entry"}:
        raise ValueError("capture_timing must be process_start or function_entry")
    repository_path = (
        Path(repository) if repository is not None else Path(__file__).resolve().parents[1]
    )
    captured_at_utc = datetime.now(timezone.utc).isoformat()
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
        "captured_at_utc": captured_at_utc,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pandas": pd.__version__,
            "pyyaml": yaml.__version__,
        },
    }


def resume_config_digest(config: Mapping[str, Any]) -> str:
    """Hash scientific settings while excluding queue/runtime mechanics.

    A queue OOM retry is allowed to alter only micro-batch size, accumulation,
    worker count and allocator/device settings.  All model, data, augmentation,
    loss, optimiser, schedule, seed and threshold-selection settings remain in
    this identity hash.
    """
    resolved = json.loads(json.dumps(config))
    runtime = resolved.get("runtime", {})
    for key in (
        "device",
        "memory_fraction",
        "queue_overrides",
        "preprocess_cache_receipt_sha256",
        "threshold_amendment_sha256",
    ):
        runtime.pop(key, None)
    training = resolved.get("training", {})
    for key in (
        "batch_size",
        "gradient_accumulation",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
    ):
        training.pop(key, None)
    return config_digest(resolved)


def seed_everything(seed: int, deterministic: bool) -> None:
    """Seed Python, NumPy and PyTorch and optionally require deterministic kernels."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic))


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto``, ``cpu`` or ``cuda`` without embedding a device index."""
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(requested)


def configure_memory_fraction(device: torch.device, fraction: float) -> None:
    """Apply a per-process CUDA allocator ceiling when running on CUDA."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("memory fraction must be in (0, 1]")
    if device.type == "cuda":
        if fraction > MAX_CUDA_MEMORY_FRACTION:
            raise ValueError(
                "CUDA memory fraction exceeds the benchmark per-process ceiling "
                f"of {MAX_CUDA_MEMORY_FRACTION}"
            )
        # ``torch.device("cuda")`` has no index.  Under CUDA_VISIBLE_DEVICES,
        # resolve it to the current process-local device before calling the
        # allocator API, which requires a concrete logical index.
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(
            float(fraction), device=int(device_index)
        )


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python from the deterministic PyTorch worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def loader_generator(seed: int) -> torch.Generator:
    """Return the explicitly seeded generator used by every DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def load_split_rows(
    manifest_path: Path | str,
    split_path: Path | str,
    *,
    subset: str | Sequence[str],
    split: str,
    mask_column: str,
    require_mask: bool = True,
    exclude_examiner_values: Sequence[str] = (),
    limit: int = 0,
) -> pd.DataFrame:
    """Load one declared subset/split, rejecting missing paths or split leakage."""
    with Path(split_path).open("r", encoding="utf-8") as handle:
        split_payload = json.load(handle)
    if not isinstance(split_payload, dict) or not isinstance(split_payload.get("splits"), dict):
        raise ValueError("split JSON must contain a 'splits' subject mapping")
    frame = attach_split(load_manifest(manifest_path), split_payload["splits"])
    if mask_column not in frame.columns:
        raise ValueError(f"manifest is missing configured mask column: {mask_column}")
    subsets = [subset] if isinstance(subset, str) else list(subset)
    if not subsets or any(not isinstance(value, str) or not value for value in subsets):
        raise ValueError("subset must be a non-empty string or sequence of strings")
    frame = frame[frame["subset"].isin(subsets) & (frame["split"] == split)].copy()
    if exclude_examiner_values and "examiner_value" in frame.columns:
        excluded = {str(item).upper() for item in exclude_examiner_values}
        frame = frame[~frame["examiner_value"].astype(str).str.upper().isin(excluded)]
    frame[mask_column] = frame[mask_column].fillna("").astype(str)
    if require_mask:
        frame = frame[frame[mask_column].str.len() > 0]
    frame = frame.sort_values("image_id", kind="stable").reset_index(drop=True)
    if limit:
        frame = frame.head(limit).copy()
    if frame.empty:
        raise ValueError(f"no rows for subset={subsets!r}, split={split!r}")
    return frame


def assert_compatible_subject_splits(
    primary_path: Path | str, target_path: Path | str
) -> None:
    """Reject domain-alignment split files that disagree on any subject.

    Separate manifests are allowed, but shared SD302 subject identifiers must
    always map to the same partition.  This prevents unlabeled test-subject
    exemplars from entering adversarial alignment through a second split file.
    """
    mappings = []
    for path in (primary_path, target_path):
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        mapping = payload.get("splits") if isinstance(payload, dict) else None
        if not isinstance(mapping, dict):
            raise ValueError(f"split file {path} lacks a subject mapping")
        normalized = {str(key): str(value) for key, value in mapping.items()}
        invalid = sorted(
            subject
            for subject, assignment in normalized.items()
            if assignment not in {"train", "val", "test"}
        )
        if invalid:
            raise ValueError(
                f"split file {path} has invalid assignments, e.g. "
                + ", ".join(invalid[:10])
            )
        mappings.append(normalized)
    overlap = set(mappings[0]).intersection(mappings[1])
    mismatches = sorted(
        subject for subject in overlap if mappings[0][subject] != mappings[1][subject]
    )
    if mismatches:
        raise ValueError(
            "domain-alignment split mismatch for shared subjects, e.g. "
            + ", ".join(mismatches[:10])
        )
    target_only = sorted(set(mappings[1]) - set(mappings[0]))
    if target_only:
        raise ValueError(
            "domain-alignment target subjects are absent from the authoritative "
            "source split, e.g. "
            + ", ".join(target_only[:10])
        )


def _canonical_target_shape(
    row: Mapping[str, Any], target_ppi: float
) -> Optional[Tuple[int, int]]:
    """Read an optional manifest-pinned 500-ppi ``(height, width)``.

    SD302 image files and examiner masks can live on different native PPI
    grids.  The manifest therefore pins their shared canonical output shape;
    using the image's PPI for a mask would silently misregister ground truth.
    """
    target_keys = ("target_ppi", "target_width", "target_height")
    present = [
        row.get(key) is not None
        and not pd.isna(row.get(key))
        and str(row.get(key)).strip() != ""
        for key in target_keys
    ]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            f"{row.get('image_id', '<unknown>')}: target_ppi, target_width and "
            "target_height must be supplied together"
        )
    return manifest_target_shape(row, target_ppi)


def _positive_ppi(row: Mapping[str, Any], column: str) -> float:
    value = float(row[column])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{row.get('image_id', '<unknown>')}: invalid {column} {value}"
        )
    return value


def verify_source_image(row: Mapping[str, Any]) -> str:
    """Verify one manifest-bound source image before any decoder opens it."""
    expected = row.get("source_sha256")
    if not isinstance(expected, str) or len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError(
            f"{row.get('image_id', '<unknown>')}: source_sha256 is absent or invalid"
        )
    observed = file_digest(Path(str(row["image_path"])))
    if observed != expected:
        raise ValueError(
            f"{row.get('image_id', '<unknown>')}: source image bytes changed"
        )
    return observed


def read_normalized_image(
    row: Mapping[str, Any],
    target_ppi: float,
    *,
    verify_source: bool = False,
    cache_dir: Path | str | None = None,
) -> np.ndarray:
    """Read and resample a manifest image, returning float32 in [0, 1]."""
    source_ppi = _positive_ppi(row, "ppi")
    if not math.isfinite(target_ppi) or target_ppi <= 0:
        raise ValueError(f"invalid target ppi: {target_ppi}")
    if verify_source:
        verify_source_image(row)
    if cache_dir is not None:
        return preprocess_cache.read_image(cache_dir, str(row["image_id"]))
    source = read_gray(row["image_path"])
    canonical_shape = _canonical_target_shape(row, target_ppi)
    if canonical_shape is None:
        image = to_target_ppi(source, source_ppi, target_ppi)
    else:
        image = cv2.resize(
            source,
            (canonical_shape[1], canonical_shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return image.astype(np.float32) / 255.0


def read_pair(
    row: Mapping[str, Any],
    target_ppi: float,
    mask_column: str,
    *,
    cache_dir: Path | str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read one image/mask pair at the common resolution."""
    if cache_dir is not None:
        return preprocess_cache.read_pair(
            cache_dir, str(row["image_id"]), mask_column
        )
    image = read_normalized_image(row, target_ppi)
    mask_ppi = mask_ppi_for(row, mask_column)
    if not math.isfinite(mask_ppi) or mask_ppi <= 0:
        raise ValueError(
            f"{row.get('image_id', '<unknown>')}: invalid mask PPI {mask_ppi}"
        )
    source_mask = read_mask(row[mask_column])
    canonical_shape = _canonical_target_shape(row, target_ppi)
    if canonical_shape is None:
        mask = to_target_ppi(source_mask, mask_ppi, target_ppi, is_mask=True)
    else:
        mask = cv2.resize(
            source_mask.astype(np.uint8),
            (canonical_shape[1], canonical_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    if image.shape != mask.shape:
        raise ValueError(
            f"{row['image_id']}: image shape {image.shape} != mask shape {mask.shape}"
        )
    return image, mask.astype(bool)


def compute_train_normalization(
    train_rows: pd.DataFrame,
    target_ppi: float,
    *,
    cache_dir: Path | str | None = None,
) -> Dict[str, float | int | str]:
    """Compute pixel mean/std from training images only using stable float64 sums."""
    total = 0.0
    total_squares = 0.0
    count = 0
    for row in train_rows.to_dict("records"):
        image = read_normalized_image(
            row, target_ppi, cache_dir=cache_dir
        ).astype(np.float64, copy=False)
        total += float(image.sum(dtype=np.float64))
        total_squares += float(np.square(image).sum(dtype=np.float64))
        count += int(image.size)
    if count == 0:
        raise ValueError("cannot compute normalization from an empty training split")
    mean = total / count
    variance = max(total_squares / count - mean * mean, 0.0)
    std = math.sqrt(variance)
    if not math.isfinite(std) or std < 1e-8:
        raise ValueError(f"training-only normalization has invalid std: {std}")
    return {"mean": mean, "std": std, "pixel_count": count, "source_split": "train"}


def _rng(seed: int, epoch: int, index: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(seed), int(epoch), int(index)])
    return np.random.default_rng(sequence)


def _pad(array: np.ndarray, top: int, bottom: int, left: int, right: int, *, mask: bool) -> np.ndarray:
    if not any((top, bottom, left, right)):
        return array
    if mask:
        return np.pad(array, ((top, bottom), (left, right)), mode="constant")
    mode = "reflect" if min(array.shape[:2]) > 1 else "edge"
    return np.pad(array, ((top, bottom), (left, right)), mode=mode)


def random_crop_pair(
    image: np.ndarray, mask: np.ndarray, size: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad if needed, then draw one paired random square crop."""
    if size <= 0:
        raise ValueError("crop size must be positive")
    pad_h = max(0, size - image.shape[0])
    pad_w = max(0, size - image.shape[1])
    image = _pad(image, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2, mask=False)
    mask = _pad(mask, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2, mask=True)
    top = int(rng.integers(0, image.shape[0] - size + 1))
    left = int(rng.integers(0, image.shape[1] - size + 1))
    return (
        image[top : top + size, left : left + size].copy(),
        mask[top : top + size, left : left + size].copy(),
    )


def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    augmentation: Mapping[str, Any],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the shared geometric and photometric augmentation distribution."""
    if rng.random() < float(augmentation["horizontal_flip_probability"]):
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()
    angle = rng.uniform(
        -float(augmentation["max_rotation_degrees"]),
        float(augmentation["max_rotation_degrees"]),
    )
    scale = rng.uniform(float(augmentation["scale_min"]), float(augmentation["scale_max"]))
    height, width = image.shape
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle, scale)
    image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    mask = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    contrast = 1.0 + rng.uniform(-float(augmentation["contrast"]), float(augmentation["contrast"]))
    brightness = rng.uniform(-float(augmentation["brightness"]), float(augmentation["brightness"]))
    image = (image - 0.5) * contrast + 0.5 + brightness
    noise_std = float(augmentation["gaussian_noise_std"])
    if noise_std > 0:
        image = image + rng.normal(0.0, noise_std, image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0).astype(np.float32), mask


class PixelCropDataset(Dataset[Tuple[Tensor, Tensor]]):
    """Deterministic-per-epoch paired crops for pixel-model training."""

    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        target_ppi: float,
        mask_column: str,
        crop_size: int,
        normalization: Mapping[str, float],
        augmentation: Mapping[str, Any],
        seed: int,
        cache_dir: Path | str | None = None,
    ) -> None:
        self.rows = rows.to_dict("records")
        self.target_ppi = float(target_ppi)
        self.mask_column = mask_column
        self.crop_size = int(crop_size)
        self.mean = float(normalization["mean"])
        self.std = float(normalization["std"])
        self.augmentation = dict(augmentation)
        self.seed = int(seed)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._epoch = mp.Value("q", 0, lock=True)

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image, mask = read_pair(
            self.rows[index],
            self.target_ppi,
            self.mask_column,
            cache_dir=self.cache_dir,
        )
        rng = _rng(self.seed, self.epoch, index)
        image, mask = random_crop_pair(image, mask, self.crop_size, rng)
        image, mask = augment_pair(image, mask, self.augmentation, rng)
        image = (image - self.mean) / self.std
        return (
            torch.from_numpy(image[None].astype(np.float32, copy=False)),
            torch.from_numpy(mask[None].astype(np.float32, copy=False)),
        )


class UnlabelledPixelCropDataset(Dataset[Tensor]):
    """Train-split-only crops for the optional RA-RUnet target domain."""

    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        target_ppi: float,
        crop_size: int,
        normalization: Mapping[str, float],
        augmentation: Mapping[str, Any],
        seed: int,
        cache_dir: Path | str | None = None,
    ) -> None:
        self.rows = rows.to_dict("records")
        self.target_ppi = float(target_ppi)
        self.crop_size = int(crop_size)
        self.mean = float(normalization["mean"])
        self.std = float(normalization["std"])
        self.augmentation = dict(augmentation)
        self.seed = int(seed)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._epoch = mp.Value("q", 0, lock=True)

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tensor:
        image = read_normalized_image(
            self.rows[index], self.target_ppi, cache_dir=self.cache_dir
        )
        dummy = np.zeros_like(image, dtype=bool)
        rng = _rng(self.seed, self.epoch, index)
        image, dummy = random_crop_pair(image, dummy, self.crop_size, rng)
        image, _ = augment_pair(image, dummy, self.augmentation, rng)
        return torch.from_numpy((((image - self.mean) / self.std)[None]).astype(np.float32))


class PatchDataset(Dataset[Tuple[Tensor, Tensor]]):
    """Balanced centred-block patch samples derived only from training rows."""

    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        target_ppi: float,
        mask_column: str,
        patch_size: int,
        block_size: int,
        blocks_per_class_per_image: int,
        normalization: Mapping[str, float],
        augmentation: Mapping[str, Any],
        seed: int,
        cache_images: int = 4,
        cache_dir: Path | str | None = None,
    ) -> None:
        if block_size != block_features.BLOCK:
            raise ValueError(
                f"shared post-processing is frozen at block size {block_features.BLOCK}"
            )
        if patch_size < block_size or (patch_size - block_size) % 2:
            raise ValueError("patch size must centre the output block exactly")
        self.rows = rows.to_dict("records")
        self.target_ppi = float(target_ppi)
        self.mask_column = mask_column
        self.patch_size = int(patch_size)
        self.block_size = int(block_size)
        self.mean = float(normalization["mean"])
        self.std = float(normalization["std"])
        self.augmentation = dict(augmentation)
        self.seed = int(seed)
        self._epoch = mp.Value("q", 0, lock=True)
        self.cache_images = int(cache_images)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._cache: MutableMapping[int, Tuple[np.ndarray, np.ndarray]] = OrderedDict()
        samples_by_class: Dict[bool, List[Tuple[int, int, bool]]] = {
            False: [],
            True: [],
        }
        for row_index, row in enumerate(self.rows):
            _, mask = read_pair(
                row,
                self.target_ppi,
                self.mask_column,
                cache_dir=self.cache_dir,
            )
            grid_h, grid_w = block_features.grid_shape(*mask.shape)
            labels = block_features.block_labels(mask, grid_h, grid_w).ravel()
            for class_index, class_value in enumerate((False, True)):
                candidates = np.flatnonzero(labels == class_value)
                row_rng = _rng(self.seed, row_index, class_index)
                take = min(int(blocks_per_class_per_image), len(candidates))
                if take:
                    selected = row_rng.choice(candidates, size=take, replace=False)
                    samples_by_class[class_value].extend(
                        (row_index, int(flat_index), class_value)
                        for flat_index in selected
                    )
        balanced_count = min(len(samples_by_class[False]), len(samples_by_class[True]))
        if balanced_count == 0:
            raise ValueError("both foreground and background blocks are required")
        balance_rng = _rng(self.seed, 0, balanced_count)
        samples: List[Tuple[int, int, bool]] = []
        for class_value in (False, True):
            class_samples = samples_by_class[class_value]
            if len(class_samples) > balanced_count:
                chosen = balance_rng.choice(
                    len(class_samples), size=balanced_count, replace=False
                )
                samples.extend(class_samples[int(index)] for index in chosen)
            else:
                samples.extend(class_samples)
        order_rng = _rng(self.seed, 0, len(samples))
        order_rng.shuffle(samples)
        self.samples = samples

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def __len__(self) -> int:
        return len(self.samples)

    def _pair(self, row_index: int) -> Tuple[np.ndarray, np.ndarray]:
        if row_index in self._cache:
            image, mask = self._cache.pop(row_index)
            self._cache[row_index] = (image, mask)
            return image, mask
        pair = read_pair(
            self.rows[row_index],
            self.target_ppi,
            self.mask_column,
            cache_dir=self.cache_dir,
        )
        self._cache[row_index] = pair
        while len(self._cache) > self.cache_images:
            self._cache.pop(next(iter(self._cache)))
        return pair

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        row_index, flat_index, _sampled_target = self.samples[index]
        image, mask = self._pair(row_index)
        _, grid_w = block_features.grid_shape(*image.shape)
        block_y, block_x = divmod(flat_index, grid_w)
        centre_y = block_y * self.block_size + self.block_size // 2
        centre_x = block_x * self.block_size + self.block_size // 2
        padding = _patch_grid_padding(image.shape, self.patch_size, self.block_size)
        padded_image = _pad(image, *padding, mask=False)
        padded_mask = _pad(mask, *padding, mask=True)
        patch = padded_image[centre_y : centre_y + self.patch_size, centre_x : centre_x + self.patch_size].copy()
        patch_mask = padded_mask[centre_y : centre_y + self.patch_size, centre_x : centre_x + self.patch_size].copy()
        rng = _rng(self.seed, self.epoch, index)
        patch, patch_mask = augment_pair(patch, patch_mask, self.augmentation, rng)
        margin = (self.patch_size - self.block_size) // 2
        central_block = patch_mask[
            margin : margin + self.block_size,
            margin : margin + self.block_size,
        ]
        target = bool(central_block.mean() >= 0.5)
        patch = (patch - self.mean) / self.std
        return (
            torch.from_numpy(patch[None].astype(np.float32, copy=False)),
            torch.tensor(float(target), dtype=torch.float32),
        )


class UnlabelledPatchDataset(Dataset[Tensor]):
    """Deterministic train-image patches for label-free autoencoder pretraining.

    This dataset never opens a mask path.  Sampling therefore cannot leak
    foreground labels into the unsupervised CNN+AE pretraining stage.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        target_ppi: float,
        patch_size: int,
        block_size: int,
        patches_per_image: int,
        normalization: Mapping[str, float],
        augmentation: Mapping[str, Any],
        seed: int,
        cache_images: int = 4,
        cache_dir: Path | str | None = None,
    ) -> None:
        if block_size != block_features.BLOCK:
            raise ValueError(
                f"shared patch grid is frozen at block size {block_features.BLOCK}"
            )
        if patch_size < block_size or (patch_size - block_size) % 2:
            raise ValueError("patch size must centre the output block exactly")
        if patches_per_image < 1:
            raise ValueError("patches_per_image must be positive")
        self.rows = rows.to_dict("records")
        self.target_ppi = float(target_ppi)
        self.patch_size = int(patch_size)
        self.block_size = int(block_size)
        self.mean = float(normalization["mean"])
        self.std = float(normalization["std"])
        self.augmentation = dict(augmentation)
        self.seed = int(seed)
        self._epoch = mp.Value("q", 0, lock=True)
        self.cache_images = int(cache_images)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._cache: MutableMapping[int, np.ndarray] = OrderedDict()
        samples: List[Tuple[int, int]] = []
        for row_index, row in enumerate(self.rows):
            image = read_normalized_image(
                row, self.target_ppi, cache_dir=self.cache_dir
            )
            grid_h, grid_w = block_features.grid_shape(*image.shape)
            candidates = np.arange(grid_h * grid_w)
            row_rng = _rng(self.seed, row_index, len(candidates))
            take = min(int(patches_per_image), len(candidates))
            selected = row_rng.choice(candidates, size=take, replace=False)
            samples.extend((row_index, int(flat_index)) for flat_index in selected)
        if not samples:
            raise ValueError("no unlabelled training patches were found")
        order_rng = _rng(self.seed, 0, len(samples))
        order_rng.shuffle(samples)
        self.samples = samples

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def __len__(self) -> int:
        return len(self.samples)

    def _image(self, row_index: int) -> np.ndarray:
        if row_index in self._cache:
            image = self._cache.pop(row_index)
            self._cache[row_index] = image
            return image
        image = read_normalized_image(
            self.rows[row_index], self.target_ppi, cache_dir=self.cache_dir
        )
        self._cache[row_index] = image
        while len(self._cache) > self.cache_images:
            self._cache.pop(next(iter(self._cache)))
        return image

    def __getitem__(self, index: int) -> Tensor:
        row_index, flat_index = self.samples[index]
        image = self._image(row_index)
        _, grid_w = block_features.grid_shape(*image.shape)
        block_y, block_x = divmod(flat_index, grid_w)
        centre_y = block_y * self.block_size + self.block_size // 2
        centre_x = block_x * self.block_size + self.block_size // 2
        padding = _patch_grid_padding(image.shape, self.patch_size, self.block_size)
        padded = _pad(image, *padding, mask=False)
        patch = padded[
            centre_y : centre_y + self.patch_size,
            centre_x : centre_x + self.patch_size,
        ].copy()
        dummy_mask = np.zeros_like(patch, dtype=bool)
        rng = _rng(self.seed, self.epoch, index)
        patch, _ = augment_pair(patch, dummy_mask, self.augmentation, rng)
        patch = (patch - self.mean) / self.std
        return torch.from_numpy(patch[None].astype(np.float32, copy=False))


def soft_dice_loss(logits: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Batch-mean soft Dice loss on raw logits."""
    probability = torch.sigmoid(logits)
    dimensions = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dimensions)
    denominator = probability.sum(dim=dimensions) + target.sum(dim=dimensions)
    return (1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


def segmentation_loss(
    output: Tensor | Sequence[Tensor],
    target: Tensor,
    *,
    bce_weight: float,
    dice_weight: float,
) -> Tensor:
    """Equally average deep-supervision heads under the shared BCE+Dice loss."""
    outputs = list(output) if isinstance(output, (list, tuple)) else [output]
    losses = []
    for logits in outputs:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        dice = soft_dice_loss(logits, target)
        losses.append(float(bce_weight) * bce + float(dice_weight) * dice)
    return torch.stack(losses).mean()


def final_segmentation_logits(output: Any) -> Tensor:
    """Select the final UNet++ head, otherwise require a tensor output."""
    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("model returned an empty output sequence")
        output = output[-1]
    if not isinstance(output, Tensor):
        raise TypeError(f"expected tensor logits, got {type(output).__name__}")
    return output


def _pad_to_multiple(image: np.ndarray, multiple: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    if multiple <= 0:
        raise ValueError("pad multiple must be positive")
    height, width = image.shape
    out_h = int(math.ceil(height / multiple) * multiple)
    out_w = int(math.ceil(width / multiple) * multiple)
    padded = _pad(image, 0, out_h - height, 0, out_w - width, mask=False)
    return padded, (height, width)


@torch.no_grad()
def predict_pixel_probability(
    model: nn.Module,
    normalized_image: np.ndarray,
    device: torch.device,
    *,
    pad_multiple: int,
    tile_size: int,
    tile_overlap: int,
    enable_dropout: bool = False,
    autocast_bf16: bool = False,
    channels_last: bool = False,
) -> np.ndarray:
    """Predict a full-image probability map with deterministic overlap tiling."""
    model.eval()
    if enable_dropout:
        # Monte Carlo dropout keeps normalisation layers frozen while enabling
        # only stochastic dropout modules.
        for module in model.modules():
            if isinstance(
                module,
                (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d),
            ):
                module.train()
    height, width = normalized_image.shape
    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        padded, original_shape = _pad_to_multiple(normalized_image, pad_multiple)
        tensor = torch.from_numpy(padded[None, None].astype(np.float32)).to(device)
        if channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_bf16,
        ):
            logits = final_segmentation_logits(model(tensor))
        probability = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
        return probability[: original_shape[0], : original_shape[1]]
    if not 0 <= tile_overlap < tile_size:
        raise ValueError("tile_overlap must be in [0, tile_size)")
    pad_h = max(0, tile_size - height)
    pad_w = max(0, tile_size - width)
    padded = _pad(normalized_image, 0, pad_h, 0, pad_w, mask=False)
    padded_h, padded_w = padded.shape
    stride = tile_size - tile_overlap

    def starts(length: int) -> List[int]:
        values = list(range(0, max(length - tile_size, 0) + 1, stride))
        last = length - tile_size
        if not values or values[-1] != last:
            values.append(last)
        return values

    accumulated = np.zeros_like(padded, dtype=np.float64)
    counts = np.zeros_like(padded, dtype=np.float64)
    for top in starts(padded_h):
        for left in starts(padded_w):
            tile = padded[top : top + tile_size, left : left + tile_size]
            tensor = torch.from_numpy(tile[None, None].astype(np.float32)).to(device)
            if channels_last:
                tensor = tensor.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_bf16,
            ):
                logits_tensor = final_segmentation_logits(model(tensor))[0, 0]
            logits = logits_tensor.float().cpu().numpy()
            accumulated[top : top + tile_size, left : left + tile_size] += logits
            counts[top : top + tile_size, left : left + tile_size] += 1.0
    if np.any(counts == 0):
        raise RuntimeError("tiling failed to cover the full image")
    merged_logits = (accumulated / counts)[:height, :width]
    probability = 1.0 / (1.0 + np.exp(-np.clip(merged_logits, -80.0, 80.0)))
    return probability.astype(np.float32)


def _patch_grid_padding(
    image_shape: Tuple[int, int], patch_size: int, block_size: int
) -> Tuple[int, int, int, int]:
    """Return contextual padding that also covers the final partial block.

    The shared block grid uses ceiling division, so its last row or column can
    extend beyond an image whose dimensions are not multiples of ``block_size``.
    Pad that partial grid extent in addition to the centred patch halo; this
    preserves every block centre and the existing outside-image label policy.
    """
    height, width = image_shape
    grid_h, grid_w = block_features.grid_shape(height, width)
    half = patch_size // 2
    return (
        half,
        half + grid_h * block_size - height,
        half,
        half + grid_w * block_size - width,
    )


def _extract_all_patches(
    image: np.ndarray, patch_size: int, block_size: int
) -> Tuple[np.ndarray, int, int]:
    grid_h, grid_w = block_features.grid_shape(*image.shape)
    padding = _patch_grid_padding(image.shape, patch_size, block_size)
    padded = _pad(image, *padding, mask=False)
    patches: List[np.ndarray] = []
    for block_y in range(grid_h):
        for block_x in range(grid_w):
            centre_y = block_y * block_size + block_size // 2
            centre_x = block_x * block_size + block_size // 2
            patches.append(
                padded[
                    centre_y : centre_y + patch_size,
                    centre_x : centre_x + patch_size,
                ]
            )
    return np.stack(patches).astype(np.float32), grid_h, grid_w


@torch.no_grad()
def predict_patch_probability(
    model: nn.Module,
    normalized_image: np.ndarray,
    device: torch.device,
    *,
    patch_size: int,
    block_size: int,
    batch_size: int,
    autocast_bf16: bool = False,
    channels_last: bool = False,
) -> np.ndarray:
    """Return one foreground probability per output block."""
    if batch_size <= 0:
        raise ValueError("patch inference batch size must be positive")
    model.eval()
    patches, grid_h, grid_w = _extract_all_patches(
        normalized_image, patch_size, block_size
    )
    outputs: List[np.ndarray] = []
    for start in range(0, len(patches), batch_size):
        batch = torch.from_numpy(patches[start : start + batch_size, None]).to(device)
        if channels_last:
            batch = batch.contiguous(memory_format=torch.channels_last)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_bf16,
        ):
            logits = model(batch)
        outputs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(outputs).reshape(grid_h, grid_w)


def blocks_to_postprocessed_mask(
    block_probability: np.ndarray,
    image_shape: Tuple[int, int],
    *,
    threshold: float,
    min_component_blocks: int,
) -> np.ndarray:
    """Apply the frozen block cleanup and expand blocks to image pixels."""
    block_mask = postprocess_blocks(
        block_probability >= threshold, min_component_blocks=min_component_blocks
    )
    return block_features.blocks_to_mask(block_mask, image_shape).astype(bool)


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    """Foreground Dice with the shared both-empty convention."""
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = int(np.count_nonzero(prediction & target))
    denominator = int(np.count_nonzero(prediction)) + int(np.count_nonzero(target))
    return 1.0 if denominator == 0 else 2.0 * intersection / denominator


def select_validation_threshold(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    candidates: Sequence[float],
) -> Tuple[float, float, Dict[str, float]]:
    """Select a threshold solely from validation predictions.

    Ties are resolved by proximity to 0.5 and then by the lower threshold.  The
    rule is deterministic and identical for all methods.
    """
    if len(probabilities) != len(targets) or not probabilities:
        raise ValueError("validation probabilities and targets must have equal non-zero length")
    scores: Dict[str, float] = {}
    for threshold in sorted(float(item) for item in candidates):
        values = [
            binary_dice(probability >= threshold, target)
            for probability, target in zip(probabilities, targets)
        ]
        scores[f"{threshold:.8g}"] = float(np.mean(values))
    selected = max(
        ((float(key), score) for key, score in scores.items()),
        key=lambda item: (item[1], -abs(item[0] - 0.5), -item[0]),
    )
    return selected[0], selected[1], scores


def full_image_validation_loss(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    *,
    bce_weight: float,
    dice_weight: float,
    epsilon: float = 1e-6,
) -> float:
    """Compute the configured BCE + soft-Dice objective on full val images.

    This is a reporting/health metric only.  Model selection remains based on
    validation Dice and its pre-registered threshold grid.  Loss is computed
    per image and macro-averaged so differently sized validation images do not
    receive different weights.  The formula is equivalent to
    :func:`segmentation_loss` after the sigmoid for a single segmentation head.
    """
    if len(probabilities) != len(targets):
        raise ValueError("validation probabilities and targets must have equal length")
    if not probabilities:
        raise ValueError("validation loss requires at least one image")
    if not math.isfinite(bce_weight) or not math.isfinite(dice_weight):
        raise ValueError("validation loss weights must be finite")
    if bce_weight < 0.0 or dice_weight < 0.0:
        raise ValueError("validation loss weights must be non-negative")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("validation loss epsilon must be in (0, 0.5)")

    losses = []
    for index, (probability_value, target_value) in enumerate(
        zip(probabilities, targets)
    ):
        probability = np.asarray(probability_value, dtype=np.float64)
        target = np.asarray(target_value, dtype=bool)
        if probability.shape != target.shape:
            raise ValueError(
                f"validation image {index}: probability shape {probability.shape} "
                f"!= target shape {target.shape}"
            )
        if probability.size == 0:
            raise ValueError(f"validation image {index}: empty probability map")
        if not np.isfinite(probability).all():
            raise FloatingPointError(
                f"validation image {index}: non-finite probability"
            )
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError(
                f"validation image {index}: probability outside [0, 1]"
            )
        clipped = np.clip(probability, epsilon, 1.0 - epsilon)
        target_float = target.astype(np.float64, copy=False)
        bce = -np.mean(
            target_float * np.log(clipped)
            + (1.0 - target_float) * np.log1p(-clipped)
        )
        intersection = float(np.sum(probability * target_float))
        denominator = float(np.sum(probability) + np.sum(target_float))
        soft_dice = 1.0 - (2.0 * intersection + epsilon) / (
            denominator + epsilon
        )
        loss = float(bce_weight) * float(bce) + float(dice_weight) * soft_dice
        if not math.isfinite(loss):
            raise FloatingPointError(
                f"validation image {index}: non-finite validation loss"
            )
        losses.append(loss)
    result = float(np.mean(losses))
    if not math.isfinite(result):
        raise FloatingPointError("non-finite mean validation loss")
    return result


def atomic_json(path: Path | str, payload: Any) -> None:
    """Atomically write JSON without leaving a partially written run record."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def write_per_epoch_profile_card(
    path: Path | str,
    *,
    run_card_path: Path | str,
    method: str,
    seed: int,
    source_config_sha256: str,
    history: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Publish the complete resumed epoch-profile series for Stage 4 audit."""
    if not history:
        raise ValueError("cannot publish an empty per-epoch training profile")
    run_card_file = Path(run_card_path)
    try:
        run_card = json.loads(run_card_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot authenticate the bound training run card") from exc
    if not isinstance(run_card, Mapping):
        raise ValueError("bound training run card must be a JSON object")
    expected_epochs = int(run_card.get("epochs_completed", -1))
    if expected_epochs < 1:
        raise ValueError("bound training run card has no completed epochs")
    if (
        str(run_card.get("method")) != str(method)
        or int(run_card.get("seed", -1)) != int(seed)
        or str(run_card.get("source_config_sha256")) != str(source_config_sha256)
    ):
        raise ValueError("per-epoch profile identity differs from the run card")
    profiles: List[Dict[str, Any]] = []
    required = {
        "data_wait_seconds",
        "compute_seconds",
        "gpu_utilization_sample_count",
        "gpu_utilization_percent_mean",
    }
    for entry in history:
        profile = entry.get("epoch_profile")
        if not isinstance(profile, Mapping) or not required.issubset(profile):
            raise ValueError("training history lacks a complete epoch profile")
        profiles.append({"epoch": int(entry["epoch"]), **dict(profile)})
    epochs = [item["epoch"] for item in profiles]
    if epochs != list(range(expected_epochs)):
        raise ValueError(
            "training history must contain every completed epoch exactly once"
        )
    utilisation_available = all(
        int(item["gpu_utilization_sample_count"]) > 0
        and item["gpu_utilization_percent_mean"] is not None
        and math.isfinite(float(item["gpu_utilization_percent_mean"]))
        for item in profiles
    )
    card: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "per_epoch_training_profile_card",
        "method": str(method),
        "seed": int(seed),
        "source_config_sha256": str(source_config_sha256),
        "run_card_sha256": file_digest(run_card_file),
        "complete": True,
        "data_wait_seconds_recorded_per_epoch": True,
        "model_compute_seconds_recorded_per_epoch": True,
        "generic_device_utilisation_recorded": utilisation_available,
        "generic_device_utilisation_unavailable_reason": (
            None
            if utilisation_available
            else "one or more epochs contain no generic device-utilisation sample"
        ),
        "epochs_recorded": len(profiles),
        "epoch_profiles": profiles,
    }
    atomic_json_once(Path(path), card)
    return card


def publish_run_with_profile(
    run_card_path: Path | str,
    profile_card_path: Path | str,
    *,
    run_record: Mapping[str, Any],
    method: str,
    seed: int,
    source_config_sha256: str,
    history: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Publish immutable run/profile cards and repair a run-only crash window."""
    run_path = Path(run_card_path)
    effective = dict(run_record)
    if run_path.is_file():
        try:
            existing = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("existing immutable run card is invalid") from exc
        stable_keys = (
            "method",
            "task",
            "seed",
            "epochs_completed",
            "source_config_sha256",
            "config_sha256",
            "resume_config_sha256",
            "manifest_sha256",
            "split_sha256",
            "checkpoint_sha256",
            "selected_validation_threshold",
            "threshold_amendment_sha256",
        )
        disagreements = {
            key: (existing.get(key), run_record.get(key))
            for key in stable_keys
            if existing.get(key) != run_record.get(key)
        }
        if disagreements:
            raise ValueError(
                f"immutable run-card scientific identity changed: {disagreements}"
            )
        effective = dict(existing)
    else:
        atomic_json_once(run_path, effective)
    write_per_epoch_profile_card(
        profile_card_path,
        run_card_path=run_path,
        method=method,
        seed=seed,
        source_config_sha256=source_config_sha256,
        history=history,
    )
    return effective


def atomic_torch_save(path: Path | str, payload: Any) -> None:
    """Atomically save a PyTorch object."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix=".pt")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def capture_rng_state() -> Dict[str, Any]:
    """Capture RNG state needed for deterministic checkpoint resume."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore an RNG state emitted by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_payload(
    *,
    model: nn.Module,
    method: str,
    task: str,
    config: Mapping[str, Any],
    normalization: Mapping[str, Any],
    epoch: int,
    selected_threshold: float,
    validation_dice: float,
    parameter_count: Mapping[str, int],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    best_validation_dice: Optional[float] = None,
    best_epoch: Optional[int] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a portable checkpoint; optimizer state is included only for last.pt."""
    queue_batch = queue_batch_metadata(config)
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "method": method,
        "task": task,
        "config": dict(config),
        "config_sha256": config_digest(config),
        "resume_config_sha256": resume_config_digest(config),
        "queue_batch_contract_sha256": queue_batch["contract_sha256"],
        "queue_batch_active_state": queue_batch["active"],
        "normalization": dict(normalization),
        "epoch": int(epoch),
        "selected_threshold": float(selected_threshold),
        "validation_dice": float(validation_dice),
        "parameter_count": dict(parameter_count),
        "model_state": model.state_dict(),
    }
    if provenance is not None:
        payload["provenance"] = dict(provenance)
    if optimizer is not None:
        payload.update(
            {
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "best_validation_dice": float(best_validation_dice)
                if best_validation_dice is not None
                else None,
                "best_epoch": int(best_epoch) if best_epoch is not None else None,
                "rng_state": capture_rng_state(),
            }
        )
    return payload
