"""Fail-closed backend for the restricted benchmark review application.

The UI intentionally knows nothing about storage.  This module is the only
production bridge: it opens artifacts listed in one SHA-256-pinned receipt,
validates their bytes before parsing, and returns in-memory arrays.  Paths are
relative to an environment-selected restricted root; no deployment location is
compiled into the application.

The receipt format is documented in ``docs/PRIVATE_DEMO_RUNBOOK.md``.  It is
deliberately narrow.  Adding an artifact to a directory does not make it
readable: the file must have a receipt entry and every catalogue reference must
name that entry.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import dataclass
import hashlib
from io import BytesIO, StringIO
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Any, BinaryIO, Dict, Iterable, Iterator, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from demo.backend import (
    ComparisonRow,
    DemoBackend,
    Example,
    HullExample,
    LoadedExample,
    LoadedHullExample,
    Prediction,
)
from segbench import features as block_features
from segbench.metrics import image_metrics
from segbench.postprocess import postprocess_blocks


RECEIPT_KIND = "segbench.restricted-demo-receipt.v1"
RECEIPT_ENV = "DEMO_ARTIFACT_RECEIPT"
RECEIPT_SHA_ENV = "DEMO_ARTIFACT_RECEIPT_SHA256"
ROOT_ENV = "DEMO_RESTRICTED_ROOT"
DEVICE_ENV = "DEMO_INFERENCE_DEVICE"
MEMORY_FRACTION_ENV = "DEMO_CUDA_MEMORY_FRACTION"
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
MAX_TEXT_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 100_000_000
MAX_CUDA_MEMORY_FRACTION = 0.1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_ALLOWED_KINDS = {
    "csv",
    "json",
    "yaml",
    "image",
    "binary_mask",
    "probability_npy",
    "probability_png",
    "torch_checkpoint",
    "joblib_model",
}


class ReceiptError(ValueError):
    """A receipt or one of its authenticated artifacts is invalid."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_identity(image: np.ndarray, ppi: float) -> str:
    value = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(f"{float(ppi):.9f}".encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes())
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReceiptError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ReceiptError(f"{label} must be an array")
    return value


def _safe_key(value: Any, label: str) -> str:
    key = str(value)
    if not _KEY.fullmatch(key):
        raise ReceiptError(f"{label} is not a safe key")
    return key


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReceiptError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ReceiptError(f"{label} is outside its allowed range")
    return number


@dataclass(frozen=True)
class Artifact:
    key: str
    relative_path: Path
    sha256: str
    byte_count: int
    kind: str


class ArtifactStore:
    """Open only regular, non-symlink files whose bytes match the receipt."""

    def __init__(self, root: Path, receipt_path: Path, expected_sha256: str) -> None:
        if not _SHA256.fullmatch(expected_sha256):
            raise ReceiptError(f"{RECEIPT_SHA_ENV} must be a lowercase SHA-256")
        if root.is_symlink() or not root.is_dir():
            raise ReceiptError(f"{ROOT_ENV} must name a regular directory")
        self.root = root.resolve(strict=True)
        receipt_absolute = receipt_path.absolute()
        try:
            receipt_input_relative = receipt_absolute.relative_to(self.root)
        except ValueError as exc:
            raise ReceiptError("the receipt must be inside the restricted root") from exc
        if ".." in receipt_input_relative.parts:
            raise ReceiptError("the receipt path must not contain parent traversal")
        receipt_cursor = self.root
        for part in receipt_input_relative.parts:
            receipt_cursor = receipt_cursor / part
            if receipt_cursor.is_symlink():
                raise ReceiptError("the receipt must not traverse a symlink")
        receipt = receipt_path.resolve(strict=True)
        try:
            receipt.relative_to(self.root)
        except ValueError as exc:
            raise ReceiptError("the receipt must be inside the restricted root") from exc
        if not receipt.is_file():
            raise ReceiptError("the receipt must be a regular non-symlink file")
        data = receipt.read_bytes()
        if not data or len(data) > MAX_RECEIPT_BYTES:
            raise ReceiptError("the receipt is empty or too large")
        if _sha256_bytes(data) != expected_sha256:
            raise ReceiptError("receipt SHA-256 differs from the pinned environment value")
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReceiptError("receipt is not valid UTF-8 JSON") from exc
        self.payload = _mapping(payload, "receipt")
        if self.payload.get("schema_version") != 1 or self.payload.get("kind") != RECEIPT_KIND:
            raise ReceiptError("unsupported restricted-demo receipt schema")
        entries = _mapping(self.payload.get("artifacts"), "receipt.artifacts")
        artifacts: Dict[str, Artifact] = {}
        for raw_key, raw_entry in entries.items():
            key = _safe_key(raw_key, "artifact key")
            entry = _mapping(raw_entry, f"artifact {key}")
            path_text = entry.get("path")
            if not isinstance(path_text, str) or not path_text:
                raise ReceiptError(f"artifact {key} has no relative path")
            relative = Path(path_text)
            if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
                raise ReceiptError(f"artifact {key} has an unsafe path")
            sha256 = str(entry.get("sha256", ""))
            if not _SHA256.fullmatch(sha256):
                raise ReceiptError(f"artifact {key} has an invalid SHA-256")
            byte_count = entry.get("bytes")
            if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
                raise ReceiptError(f"artifact {key} has an invalid byte count")
            kind = str(entry.get("kind", ""))
            if kind not in _ALLOWED_KINDS:
                raise ReceiptError(f"artifact {key} has an unsupported kind")
            artifacts[key] = Artifact(key, relative, sha256, byte_count, kind)
        if not artifacts:
            raise ReceiptError("receipt has no artifacts")
        self.artifacts = artifacts

    def artifact(self, key: str, allowed_kinds: Iterable[str]) -> Artifact:
        safe = _safe_key(key, "artifact reference")
        try:
            artifact = self.artifacts[safe]
        except KeyError as exc:
            raise ReceiptError(f"receipt does not authenticate artifact {safe!r}") from exc
        allowed = set(allowed_kinds)
        if artifact.kind not in allowed:
            raise ReceiptError(
                f"artifact {safe!r} has kind {artifact.kind!r}, expected {sorted(allowed)}"
            )
        return artifact

    def _path(self, artifact: Artifact) -> Path:
        path = self.root / artifact.relative_path
        current = self.root
        for part in artifact.relative_path.parts:
            current = current / part
            if current.is_symlink():
                raise ReceiptError(f"artifact {artifact.key!r} traverses a symlink")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self.root)
        except (OSError, ValueError) as exc:
            raise ReceiptError(f"artifact {artifact.key!r} is absent or escapes the root") from exc
        return resolved

    @contextmanager
    def open_verified(
        self, key: str, allowed_kinds: Iterable[str], *, max_bytes: int | None = None
    ) -> Iterator[BinaryIO]:
        artifact = self.artifact(key, allowed_kinds)
        if max_bytes is not None and artifact.byte_count > max_bytes:
            raise ReceiptError(f"artifact {key!r} exceeds its parser size limit")
        path = self._path(artifact)
        handle = path.open("rb")
        try:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != artifact.byte_count:
                raise ReceiptError(f"artifact {key!r} size/type differs from its receipt")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != artifact.sha256:
                raise ReceiptError(f"artifact {key!r} SHA-256 differs from its receipt")
            handle.seek(0)
            yield handle
        finally:
            handle.close()

    def bytes(
        self, key: str, allowed_kinds: Iterable[str], *, max_bytes: int | None = None
    ) -> bytes:
        with self.open_verified(key, allowed_kinds, max_bytes=max_bytes) as handle:
            return handle.read()

    def text(self, key: str, allowed_kinds: Iterable[str]) -> str:
        try:
            return self.bytes(key, allowed_kinds, max_bytes=MAX_TEXT_BYTES).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReceiptError(f"artifact {key!r} is not UTF-8") from exc


@dataclass(frozen=True)
class MethodSpec:
    key: str
    display_name: str
    threshold: float
    runner: Mapping[str, Any]


@dataclass(frozen=True)
class TestRecord:
    key: str
    label: str
    image_id: str
    examiner_value: str
    ppi: float
    image_artifact: str
    ground_truth_artifact: str


@dataclass(frozen=True)
class HullRecord:
    key: str
    label: str
    image_id: str
    group: str
    device: str
    capture: str
    ppi: float
    image_artifact: str
    hull_artifact: str
    minutiae_artifact: str


@dataclass(frozen=True)
class PredictionSpec:
    dataset: str
    example_key: str
    method_key: str
    artifact: str
    artifact_kind: str
    representation: str
    block_size: int
    min_component_blocks: int


def _csv_rows(store: ArtifactStore, key: str) -> list[dict[str, str]]:
    try:
        reader = csv.DictReader(StringIO(store.text(key, {"csv"})))
        if not reader.fieldnames:
            raise ReceiptError(f"CSV artifact {key!r} has no header")
        return [dict(row) for row in reader]
    except csv.Error as exc:
        raise ReceiptError(f"CSV artifact {key!r} is malformed") from exc


def _require_columns(rows: Sequence[Mapping[str, str]], required: set[str], label: str) -> None:
    if not rows:
        raise ReceiptError(f"{label} is empty")
    missing = required - set(rows[0])
    if missing:
        raise ReceiptError(f"{label} is missing columns: {sorted(missing)}")


def _read_image_bytes(value: bytes, label: str) -> np.ndarray:
    try:
        with Image.open(BytesIO(value)) as decoded:
            if decoded.format not in {"PNG", "JPEG", "WEBP"}:
                raise ReceiptError(f"{label} has an unsupported image format")
            width, height = decoded.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ReceiptError(f"{label} exceeds the decoded-pixel limit")
            decoded.load()
            image = np.asarray(decoded.convert("L"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ReceiptError(f"{label} is not an allowed grayscale image") from exc
    return np.ascontiguousarray(image)


def _resize_image(image: np.ndarray, source_ppi: float, target_ppi: float) -> np.ndarray:
    if math.isclose(source_ppi, target_ppi, abs_tol=1e-9, rel_tol=0.0):
        return image
    scale = target_ppi / source_ppi
    height, width = image.shape
    shape = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, shape, interpolation=interpolation)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    return cv2.resize(
        mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
    ).astype(bool)


class _DeepRunner:
    def __init__(
        self,
        store: ArtifactStore,
        method: MethodSpec,
        runner: Mapping[str, Any],
        device_name: str,
        memory_fraction: float,
    ) -> None:
        self.store = store
        self.method = method
        self.runner = runner
        self.device_name = device_name
        self.memory_fraction = memory_fraction
        self._loaded: tuple[Any, Mapping[str, Any], Mapping[str, float], Any] | None = None

    def _load(self) -> tuple[Any, Mapping[str, Any], Mapping[str, float], Any]:
        if self._loaded is not None:
            return self._loaded
        import torch
        import yaml

        from segbench.deep_models import build_model
        from segbench.deep_utils import configure_memory_fraction

        config_key = _safe_key(self.runner.get("config_artifact"), "deep config artifact")
        config = yaml.safe_load(self.store.text(config_key, {"yaml", "json"}))
        config = _mapping(config, "deep config")
        if config.get("method") != self.method.key or config.get("task") not in {"pixel", "patch"}:
            raise ReceiptError(f"deep configuration disagrees with method {self.method.key}")
        checkpoint_key = _safe_key(
            self.runner.get("model_artifact"), "deep checkpoint artifact"
        )
        with self.store.open_verified(checkpoint_key, {"torch_checkpoint"}) as handle:
            try:
                checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise ReceiptError("deep checkpoint cannot be loaded in weights-only mode") from exc
        checkpoint = _mapping(checkpoint, "deep checkpoint")
        if checkpoint.get("method") != self.method.key or checkpoint.get("task") != config["task"]:
            raise ReceiptError("deep checkpoint identity disagrees with its configuration")
        checkpoint_config = checkpoint.get("config")
        if not isinstance(checkpoint_config, Mapping) or dict(checkpoint_config) != dict(config):
            raise ReceiptError("deep checkpoint embeds a different configuration")
        normalization = _mapping(checkpoint.get("normalization"), "checkpoint normalization")
        mean = _finite(normalization.get("mean"), "normalization.mean")
        std = _finite(normalization.get("std"), "normalization.std", minimum=1e-8)
        if std <= 0.0:
            raise ReceiptError("normalization.std must be positive")
        model_config = dict(_mapping(config.get("model"), "model configuration"))
        # A complete checkpoint already contains the initialized SegFormer
        # tensors; rebuilding must never fetch or open an unreceipted snapshot.
        if self.method.key == "segformer_sub":
            model_config["pretrained"] = False
        model = build_model(self.method.key, model_config)
        try:
            model.load_state_dict(checkpoint["model_state"], strict=True)
        except (KeyError, RuntimeError) as exc:
            raise ReceiptError("deep checkpoint state does not match the declared model") from exc
        device = torch.device(self.device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ReceiptError("CUDA inference was requested but is unavailable")
        configure_memory_fraction(device, self.memory_fraction)
        model.to(device).eval()
        self._loaded = model, config, {"mean": mean, "std": std}, device
        return self._loaded

    def predict(self, image: np.ndarray, ppi: float, threshold: float) -> tuple[np.ndarray, float]:
        from segbench.deep_utils import (
            blocks_to_postprocessed_mask,
            predict_patch_probability,
            predict_pixel_probability,
        )

        model, config, normalization, device = self._load()
        target_ppi = _finite(
            _mapping(config.get("data"), "data configuration").get("target_ppi", 500),
            "data.target_ppi",
            minimum=1e-8,
        )
        target = _resize_image(image, ppi, target_ppi).astype(np.float32) / 255.0
        normalized = (target - normalization["mean"]) / normalization["std"]
        inference = _mapping(config.get("inference"), "inference configuration")
        acceleration = _mapping(
            _mapping(config.get("runtime"), "runtime configuration").get(
                "acceleration", {"precision": "fp32", "channels_last": False}
            ),
            "runtime.acceleration",
        )
        started = time.perf_counter()
        if config["task"] == "pixel":
            probability = predict_pixel_probability(
                model,
                normalized,
                device,
                pad_multiple=int(inference["pad_multiple"]),
                tile_size=int(inference["tile_size"]),
                tile_overlap=int(inference["tile_overlap"]),
                autocast_bf16=acceleration.get("precision") == "bf16",
                channels_last=bool(acceleration.get("channels_last", False)),
            )
            mask = probability >= threshold
        else:
            data = _mapping(config["data"], "data configuration")
            probability = predict_patch_probability(
                model,
                normalized,
                device,
                patch_size=int(data["patch_size"]),
                block_size=int(data["block_size"]),
                batch_size=int(inference["patch_batch_size"]),
                autocast_bf16=acceleration.get("precision") == "bf16",
                channels_last=bool(acceleration.get("channels_last", False)),
            )
            mask = blocks_to_postprocessed_mask(
                probability,
                target.shape,
                threshold=threshold,
                min_component_blocks=int(inference["min_component_blocks"]),
            )
        mask = _resize_mask(mask, image.shape)
        return mask, (time.perf_counter() - started) * 1000.0


class _ClassicalRunner:
    def __init__(
        self, store: ArtifactStore, method: MethodSpec, runner: Mapping[str, Any]
    ) -> None:
        self.store = store
        self.method = method
        self.runner = runner
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        import joblib
        from segbench.models import set_single_thread

        key = _safe_key(self.runner.get("model_artifact"), "classical model artifact")
        with self.store.open_verified(key, {"joblib_model"}) as handle:
            try:
                model = joblib.load(handle)
            except Exception as exc:
                raise ReceiptError("classical model could not be loaded") from exc
        self._model = set_single_thread(model)
        return self._model

    def predict(self, image: np.ndarray, ppi: float, threshold: float) -> tuple[np.ndarray, float]:
        del threshold
        from threadpoolctl import threadpool_limits

        target_ppi = _finite(self.runner.get("target_ppi", 500), "target_ppi", minimum=1e-8)
        target = _resize_image(image, ppi, target_ppi)
        started = time.perf_counter()
        features = block_features.extract_block_features(target)
        grid_h, grid_w, n_features = features.shape
        with threadpool_limits(limits=1):
            blocks = np.asarray(
                self._load().predict(features.reshape(-1, n_features))
            ).astype(bool).reshape(grid_h, grid_w)
        blocks = postprocess_blocks(
            blocks, int(self.runner.get("min_component_blocks", 4))
        )
        mask = block_features.blocks_to_mask(blocks, target.shape)
        mask = _resize_mask(mask, image.shape)
        return mask, (time.perf_counter() - started) * 1000.0


class RestrictedDemoBackend:
    """Receipt-authenticated production implementation of :class:`DemoBackend`."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        device_name: str = "cpu",
        memory_fraction: float = MAX_CUDA_MEMORY_FRACTION,
    ) -> None:
        self.store = store
        self.receipt = store.payload
        self.target_ppi = _finite(
            self.receipt.get("target_ppi"), "receipt.target_ppi", minimum=1e-8
        )
        if device_name not in {"cpu", "cuda"}:
            raise ReceiptError(f"{DEVICE_ENV} must be cpu or cuda")
        if not 0.0 < memory_fraction <= MAX_CUDA_MEMORY_FRACTION:
            raise ReceiptError(
                f"{MEMORY_FRACTION_ENV} must be in (0, {MAX_CUDA_MEMORY_FRACTION}]"
            )
        self.methods = self._load_methods(device_name, memory_fraction)
        datasets = _mapping(self.receipt.get("datasets"), "receipt.datasets")
        self.test_records, self.test_results = self._load_test_dataset(
            _mapping(datasets.get("test"), "datasets.test")
        )
        self.hull_records, self.hull_results = self._load_hull_dataset(
            _mapping(datasets.get("b1"), "datasets.b1")
        )
        self.predictions: Dict[tuple[str, str, str], PredictionSpec] = {}
        for dataset, section in (("test", datasets["test"]), ("b1", datasets["b1"])):
            self._load_prediction_index(dataset, _mapping(section, f"datasets.{dataset}"))
        self._known_images: Dict[str, tuple[str, str]] = {}
        self._register_known_images()

    def _load_methods(
        self, device_name: str, memory_fraction: float
    ) -> Dict[str, tuple[MethodSpec, Any | None]]:
        output: Dict[str, tuple[MethodSpec, Any | None]] = {}
        displays: set[str] = set()
        for index, raw in enumerate(_sequence(self.receipt.get("methods"), "methods")):
            item = _mapping(raw, f"methods[{index}]")
            key = _safe_key(item.get("key"), "method key")
            display = str(item.get("display_name", "")).strip()
            if not display or display in displays or key in output:
                raise ReceiptError("method keys and display names must be unique and non-empty")
            threshold = _finite(item.get("reported_threshold"), "reported threshold")
            if not 0.0 <= threshold <= 1.0:
                raise ReceiptError("reported threshold must lie in [0,1]")
            runner_spec = _mapping(item.get("runner"), f"method {key} runner")
            runner_type = runner_spec.get("type")
            method = MethodSpec(key, display, threshold, runner_spec)
            if runner_type == "deep_checkpoint":
                runner: Any | None = _DeepRunner(
                    self.store, method, runner_spec, device_name, memory_fraction
                )
            elif runner_type == "classical_joblib":
                runner = _ClassicalRunner(self.store, method, runner_spec)
            elif runner_type == "precomputed_only":
                runner = None
            else:
                raise ReceiptError(f"method {key} has an unsupported runner type")
            output[key] = (method, runner)
            displays.add(display)
        if not output:
            raise ReceiptError("receipt has no methods")
        return output

    def _load_test_dataset(
        self, section: Mapping[str, Any]
    ) -> tuple[Dict[str, TestRecord], Dict[tuple[str, str], Mapping[str, str]]]:
        rows = _csv_rows(self.store, _safe_key(section.get("manifest"), "test manifest"))
        required = {
            "example_key", "label", "image_id", "examiner_value", "ppi",
            "image_artifact", "ground_truth_artifact", "split",
        }
        _require_columns(rows, required, "test manifest")
        records: Dict[str, TestRecord] = {}
        for row in rows:
            if row["split"] != "test":
                raise ReceiptError("restricted test manifest may contain only test rows")
            key = _safe_key(row["example_key"], "test example key")
            if key in records:
                raise ReceiptError("test manifest has duplicate example keys")
            records[key] = TestRecord(
                key,
                row["label"].strip() or key,
                row["image_id"],
                row["examiner_value"].strip().upper(),
                _finite(row["ppi"], "test PPI", minimum=1e-8),
                _safe_key(row["image_artifact"], "test image artifact"),
                _safe_key(row["ground_truth_artifact"], "test ground-truth artifact"),
            )
        results = self._load_results(section, "test", records)
        return records, results

    def _load_hull_dataset(
        self, section: Mapping[str, Any]
    ) -> tuple[Dict[str, HullRecord], Dict[tuple[str, str], Mapping[str, str]]]:
        rows = _csv_rows(self.store, _safe_key(section.get("manifest"), "B1 manifest"))
        required = {
            "example_key", "label", "image_id", "group", "device", "capture",
            "ppi", "image_artifact", "hull_artifact", "minutiae_artifact", "split",
        }
        _require_columns(rows, required, "B1 manifest")
        records: Dict[str, HullRecord] = {}
        for row in rows:
            if row["split"] != "test":
                raise ReceiptError("restricted B1 manifest may contain only test rows")
            key = _safe_key(row["example_key"], "B1 example key")
            if key in records:
                raise ReceiptError("B1 manifest has duplicate example keys")
            records[key] = HullRecord(
                key,
                row["label"].strip() or key,
                row["image_id"],
                row["group"],
                row["device"],
                row["capture"],
                _finite(row["ppi"], "B1 PPI", minimum=1e-8),
                _safe_key(row["image_artifact"], "B1 image artifact"),
                _safe_key(row["hull_artifact"], "B1 hull artifact"),
                _safe_key(row["minutiae_artifact"], "B1 minutiae artifact"),
            )
        results = self._load_results(section, "b1", records)
        return records, results

    def _load_results(
        self,
        section: Mapping[str, Any],
        dataset: str,
        records: Mapping[str, Any],
    ) -> Dict[tuple[str, str], Mapping[str, str]]:
        rows = _csv_rows(self.store, _safe_key(section.get("results"), f"{dataset} results"))
        metric_columns = (
            {"dice", "mdr", "fdr"} if dataset == "test" else {"mc", "hr", "par", "ohf"}
        )
        required = {"example_key", "method", "ms_per_img", *metric_columns}
        _require_columns(rows, required, f"{dataset} results")
        output: Dict[tuple[str, str], Mapping[str, str]] = {}
        for row in rows:
            key = _safe_key(row["example_key"], "result example key")
            method = _safe_key(row["method"], "result method")
            if key not in records or method not in self.methods:
                raise ReceiptError(f"{dataset} results reference an unknown key or method")
            pair = (key, method)
            if pair in output:
                raise ReceiptError(f"{dataset} results contain duplicate rows")
            _finite(row["ms_per_img"], "result runtime", minimum=0.0)
            for column in metric_columns:
                value = _finite(row[column], f"result {column}", minimum=0.0)
                if column != "par" and value > 1.0:
                    raise ReceiptError(f"result {column} must lie in [0,1]")
            output[pair] = row
        expected = {(key, method) for key in records for method in self.methods}
        missing = expected - set(output)
        extra = set(output) - expected
        if missing or extra:
            raise ReceiptError(
                f"{dataset} results do not cover the complete example/method grid "
                f"(missing={len(missing)}, extra={len(extra)})"
            )
        return output

    def _load_prediction_index(self, dataset: str, section: Mapping[str, Any]) -> None:
        rows = _csv_rows(
            self.store, _safe_key(section.get("predictions"), f"{dataset} prediction index")
        )
        required = {
            "example_key", "method", "artifact", "artifact_kind", "representation",
            "block_size", "min_component_blocks",
        }
        _require_columns(rows, required, f"{dataset} prediction index")
        examples = self.test_records if dataset == "test" else self.hull_records
        for row in rows:
            example = _safe_key(row["example_key"], "prediction example key")
            method = _safe_key(row["method"], "prediction method")
            if example not in examples or method not in self.methods:
                raise ReceiptError("prediction index references an unknown example or method")
            artifact = _safe_key(row["artifact"], "prediction artifact")
            kind = row["artifact_kind"]
            representation = row["representation"]
            if kind not in {"probability_npy", "probability_png", "binary_mask"}:
                raise ReceiptError("prediction artifact kind is unsupported")
            if representation not in {"pixel", "block", "mask"}:
                raise ReceiptError("prediction representation is unsupported")
            if (kind == "binary_mask") != (representation == "mask"):
                raise ReceiptError("binary mask kind and mask representation must agree")
            self.store.artifact(artifact, {kind})
            pair = (dataset, example, method)
            if pair in self.predictions:
                raise ReceiptError("prediction index contains duplicate rows")
            self.predictions[pair] = PredictionSpec(
                dataset,
                example,
                method,
                artifact,
                kind,
                representation,
                int(row["block_size"] or block_features.BLOCK),
                int(row["min_component_blocks"] or 4),
            )
        expected = {
            (dataset, example, method)
            for example in examples
            for method in self.methods
            if (example, method)
            in (self.test_results if dataset == "test" else self.hull_results)
        }
        missing = expected - set(self.predictions)
        if missing:
            raise ReceiptError(f"prediction index is incomplete ({len(missing)} rows missing)")

    def _image(self, artifact: str) -> np.ndarray:
        value = self.store.bytes(artifact, {"image"})
        return _read_image_bytes(value, artifact)

    def _mask(self, artifact: str) -> np.ndarray:
        value = _read_image_bytes(
            self.store.bytes(artifact, {"binary_mask"}), artifact
        )
        unique = set(np.unique(value).tolist())
        if not unique.issubset({0, 255}):
            raise ReceiptError(f"binary mask {artifact!r} has non-binary values")
        return value > 0

    def _probability(self, spec: PredictionSpec) -> np.ndarray:
        if spec.artifact_kind == "probability_npy":
            value = self.store.bytes(spec.artifact, {"probability_npy"})
            try:
                probability = np.load(BytesIO(value), allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise ReceiptError("probability NPY is malformed") from exc
            if probability.dtype not in {np.dtype(np.float16), np.dtype(np.float32)}:
                raise ReceiptError("probability NPY must be float16 or float32")
            probability = probability.astype(np.float32)
        else:
            value = _read_image_bytes(
                self.store.bytes(spec.artifact, {"probability_png"}), spec.artifact
            )
            probability = value.astype(np.float32) / 255.0
        if (
            probability.ndim != 2
            or probability.size == 0
            or probability.size > MAX_IMAGE_PIXELS
            or not np.isfinite(probability).all()
            or bool((probability < 0.0).any())
            or bool((probability > 1.0).any())
        ):
            raise ReceiptError("probability artifact has invalid shape or values")
        return probability

    def _minutiae(self, artifact: str, source_shape: tuple[int, int]) -> np.ndarray:
        rows = _csv_rows(self.store, artifact)
        _require_columns(rows, {"x_px", "y_px"}, "minutiae")
        points = np.asarray(
            [[float(row["x_px"]), float(row["y_px"])] for row in rows],
            dtype=np.float64,
        )
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3:
            raise ReceiptError("minutiae must contain at least three points")
        if not np.isfinite(points).all():
            raise ReceiptError("minutiae contain non-finite coordinates")
        height, width = source_shape
        if (
            (points[:, 0] < 0).any()
            or (points[:, 0] >= width).any()
            or (points[:, 1] < 0).any()
            or (points[:, 1] >= height).any()
        ):
            raise ReceiptError("minutiae lie outside their source image")
        return points

    def _register_known_images(self) -> None:
        for key in self.test_records:
            loaded = self.load_example(key)
            identity = _array_identity(loaded.image, loaded.ppi)
            if identity in self._known_images:
                raise ReceiptError("two test/B1 examples have indistinguishable source arrays")
            self._known_images[identity] = ("test", key)
        for key in self.hull_records:
            loaded = self.load_hull_example(key)
            identity = _array_identity(loaded.image, loaded.ppi)
            if identity in self._known_images:
                raise ReceiptError("two test/B1 examples have indistinguishable source arrays")
            self._known_images[identity] = ("b1", key)

    def method_names(self) -> Sequence[str]:
        return [spec.display_name for spec, _ in self.methods.values()]

    def upload_method_names(self) -> Sequence[str]:
        return [spec.display_name for spec, runner in self.methods.values() if runner is not None]

    def supports_threshold_exploration(self, method: str) -> bool:
        key = self._method_key(method)
        if self.methods[key][1] is not None and self.methods[key][0].runner.get("type") == "deep_checkpoint":
            return True
        return any(
            pair[2] == key and spec.representation in {"pixel", "block"}
            for pair, spec in self.predictions.items()
        )

    def examples(self) -> Sequence[Example]:
        return [Example(row.key, row.label, row.examiner_value) for row in self.test_records.values()]

    def hull_examples(self) -> Sequence[HullExample]:
        return [
            HullExample(row.key, row.label, row.group, row.device, row.capture)
            for row in self.hull_records.values()
        ]

    def load_example(self, key: str) -> LoadedExample:
        try:
            row = self.test_records[key]
        except KeyError as exc:
            raise KeyError("unknown private example") from exc
        source = self._image(row.image_artifact)
        ground_truth = self._mask(row.ground_truth_artifact)
        if ground_truth.shape != source.shape:
            raise ReceiptError("test image and ground truth have different native shapes")
        target = _resize_image(source, row.ppi, self.target_ppi)
        ground_truth = _resize_mask(ground_truth, target.shape)
        return LoadedExample(target, self.target_ppi, ground_truth)

    def load_hull_example(self, key: str) -> LoadedHullExample:
        try:
            row = self.hull_records[key]
        except KeyError as exc:
            raise KeyError("unknown private hull example") from exc
        source = self._image(row.image_artifact)
        hull = self._mask(row.hull_artifact)
        if hull.shape != source.shape:
            raise ReceiptError("B1 image and hull have different native shapes")
        points = self._minutiae(row.minutiae_artifact, source.shape)
        target = _resize_image(source, row.ppi, self.target_ppi)
        scale = self.target_ppi / row.ppi
        points = points * scale
        hull = _resize_mask(hull, target.shape)
        return LoadedHullExample(target, self.target_ppi, points, hull)

    def _method_key(self, display_name: str) -> str:
        matches = [key for key, (spec, _) in self.methods.items() if spec.display_name == display_name]
        if len(matches) != 1:
            raise KeyError("unknown method")
        return matches[0]

    def reported_threshold(self, method: str) -> float:
        return self.methods[self._method_key(method)][0].threshold

    def _precomputed_mask(
        self,
        dataset: str,
        key: str,
        method: str,
        shape: tuple[int, int],
        threshold: float,
    ) -> np.ndarray:
        try:
            spec = self.predictions[(dataset, key, method)]
        except KeyError as exc:
            raise ReceiptError("no authenticated precomputed prediction is available") from exc
        if spec.representation == "mask":
            mask = self._mask(spec.artifact)
            if not math.isclose(
                threshold, self.methods[method][0].threshold, abs_tol=1e-9, rel_tol=0.0
            ):
                raise ReceiptError("this method has no probability map for threshold exploration")
            return _resize_mask(mask, shape)
        probability = self._probability(spec)
        if spec.representation == "pixel":
            if probability.shape != shape:
                raise ReceiptError("pixel probability shape differs from its source panel")
            return probability >= threshold
        expected = block_features.grid_shape(*shape)
        if probability.shape != expected or spec.block_size != block_features.BLOCK:
            raise ReceiptError("block probability shape/grid differs from its source panel")
        blocks = postprocess_blocks(
            probability >= threshold, min_component_blocks=spec.min_component_blocks
        )
        return block_features.blocks_to_mask(blocks, shape).astype(bool)

    @staticmethod
    def _require_metric_match(
        recorded: Mapping[str, str], observed: Mapping[str, float], columns: Sequence[str]
    ) -> None:
        for column in columns:
            expected = _finite(recorded[column], f"recorded {column}", minimum=0.0)
            if not math.isclose(expected, float(observed[column]), abs_tol=5e-5, rel_tol=0.0):
                raise ReceiptError(
                    f"authenticated results disagree with the derived {column} panel metric"
                )

    def _validate_precomputed_result(
        self, dataset: str, example_key: str, method_key: str, mask: np.ndarray
    ) -> None:
        if dataset == "test":
            loaded = self.load_example(example_key)
            assert loaded.ground_truth is not None
            observed = image_metrics(mask, loaded.ground_truth)
            self._require_metric_match(
                self.test_results[(example_key, method_key)],
                observed,
                ("dice", "mdr", "fdr"),
            )
            return
        loaded_hull = self.load_hull_example(example_key)
        hull = loaded_hull.hull.astype(bool)
        points = np.rint(loaded_hull.minutiae_xy).astype(np.int64)
        predicted_area = int(mask.sum())
        hull_area = int(hull.sum())
        observed_hull = {
            "mc": float(mask[points[:, 1], points[:, 0]].mean()),
            "hr": float(np.count_nonzero(mask & hull) / hull_area),
            "par": float(predicted_area / hull_area),
            "ohf": (
                float(np.count_nonzero(mask & ~hull) / predicted_area)
                if predicted_area
                else 0.0
            ),
        }
        self._require_metric_match(
            self.hull_results[(example_key, method_key)],
            observed_hull,
            ("mc", "hr", "par", "ohf"),
        )

    def predict(
        self,
        method: str,
        image: np.ndarray,
        ppi: float,
        threshold: float | None = None,
    ) -> Prediction:
        key = self._method_key(method)
        spec, runner = self.methods[key]
        applied = spec.threshold if threshold is None else _finite(threshold, "threshold")
        if not 0.0 <= applied <= 1.0:
            raise ReceiptError("threshold must lie in [0,1]")
        source = np.asarray(image)
        if source.dtype != np.uint8 or source.ndim != 2 or source.size > MAX_IMAGE_PIXELS:
            raise ReceiptError("inference input must be an allowed uint8 grayscale image")
        source_ppi = _finite(ppi, "inference PPI", minimum=1e-8)
        known = self._known_images.get(_array_identity(source, source_ppi))
        if known is not None:
            dataset, example_key = known
            mask = self._precomputed_mask(dataset, example_key, key, source.shape, applied)
            results = self.test_results if dataset == "test" else self.hull_results
            if threshold is None or math.isclose(
                applied, spec.threshold, abs_tol=1e-9, rel_tol=0.0
            ):
                self._validate_precomputed_result(dataset, example_key, key, mask)
            elapsed = _finite(results[(example_key, key)]["ms_per_img"], "precomputed runtime")
            return Prediction(mask, elapsed, applied)
        if runner is None:
            raise ReceiptError(f"{method} is available for frozen panels only")
        mask, elapsed = runner.predict(source, source_ppi, applied)
        return Prediction(mask, elapsed, applied)

    def comparison_rows(
        self, method_a: str, method_b: str, limit: int = 20
    ) -> Sequence[ComparisonRow]:
        if not 1 <= int(limit) <= 100:
            raise ReceiptError("comparison limit must be between 1 and 100")
        key_a, key_b = self._method_key(method_a), self._method_key(method_b)
        if key_a == key_b:
            raise ReceiptError("comparison methods must differ")
        rows: list[ComparisonRow] = []
        for key, record in self.test_records.items():
            loaded = self.load_example(key)
            assert loaded.ground_truth is not None
            prediction_a = self.predict(method_a, loaded.image, loaded.ppi).mask
            prediction_b = self.predict(method_b, loaded.image, loaded.ppi).mask
            dice_a = float(image_metrics(prediction_a, loaded.ground_truth)["dice"])
            dice_b = float(image_metrics(prediction_b, loaded.ground_truth)["dice"])
            for method_key, observed in ((key_a, dice_a), (key_b, dice_b)):
                recorded = _finite(self.test_results[(key, method_key)]["dice"], "recorded Dice")
                if not math.isclose(recorded, observed, abs_tol=5e-5, rel_tol=0.0):
                    raise ReceiptError("authenticated results disagree with derived panel Dice")
            rows.append(
                ComparisonRow(key, record.label, record.examiner_value, dice_a, dice_b)
            )
        rows.sort(key=lambda row: (-abs(row.dice_a - row.dice_b), row.label, row.key))
        return rows[: int(limit)]


def create_backend(environment: Mapping[str, str] | None = None) -> DemoBackend:
    """Factory loaded by ``DEMO_BACKEND_FACTORY``.

    Three environment values are mandatory: restricted root, receipt path, and
    the independently pinned receipt SHA-256.  The device defaults to CPU;
    CUDA must be selected explicitly and remains subject to the allocator cap.
    """
    env = os.environ if environment is None else environment
    required = (ROOT_ENV, RECEIPT_ENV, RECEIPT_SHA_ENV)
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise ReceiptError("missing required backend settings: " + ", ".join(missing))
    root = Path(env[ROOT_ENV])
    receipt = Path(env[RECEIPT_ENV])
    device = env.get(DEVICE_ENV, "cpu").lower()
    memory_fraction = _finite(
        env.get(MEMORY_FRACTION_ENV, str(MAX_CUDA_MEMORY_FRACTION)),
        MEMORY_FRACTION_ENV,
        minimum=0.0,
    )
    store = ArtifactStore(root, receipt, env[RECEIPT_SHA_ENV])
    return RestrictedDemoBackend(
        store, device_name=device, memory_fraction=memory_fraction
    )
