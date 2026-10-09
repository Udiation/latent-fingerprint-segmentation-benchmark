"""Typed boundary between the private Gradio UI and restricted model loaders.

This module contains contracts only. Dataset paths, model paths, identifiers,
predictions, and credentials belong to the restricted backend implementation
selected at runtime through ``DEMO_BACKEND_FACTORY``.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Example:
    """One private test example exposed by a backend."""

    key: str
    label: str
    examiner_value: str


@dataclass(frozen=True)
class LoadedExample:
    """In-memory source image and optional ground truth."""

    image: np.ndarray
    ppi: float
    ground_truth: np.ndarray | None


@dataclass(frozen=True)
class Prediction:
    """One thresholded method prediction at the source image shape."""

    mask: np.ndarray
    milliseconds: float
    applied_threshold: float


@dataclass(frozen=True)
class HullExample:
    """One private exemplar exposed without filesystem details."""

    key: str
    label: str
    group: str
    device: str
    capture_type: str


@dataclass(frozen=True)
class LoadedHullExample:
    """In-memory exemplar, expert minutiae, and lower-bound hull."""

    image: np.ndarray
    ppi: float
    minutiae_xy: np.ndarray
    hull: np.ndarray


@dataclass(frozen=True)
class ComparisonRow:
    """Private, precomputed per-example Dice comparison."""

    key: str
    label: str
    examiner_value: str
    dice_a: float
    dice_b: float


@runtime_checkable
class DemoBackend(Protocol):
    """Restricted backend contract; implementations remain outside public pages."""

    def method_names(self) -> Sequence[str]:
        """Return display-safe method names in table order."""

    def examples(self) -> Sequence[Example]:
        """Return private test examples and examiner groups."""

    def load_example(self, key: str) -> LoadedExample:
        """Load one selected example into memory."""

    def reported_threshold(self, method: str) -> float:
        """Return the validation-selected threshold reported in the tables."""

    def predict(
        self,
        method: str,
        image: np.ndarray,
        ppi: float,
        threshold: float | None = None,
    ) -> Prediction:
        """Run a frozen method without persisting the input.

        ``threshold=None`` means the reported validation-selected threshold.
        An explicit threshold is exploration-only and must reuse the same
        frozen model and post-processing path.
        """

    def comparison_rows(
        self, method_a: str, method_b: str, limit: int = 20
    ) -> Sequence[ComparisonRow]:
        """Return largest absolute Dice gaps from frozen test predictions."""

    def hull_examples(self) -> Sequence[HullExample]:
        """Return private SD302g exemplar choices."""

    def load_hull_example(self, key: str) -> LoadedHullExample:
        """Load one exemplar plus its expert minutiae and hull in memory."""


def load_backend(factory_spec: str) -> DemoBackend:
    """Load ``module:function`` and validate its returned backend."""
    if factory_spec.count(":") != 1:
        raise ValueError("DEMO_BACKEND_FACTORY must be module:function")
    module_name, function_name = factory_spec.split(":")
    if not module_name or not function_name or not all(
        part.isidentifier() for part in module_name.split(".")
    ) or not function_name.isidentifier():
        raise ValueError("DEMO_BACKEND_FACTORY is not a valid dotted entry point")
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise TypeError("DEMO_BACKEND_FACTORY does not resolve to a callable")
    backend = factory()
    if not isinstance(backend, DemoBackend):
        raise TypeError("backend does not implement the DemoBackend protocol")
    methods = list(backend.method_names())
    examples = list(backend.examples())
    hull_examples = list(backend.hull_examples())
    if not methods or len(methods) != len(set(methods)) or any(not item for item in methods):
        raise ValueError("backend method names must be unique and non-empty")
    if not examples or len({item.key for item in examples}) != len(examples):
        raise ValueError("backend examples must be non-empty with unique keys")
    if not hull_examples or len({item.key for item in hull_examples}) != len(hull_examples):
        raise ValueError("backend hull examples must be non-empty with unique keys")
    for method in methods:
        threshold = float(backend.reported_threshold(method))
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"backend reported threshold is invalid for {method}")
    upload_names = getattr(backend, "upload_method_names", None)
    if upload_names is not None:
        if not callable(upload_names):
            raise TypeError("backend upload_method_names is not callable")
        upload_methods = list(upload_names())
        if (
            not upload_methods
            or len(upload_methods) != len(set(upload_methods))
            or any(method not in methods for method in upload_methods)
        ):
            raise ValueError(
                "backend upload methods must be a non-empty unique subset of methods"
            )
    explorable = getattr(backend, "supports_threshold_exploration", None)
    if explorable is not None:
        if not callable(explorable):
            raise TypeError("backend supports_threshold_exploration is not callable")
        if any(not isinstance(explorable(method), bool) for method in methods):
            raise ValueError("backend threshold-exploration capabilities must be boolean")
    return backend
