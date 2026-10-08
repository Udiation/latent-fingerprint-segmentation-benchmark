"""Typed boundary between the private Gradio UI and restricted model loaders."""
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
    """One method prediction at the source image shape."""

    mask: np.ndarray
    milliseconds: float


@runtime_checkable
class DemoBackend(Protocol):
    """Restricted backend contract; implementations remain outside public pages."""

    def method_names(self) -> Sequence[str]:
        """Return display-safe method names in table order."""

    def examples(self) -> Sequence[Example]:
        """Return private test examples and examiner groups."""

    def load_example(self, key: str) -> LoadedExample:
        """Load one selected example into memory."""

    def predict(self, method: str, image: np.ndarray, ppi: float) -> Prediction:
        """Run a frozen method without persisting the input."""


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
    if not methods or len(methods) != len(set(methods)) or any(not item for item in methods):
        raise ValueError("backend method names must be unique and non-empty")
    if not examples or len({item.key for item in examples}) != len(examples):
        raise ValueError("backend examples must be non-empty with unique keys")
    return backend

