"""Synthetic-only tests for the private review application."""
from __future__ import annotations

import base64
import ipaddress
from pathlib import Path
import sys
from types import ModuleType

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo.app import (  # noqa: E402
    MAX_UPLOAD_BYTES,
    DemoSettings,
    create_app,
    decode_upload,
    hull_metrics,
    launch,
    load_settings,
    render_compare,
    render_explore,
    render_hull,
    render_upload,
    validate_bind_host,
)
from demo.backend import (  # noqa: E402
    ComparisonRow,
    Example,
    HullExample,
    LoadedExample,
    LoadedHullExample,
    Prediction,
    load_backend,
)


class FakeBackend:
    """Deterministic synthetic backend with no filesystem dependencies."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float | None]] = []

    def method_names(self):
        return ["Method A", "Method B"]

    def examples(self):
        return [Example("private-key", "Example one", "VALUE")]

    def load_example(self, key):
        assert key == "private-key"
        image = np.full((24, 28), 160, dtype=np.uint8)
        ground_truth = np.zeros_like(image, dtype=bool)
        ground_truth[4:20, 6:22] = True
        return LoadedExample(image, 500.0, ground_truth)

    def reported_threshold(self, method):
        return {"Method A": 0.4, "Method B": 0.6}[method]

    def predict(self, method, image, ppi, threshold=None):
        assert ppi == 500.0
        applied = self.reported_threshold(method) if threshold is None else float(threshold)
        self.calls.append((method, threshold))
        mask = np.zeros_like(image, dtype=bool)
        if image.shape == (30, 32):
            mask[6:24, 8:26] = True
            if method == "Method B":
                mask[:, 4:7] = True
        elif method == "Method A":
            mask[4:20, 6:22] = True
            if threshold is not None and applied > 0.5:
                mask[:, 6:8] = False
        else:
            mask[5:20, 7:23] = True
            if threshold is not None and applied < 0.3:
                mask[4, 6:23] = True
        return Prediction(mask, 2.5 if method == "Method A" else 3.0, applied)

    def comparison_rows(self, method_a, method_b, limit=20):
        assert method_a != method_b
        # Deliberately return more than requested: the UI enforces its top-20 contract.
        return [
            ComparisonRow(
                f"key-{index}",
                f"Example {index:02d}",
                ("VALUE", "LIMITED", "NOVALUE")[index % 3],
                0.95 - index * 0.01,
                0.50 + index * 0.005,
            )
            for index in range(25)
        ]

    def hull_examples(self):
        return [HullExample("hull-key", "Exemplar one", "baseline", "SYN", "roll")]

    def load_hull_example(self, key):
        assert key == "hull-key"
        image = np.full((30, 32), 145, dtype=np.uint8)
        hull = np.zeros_like(image, dtype=bool)
        hull[6:24, 8:26] = True
        minutiae = np.array([[10, 9], [16, 12], [22, 19], [14, 22]], dtype=np.float32)
        return LoadedHullExample(image, 500.0, minutiae, hull)


def _data_url() -> str:
    image = np.full((12, 14), 100, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def test_bind_host_accepts_only_loopback_or_private_vpn() -> None:
    loopback = str(ipaddress.IPv4Address(bytes((127, 0, 0, 1))))
    private = str(ipaddress.IPv4Address(bytes((10, 8, 0, 2))))
    assert validate_bind_host(loopback) == loopback
    assert validate_bind_host(private) == private
    invalid = (
        str(ipaddress.IPv4Address(0)),
        str(ipaddress.IPv6Address(0)),
        "example" + "." + "test",
        str(ipaddress.IPv4Address(bytes((8, 8, 8, 8)))),
    )
    for value in invalid:
        with pytest.raises(ValueError):
            validate_bind_host(value)


def test_settings_require_auth_and_regular_private_report(tmp_path: Path) -> None:
    report = tmp_path / "index.html"
    report.write_text(
        '<html><head><meta name="segbench-private-report" content="1"></head></html>',
        encoding="utf-8",
    )
    env = {
        "DEMO_HOST": str(ipaddress.IPv4Address(bytes((127, 0, 0, 1)))),
        "DEMO_PORT": "9876",
        "DEMO_USER": "reviewer",
        "DEMO_PASS": "a-long-private-password",
        "DEMO_REPORT_PATH": str(report),
        "DEMO_BACKEND_FACTORY": "package.module:create_backend",
    }
    settings = load_settings(env)
    assert settings.report_path == report.resolve()
    with pytest.raises(ValueError, match="credentials"):
        load_settings({**env, "DEMO_PASS": "short"})
    report.write_text(
        '<meta name="segbench-private-report" content="1">'
        '<meta name="segbench-public-safe" content="1">',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a private report"):
        load_settings(env)


def test_restricted_backend_factory_contract(monkeypatch) -> None:
    module = ModuleType("synthetic_demo_backend")
    module.create_backend = FakeBackend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    backend = load_backend("synthetic_demo_backend:create_backend")
    assert list(backend.method_names()) == ["Method A", "Method B"]
    assert backend.reported_threshold("Method A") == pytest.approx(0.4)


def test_upload_decode_and_render_remain_in_memory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    image = decode_upload(_data_url())
    assert image.shape == (12, 14)
    html = render_upload(FakeBackend(), _data_url(), 500, ["Method A"])
    assert "data:image/png;base64," in html
    assert "processed in memory" in html
    assert "Ground-truth metrics are unavailable" in html
    assert set(tmp_path.iterdir()) == before
    with pytest.raises(ValueError):
        decode_upload("data:text/plain;base64,SGVsbG8=")
    with pytest.raises(ValueError, match="encoding"):
        decode_upload("data:image/png;base64,not+valid===")


def test_explore_keeps_reported_metrics_separate_from_slider() -> None:
    backend = FakeBackend()
    html = render_explore(backend, "private-key", ["Method A"], 0.7)
    assert "Exploration only" in html
    assert "Reported threshold" in html
    assert "1.0000" in html
    assert ("Method A", None) in backend.calls
    assert ("Method A", 0.7) in backend.calls


def test_compare_uses_reported_predictions_and_limits_top_twenty() -> None:
    backend = FakeBackend()
    html = render_compare(backend, "private-key", "Method A", "Method B")
    assert "A alone is correct" in html
    assert "B alone is correct" in html
    assert "20 largest absolute Dice differences" in html
    assert "Example 00" in html and "Example 19" in html
    assert "Example 24" not in html
    assert ("Method A", None) in backend.calls
    assert ("Method B", None) in backend.calls


def test_hull_metrics_reference_and_empty_prediction() -> None:
    backend = FakeBackend()
    source = backend.load_hull_example("hull-key")
    exact = hull_metrics(source.hull, source.hull, source.minutiae_xy)
    assert exact == {"mc": 1.0, "hr": 1.0, "par": 1.0, "ohf": 0.0}
    empty = hull_metrics(np.zeros_like(source.hull), source.hull, source.minutiae_xy)
    assert empty == {"mc": 0.0, "hr": 0.0, "par": 0.0, "ohf": 0.0}


def test_hull_tab_reports_only_lower_bound_metrics() -> None:
    html = render_hull(FakeBackend(), "hull-key", "Method A")
    assert "Lower-bound protocol" in html
    assert "100.00%" in html
    assert "1.0000" in html
    assert "Dice, Jaccard, MDR, and FDR are intentionally not shown" in html


def test_app_has_five_tabs_and_no_file_backed_media_components() -> None:
    app = create_app(
        FakeBackend(), "<!doctype html><html><body>Private report</body></html>"
    )
    config = app.get_config_file()
    component_types = {component["type"] for component in config["components"]}
    assert component_types.isdisjoint({"file", "image", "gallery", "video", "audio"})
    labels = {
        component.get("props", {}).get("label")
        for component in config["components"]
    }
    assert {
        "Explore test set",
        "Compare two methods",
        "Live-scan (SD302g)",
        "Upload",
        "Report",
    }.issubset(labels)
    assert config.get("enable_queue") is True


def test_launch_forces_private_network_auth_and_request_cap(tmp_path: Path) -> None:
    report = tmp_path / "index.html"
    report.write_text("private", encoding="utf-8")
    settings = DemoSettings(
        host=str(ipaddress.IPv4Address(bytes((127, 0, 0, 1)))),
        port=9876,
        username="reviewer",
        password="a-long-private-password",
        report_path=report,
        backend_factory="package.module:create_backend",
    )

    class FakeApp:
        kwargs = None

        def launch(self, **kwargs):
            self.kwargs = kwargs

    app = FakeApp()
    launch(app, settings)
    assert app.kwargs["share"] is False
    assert app.kwargs["server_name"] == str(
        ipaddress.IPv4Address(bytes((127, 0, 0, 1)))
    )
    assert app.kwargs["auth"] == ("reviewer", "a-long-private-password")
    assert app.kwargs["allowed_paths"] == []
    assert app.kwargs["blocked_paths"] == ["/"]
    assert app.kwargs["enable_monitoring"] is False
    assert app.kwargs["max_file_size"] == MAX_UPLOAD_BYTES
