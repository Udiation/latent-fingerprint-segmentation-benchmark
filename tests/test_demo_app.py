from __future__ import annotations

import base64
import ipaddress
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo.app import (  # noqa: E402
    DemoSettings, create_app, decode_upload, launch, load_settings, render_predictions,
    validate_bind_host,
)
from demo.backend import Example, LoadedExample, Prediction  # noqa: E402


class FakeBackend:
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

    def predict(self, method, image, ppi):
        assert ppi == 500.0
        mask = np.zeros_like(image, dtype=bool)
        mask[4:20, 6:22] = True
        return Prediction(mask, 2.5 if method == "Method A" else 3.0)


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
        str(ipaddress.IPv4Address(0)), str(ipaddress.IPv6Address(0)), "example" + "." + "test",
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
        "DEMO_PORT": "9876", "DEMO_USER": "reviewer",
        "DEMO_PASS": "a-long-private-password", "DEMO_REPORT_PATH": str(report),
        "DEMO_BACKEND_FACTORY": "package.module:create_backend",
    }
    settings = load_settings(env)
    assert settings.report_path == report.resolve()
    with pytest.raises(ValueError, match="credentials"):
        load_settings({**env, "DEMO_PASS": "short"})


def test_upload_decode_and_predictions_remain_in_memory(tmp_path: Path) -> None:
    before = set(tmp_path.iterdir())
    image = decode_upload(_data_url())
    assert image.shape == (12, 14)
    html = render_predictions(FakeBackend(), "In-memory upload", "", _data_url(), 500, ["Method A"])
    assert "data:image/png;base64," in html
    assert "Method A" in html
    assert set(tmp_path.iterdir()) == before
    with pytest.raises(ValueError):
        decode_upload("data:text/plain;base64,SGVsbG8=")


def test_private_example_reports_metrics() -> None:
    html = render_predictions(
        FakeBackend(), "Private test example", "private-key", "", 500, ["Method A", "Method B"]
    )
    assert html.count("1.0000") >= 2
    assert "2.5" in html and "3.0" in html


def test_app_has_no_file_or_image_components() -> None:
    app = create_app(FakeBackend(), "<!doctype html><html><body>Private report</body></html>")
    config = app.get_config_file()
    component_types = {component["type"] for component in config["components"]}
    assert component_types.isdisjoint({"file", "image", "gallery", "video", "audio"})
    assert config.get("enable_queue") is True


def test_launch_forces_private_network_and_no_share(tmp_path: Path) -> None:
    report = tmp_path / "index.html"
    report.write_text("private", encoding="utf-8")
    settings = DemoSettings(
        host=str(ipaddress.IPv4Address(bytes((127, 0, 0, 1)))),
        port=9876, username="reviewer",
        password="a-long-private-password", report_path=report,
        backend_factory="package.module:create_backend",
    )

    class FakeApp:
        kwargs = None

        def launch(self, **kwargs):
            self.kwargs = kwargs

    app = FakeApp()
    launch(app, settings)
    assert app.kwargs["share"] is False
    assert app.kwargs["server_name"] == str(ipaddress.IPv4Address(bytes((127, 0, 0, 1))))
    assert app.kwargs["auth"] == ("reviewer", "a-long-private-password")
    assert app.kwargs["allowed_paths"] == []
    assert app.kwargs["enable_monitoring"] is False
