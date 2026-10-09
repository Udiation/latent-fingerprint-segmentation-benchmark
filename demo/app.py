#!/usr/bin/env python3
"""Authenticated private review UI for the segmentation benchmark.

The UI contains no dataset, checkpoint, prediction, credential, or filesystem
location. A restricted runtime backend supplies those objects as in-memory
arrays. Browser uploads use a FileReader data URL and are never handed to a
Gradio file component or persisted by this application.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from html import escape
import ipaddress
from io import BytesIO
import os
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2
import gradio as gr
import numpy as np
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.backend import (  # noqa: E402
    ComparisonRow,
    DemoBackend,
    Example,
    LoadedExample,
    LoadedHullExample,
    Prediction,
    load_backend,
)
from segbench.metrics import image_metrics  # noqa: E402


MAX_UPLOAD_BYTES = 1024**3
MAX_UPLOAD_PIXELS = 100_000_000
# OpenCV drawing and encoding use BGR channel order.
_COLOR_GT = (224, 169, 38)
_COLOR_REPORTED = (117, 155, 33)
_COLOR_EXPLORE = (11, 158, 245)
_COLOR_HULL = (38, 38, 220)
_COLOR_MINUTIAE = (224, 130, 38)
_DATA_URL = re.compile(
    r"^data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=]+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DemoSettings:
    """Validated launch configuration sourced only from environment variables."""

    host: str
    port: int
    username: str
    password: str
    report_path: Path
    backend_factory: str


def validate_bind_host(value: str) -> str:
    """Accept only an IP-literal loopback or private/VPN address."""
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError("DEMO_HOST must be an IP literal on loopback or the private VPN") from exc
    allowed = address.is_loopback or address.is_private
    if address.is_unspecified or address.is_multicast or address.is_link_local or not allowed:
        raise ValueError("DEMO_HOST must be an IP literal on loopback or the private VPN")
    return value


def load_settings(environment: Mapping[str, str] | None = None) -> DemoSettings:
    """Load required settings without printing credentials."""
    env = os.environ if environment is None else environment
    required = (
        "DEMO_HOST",
        "DEMO_PORT",
        "DEMO_USER",
        "DEMO_PASS",
        "DEMO_REPORT_PATH",
        "DEMO_BACKEND_FACTORY",
    )
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise ValueError("missing required demo settings: " + ", ".join(missing))
    host = validate_bind_host(env["DEMO_HOST"])
    try:
        port = int(env["DEMO_PORT"])
    except ValueError as exc:
        raise ValueError("DEMO_PORT must be an integer") from exc
    if not 1024 <= port <= 65535:
        raise ValueError("DEMO_PORT must be between 1024 and 65535")
    username, password = env["DEMO_USER"], env["DEMO_PASS"]
    if len(username) < 3 or len(password) < 12 or username == password:
        raise ValueError("demo credentials do not meet the minimum safety policy")
    report = Path(env["DEMO_REPORT_PATH"])
    if report.is_symlink() or not report.is_file() or report.name != "index.html":
        raise ValueError("DEMO_REPORT_PATH must name a regular private index.html")
    try:
        report_head = report.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("DEMO_REPORT_PATH is not readable UTF-8 HTML") from exc
    if (
        '<meta name="segbench-private-report" content="1">' not in report_head
        or 'name="segbench-public-safe"' in report_head
    ):
        raise ValueError("DEMO_REPORT_PATH is not a private report produced by the builder")
    return DemoSettings(
        host,
        port,
        username,
        password,
        report.resolve(),
        env["DEMO_BACKEND_FACTORY"],
    )


def _finite(value: float) -> bool:
    return bool(np.isfinite(value))


def _normalise_ppi(value: float) -> float:
    ppi = float(value)
    if not _finite(ppi) or not 100.0 <= ppi <= 2000.0:
        raise ValueError("PPI must be between 100 and 2000.")
    return ppi


def _normalise_image(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim == 3 and image.shape[2] in (3, 4):
        code = cv2.COLOR_RGBA2GRAY if image.shape[2] == 4 else cv2.COLOR_RGB2GRAY
        image = cv2.cvtColor(image, code)
    if image.ndim != 2 or image.size == 0 or image.size > MAX_UPLOAD_PIXELS:
        raise ValueError("backend returned an invalid source image")
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise ValueError("backend returned a non-finite source image")
        if float(image.max()) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def decode_upload(data_url: str) -> np.ndarray:
    """Decode a browser data URL to memory without filesystem I/O."""
    match = _DATA_URL.fullmatch((data_url or "").strip())
    if not match:
        raise ValueError("Upload a PNG, JPEG, or WebP image.")
    encoded = match.group(2)
    maximum_encoded = 4 * ((MAX_UPLOAD_BYTES + 2) // 3)
    if len(encoded) > maximum_encoded:
        raise ValueError("The uploaded image exceeds the 1 GB limit.")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("The uploaded image encoding is invalid.") from exc
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("The uploaded image exceeds the 1 GB limit.")
    try:
        with Image.open(BytesIO(payload)) as decoded:
            if decoded.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("The uploaded image format is not allowed.")
            width, height = decoded.size
            if width <= 0 or height <= 0 or width * height > MAX_UPLOAD_PIXELS:
                raise ValueError("The uploaded image dimensions are too large.")
            decoded.load()
            image = np.asarray(decoded.convert("L"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded image is invalid or its dimensions are too large.") from exc
    return np.ascontiguousarray(image)


def _normalise_source(value: LoadedExample) -> tuple[np.ndarray, float, np.ndarray | None]:
    image = _normalise_image(value.image)
    ppi = _normalise_ppi(value.ppi)
    ground_truth = None if value.ground_truth is None else np.asarray(value.ground_truth).astype(bool)
    if ground_truth is not None and ground_truth.shape != image.shape:
        raise ValueError("backend ground truth shape differs from its image")
    return image, ppi, ground_truth


def _normalise_hull_source(
    value: LoadedHullExample,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    image = _normalise_image(value.image)
    ppi = _normalise_ppi(value.ppi)
    hull = np.asarray(value.hull).astype(bool)
    if hull.shape != image.shape or not hull.any():
        raise ValueError("backend returned an invalid or empty expert hull")
    minutiae = np.asarray(value.minutiae_xy, dtype=np.float64)
    if minutiae.ndim != 2 or minutiae.shape[1:] != (2,) or len(minutiae) < 3:
        raise ValueError("backend returned invalid expert minutiae")
    if not np.isfinite(minutiae).all():
        raise ValueError("backend returned non-finite expert minutiae")
    rounded = np.rint(minutiae).astype(np.int64)
    if (
        (rounded[:, 0] < 0).any()
        or (rounded[:, 0] >= image.shape[1]).any()
        or (rounded[:, 1] < 0).any()
        or (rounded[:, 1] >= image.shape[0]).any()
    ):
        raise ValueError("backend expert minutiae fall outside the image")
    return image, ppi, rounded, hull


def _png_data(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("could not encode an in-memory result")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _contour(canvas: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], width: int = 2) -> None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(canvas, contours, -1, color, width, lineType=cv2.LINE_AA)


def _figure(image: np.ndarray, caption: str, alt: str = "Segmentation overlay") -> str:
    return (
        '<figure><img alt="'
        + escape(alt, quote=True)
        + '" src="'
        + _png_data(image)
        + '"><figcaption>'
        + escape(caption)
        + "</figcaption></figure>"
    )


def _validate_methods(backend: DemoBackend, methods: Sequence[str]) -> list[str]:
    allowed = list(backend.method_names())
    selected = list(dict.fromkeys(methods or []))
    if not selected or any(method not in allowed for method in selected):
        raise ValueError("Select at least one available method.")
    return selected


def _prediction(
    backend: DemoBackend,
    method: str,
    image: np.ndarray,
    ppi: float,
    threshold: float | None,
) -> Prediction:
    expected = float(backend.reported_threshold(method) if threshold is None else threshold)
    if not _finite(expected) or not 0.0 <= expected <= 1.0:
        raise ValueError(f"{method} has an invalid threshold")
    prediction = backend.predict(method, image.copy(), ppi, threshold=threshold)
    if not isinstance(prediction, Prediction):
        raise ValueError(f"{method} returned an invalid prediction record")
    mask = np.asarray(prediction.mask).astype(bool)
    milliseconds = float(prediction.milliseconds)
    applied = float(prediction.applied_threshold)
    if mask.shape != image.shape:
        raise ValueError(f"{method} returned a mask with the wrong shape")
    if not _finite(milliseconds) or milliseconds < 0:
        raise ValueError(f"{method} returned an invalid runtime")
    if not _finite(applied) or not np.isclose(applied, expected, atol=1e-8, rtol=0.0):
        raise ValueError(f"{method} did not apply the requested threshold")
    return Prediction(mask, milliseconds, applied)


def render_explore(
    backend: DemoBackend,
    example_key: str,
    methods: Sequence[str],
    exploration_threshold: float,
) -> str:
    """Render reported results plus an explicitly non-reportable threshold view."""
    try:
        selected = _validate_methods(backend, methods)
        if example_key not in {item.key for item in backend.examples()}:
            raise ValueError("Select a valid test example.")
        image, ppi, ground_truth = _normalise_source(backend.load_example(example_key))
        if ground_truth is None:
            raise ValueError("The selected test example has no ground truth.")
        exploratory = float(exploration_threshold)
        if not _finite(exploratory) or not 0.0 <= exploratory <= 1.0:
            raise ValueError("Exploration threshold must be between 0 and 1.")
        cards: list[str] = []
        rows: list[tuple[str, ...]] = []
        for method in selected:
            reported = _prediction(backend, method, image, ppi, threshold=None)
            capability = getattr(backend, "supports_threshold_exploration", None)
            can_explore = True if capability is None else bool(capability(method))
            viewed = (
                _prediction(backend, method, image, ppi, threshold=exploratory)
                if can_explore
                else reported
            )
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            _contour(overlay, ground_truth, _COLOR_GT)
            _contour(overlay, reported.mask, _COLOR_REPORTED)
            if can_explore:
                _contour(overlay, viewed.mask, _COLOR_EXPLORE, 1)
            cards.append(
                _figure(
                    overlay,
                    method
                    + ("" if can_explore else " — threshold exploration unavailable"),
                )
            )
            metrics = image_metrics(reported.mask, ground_truth)
            rows.append(
                (
                    method,
                    f"{reported.applied_threshold:.2f}",
                    f"{metrics['dice']:.4f}",
                    f"{metrics['mdr']:.4f}",
                    f"{metrics['fdr']:.4f}",
                    f"{reported.milliseconds:.1f}",
                )
            )
        body = "".join(
            "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>"
            for row in rows
        )
        return (
            '<div class="legend">Contours: <b class="gt">ground truth</b>, '
            '<b class="reported">reported threshold</b>, '
            '<b class="explore">exploration threshold</b>.</div>'
            '<p class="warning"><strong>Exploration only:</strong> the slider never changes reported '
            "metrics; those use the validation-selected threshold shown below.</p>"
            '<div class="result-grid">'
            + "".join(cards)
            + "</div><div class=\"result-table\"><table><thead><tr><th>Method</th>"
            "<th>Reported threshold</th><th>Dice</th><th>MDR</th><th>FDR</th><th>ms</th>"
            "</tr></thead><tbody>"
            + body
            + "</tbody></table></div>"
        )
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def _difference_map(
    image: np.ndarray, ground_truth: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray
) -> np.ndarray:
    """Colour disagreements by which method alone is correct."""
    base = cv2.cvtColor(
        (image.astype(np.float32) * 0.55).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )
    correct_a = mask_a == ground_truth
    correct_b = mask_b == ground_truth
    base[correct_a & ~correct_b] = _COLOR_REPORTED
    base[correct_b & ~correct_a] = _COLOR_EXPLORE
    return base


def _comparison_table(
    rows: Sequence[ComparisonRow], method_a: str, method_b: str
) -> str:
    checked: list[ComparisonRow] = []
    for row in rows:
        if not isinstance(row, ComparisonRow):
            raise ValueError("backend returned an invalid comparison row")
        values = (row.dice_a, row.dice_b)
        if not all(
            _finite(float(value)) and 0.0 <= float(value) <= 1.0 for value in values
        ):
            raise ValueError("backend returned an invalid comparison Dice")
        checked.append(row)
    checked.sort(key=lambda row: (-abs(row.dice_a - row.dice_b), row.label, row.key))
    body = "".join(
        "<tr><td>"
        + escape(row.label)
        + "</td><td>"
        + escape(row.examiner_value)
        + f"</td><td>{row.dice_a:.4f}</td><td>{row.dice_b:.4f}</td>"
        + f"<td>{row.dice_a - row.dice_b:+.4f}</td></tr>"
        for row in checked[:20]
    )
    return (
        '<h3>20 largest absolute Dice differences</h3><div class="result-table"><table>'
        "<thead><tr><th>Private example</th><th>Examiner value</th><th>"
        + escape(method_a)
        + " Dice</th><th>"
        + escape(method_b)
        + " Dice</th><th>Delta (A−B)</th></tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def render_compare(
    backend: DemoBackend, example_key: str, method_a: str, method_b: str
) -> str:
    """Compare frozen reported predictions for one image and across test images."""
    try:
        _validate_methods(backend, [method_a, method_b])
        if method_a == method_b:
            raise ValueError("Choose two different methods.")
        if example_key not in {item.key for item in backend.examples()}:
            raise ValueError("Select a valid test example.")
        image, ppi, ground_truth = _normalise_source(backend.load_example(example_key))
        if ground_truth is None:
            raise ValueError("The selected test example has no ground truth.")
        pred_a = _prediction(backend, method_a, image, ppi, threshold=None)
        pred_b = _prediction(backend, method_b, image, ppi, threshold=None)
        metrics_a = image_metrics(pred_a.mask, ground_truth)
        metrics_b = image_metrics(pred_b.mask, ground_truth)
        diff = _difference_map(image, ground_truth, pred_a.mask, pred_b.mask)
        rows = backend.comparison_rows(method_a, method_b, limit=20)
        return (
            '<div class="compare-layout">'
            + _figure(diff, "Difference map", "Method correctness difference map")
            + '<div><h3>Selected example</h3><p><b class="reported">Green:</b> A alone is '
            'correct. <b class="explore">Orange:</b> B alone is correct.</p>'
            '<div class="result-table"><table><thead><tr><th>Method</th><th>Reported '
            "threshold</th><th>Dice</th><th>MDR</th><th>FDR</th><th>ms</th></tr></thead><tbody>"
            + "<tr><td>"
            + escape(method_a)
            + f"</td><td>{pred_a.applied_threshold:.2f}</td><td>{metrics_a['dice']:.4f}</td>"
            + f"<td>{metrics_a['mdr']:.4f}</td><td>{metrics_a['fdr']:.4f}</td>"
            + f"<td>{pred_a.milliseconds:.1f}</td></tr><tr><td>"
            + escape(method_b)
            + f"</td><td>{pred_b.applied_threshold:.2f}</td><td>{metrics_b['dice']:.4f}</td>"
            + f"<td>{metrics_b['mdr']:.4f}</td><td>{metrics_b['fdr']:.4f}</td>"
            + f"<td>{pred_b.milliseconds:.1f}</td></tr></tbody></table></div></div></div>"
            + _comparison_table(rows, method_a, method_b)
        )
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def hull_metrics(
    mask: np.ndarray, hull: np.ndarray, minutiae_xy: np.ndarray
) -> dict[str, float]:
    """Compute lower-bound exemplar metrics without treating the hull as full GT."""
    pred = np.asarray(mask).astype(bool)
    reference = np.asarray(hull).astype(bool)
    points = np.asarray(minutiae_xy, dtype=np.int64)
    if pred.shape != reference.shape or not reference.any():
        raise ValueError("prediction and non-empty hull must have the same shape")
    if points.ndim != 2 or points.shape[1:] != (2,) or not len(points):
        raise ValueError("minutiae must be a non-empty N×2 array")
    if (
        (points[:, 0] < 0).any()
        or (points[:, 0] >= pred.shape[1]).any()
        or (points[:, 1] < 0).any()
        or (points[:, 1] >= pred.shape[0]).any()
    ):
        raise ValueError("minutiae fall outside the prediction")
    pred_area = int(pred.sum())
    hull_area = int(reference.sum())
    return {
        "mc": float(pred[points[:, 1], points[:, 0]].mean()),
        "hr": float(np.count_nonzero(pred & reference) / hull_area),
        "par": float(pred_area / hull_area),
        "ohf": float(np.count_nonzero(pred & ~reference) / pred_area) if pred_area else 0.0,
    }


def render_hull(backend: DemoBackend, example_key: str, method: str) -> str:
    """Render an SD302g lower-bound hull review without pixel-GT metrics."""
    try:
        _validate_methods(backend, [method])
        if example_key not in {item.key for item in backend.hull_examples()}:
            raise ValueError("Select a valid live-scan example.")
        image, ppi, minutiae, hull = _normalise_hull_source(
            backend.load_hull_example(example_key)
        )
        prediction = _prediction(backend, method, image, ppi, threshold=None)
        metrics = hull_metrics(prediction.mask, hull, minutiae)
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        _contour(overlay, hull, _COLOR_HULL)
        _contour(overlay, prediction.mask, _COLOR_REPORTED)
        for x, y in minutiae:
            cv2.circle(
                overlay,
                (int(x), int(y)),
                2,
                _COLOR_MINUTIAE,
                -1,
                lineType=cv2.LINE_AA,
            )
        return (
            '<p class="warning"><strong>Lower-bound protocol:</strong> the minutiae hull is not '
            "full pixel ground truth. Dice, Jaccard, MDR, and FDR are intentionally not shown.</p>"
            '<div class="hull-layout">'
            + _figure(overlay, method, "Prediction, minutiae, and lower-bound hull")
            + '<div><p><b class="hull">Red:</b> expert-minutiae hull; '
            '<b class="minutiae">blue:</b> expert minutiae; '
            '<b class="reported">green:</b> prediction.</p>'
            '<div class="result-table"><table><thead><tr><th>Method</th><th>MC</th><th>HR</th>'
            "<th>PAR</th><th>OHF</th><th>Reported threshold</th><th>ms</th></tr></thead><tbody>"
            + "<tr><td>"
            + escape(method)
            + f"</td><td>{metrics['mc'] * 100:.2f}%</td><td>{metrics['hr'] * 100:.2f}%</td>"
            + f"<td>{metrics['par']:.4f}</td><td>{metrics['ohf'] * 100:.2f}%</td>"
            + f"<td>{prediction.applied_threshold:.2f}</td><td>{prediction.milliseconds:.1f}</td>"
            "</tr></tbody></table></div></div></div>"
        )
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def render_upload(
    backend: DemoBackend,
    upload_data_url: str,
    upload_ppi: float,
    methods: Sequence[str],
) -> str:
    """Run frozen methods on a browser upload without retaining the image."""
    try:
        selected = _validate_methods(backend, methods)
        image = decode_upload(upload_data_url)
        ppi = _normalise_ppi(upload_ppi)
        cards: list[str] = []
        rows: list[str] = []
        for method in selected:
            prediction = _prediction(backend, method, image, ppi, threshold=None)
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            _contour(overlay, prediction.mask, _COLOR_REPORTED)
            cards.append(_figure(overlay, method))
            rows.append(
                "<tr><td>"
                + escape(method)
                + f"</td><td>{prediction.applied_threshold:.2f}</td>"
                + f"<td>{prediction.milliseconds:.1f}</td></tr>"
            )
        return (
            '<p class="privacy-note">The upload was decoded and processed in memory. It was not '
            "stored by this application. Ground-truth metrics are unavailable for uploads.</p>"
            '<div class="result-grid">'
            + "".join(cards)
            + '</div><div class="result-table"><table><thead><tr><th>Method</th>'
            "<th>Reported threshold</th><th>ms</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def filter_examples(examiner_value: str, examples: Sequence[Example]) -> gr.Dropdown:
    """Filter private example choices by examiner assessment."""
    chosen = [
        item
        for item in examples
        if examiner_value == "All" or item.examiner_value == examiner_value
    ]
    choices = [(item.label, item.key) for item in chosen]
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)


UPLOAD_HTML = """
<div class="memory-upload">
  <label for="memory-file-input">Upload fingerprint image</label>
  <input id="memory-file-input" type="file" accept="image/png,image/jpeg,image/webp">
  <p>Decoded in memory and never stored by this app. Maximum request size: 1 GB.</p>
</div>
"""


UPLOAD_JS = r"""
const input = element.querySelector('input[type="file"]');
input.addEventListener('change', () => {
  const file = input.files && input.files[0];
  if (!file) return;
  if (!['image/png','image/jpeg','image/webp'].includes(file.type) || file.size > 1073741824) {
    input.value = '';
    window.alert('Choose a PNG, JPEG, or WebP image no larger than 1 GB.');
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    const root = document.getElementById('upload-data');
    const field = root && root.querySelector('textarea, input');
    if (!field) throw new Error('Upload bridge is unavailable.');
    const proto = field.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(field, reader.result);
    field.dispatchEvent(new Event('input', {bubbles: true}));
    field.dispatchEvent(new Event('change', {bubbles: true}));
  };
  reader.readAsDataURL(file);
});
"""


CSS = """
.gradio-container{max-width:1280px!important}.memory-upload{border:1px dashed #94a3b8;border-radius:10px;padding:18px}
.memory-upload label{display:block;font-weight:700;margin-bottom:8px}.memory-upload p,.privacy-note{color:#64748b;margin:8px 0 0}
.sr-only{position:absolute!important;left:-10000px!important;width:1px!important;height:1px!important;overflow:hidden!important}
.result-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}.result-grid figure,.compare-layout figure,.hull-layout figure{margin:0;border:1px solid #dbe3eb;border-radius:10px;overflow:hidden}
.result-grid img,.compare-layout img,.hull-layout img{display:block;width:100%}.result-grid figcaption,.compare-layout figcaption,.hull-layout figcaption{padding:8px 10px}
.result-table{overflow-x:auto}.result-table table{width:100%;border-collapse:collapse;margin-top:16px}.result-table th,.result-table td{padding:8px;border-bottom:1px solid #dbe3eb;text-align:right;white-space:nowrap}.result-table th:first-child,.result-table td:first-child{text-align:left}
.compare-layout,.hull-layout{display:grid;grid-template-columns:minmax(260px,0.8fr) minmax(420px,1.2fr);gap:18px;align-items:start}.warning{padding:10px 12px;border-left:4px solid #d97706;background:#fffbeb;color:#78350f}.legend{margin:6px 0}.gt,.minutiae{color:#2682e0}.reported{color:#219b75}.explore{color:#d97706}.hull{color:#dc2626}.report-frame{width:100%;min-height:76vh;border:1px solid #dbe3eb;border-radius:10px}
@media(max-width:760px){.compare-layout,.hull-layout{grid-template-columns:1fr}.result-table th,.result-table td{padding:6px;font-size:.86rem}}
"""


def create_app(backend: DemoBackend, report_html: str) -> gr.Blocks:
    """Construct the five-tab private UI without starting a server."""
    examples = list(backend.examples())
    hull_examples = list(backend.hull_examples())
    methods = list(backend.method_names())
    upload_capability = getattr(backend, "upload_method_names", None)
    upload_methods_available = (
        list(upload_capability()) if callable(upload_capability) else methods
    )
    if not examples or not hull_examples or not methods:
        raise ValueError("backend must provide methods, test examples, and hull examples")
    if not upload_methods_available or any(
        method not in methods for method in upload_methods_available
    ):
        raise ValueError("backend must provide at least one valid upload method")
    examiner_values = ["All", *sorted({item.examiner_value for item in examples})]
    example_choices = [(item.label, item.key) for item in examples]
    hull_choices = [
        (
            f"{item.label} — {item.group} / {item.device} / {item.capture_type}",
            item.key,
        )
        for item in hull_examples
    ]
    default_methods = methods[: min(3, len(methods))]
    default_upload_methods = upload_methods_available[
        : min(3, len(upload_methods_available))
    ]
    method_b = methods[1] if len(methods) > 1 else methods[0]

    with gr.Blocks(title="Fingerprint segmentation benchmark", analytics_enabled=False) as app:
        gr.Markdown(
            "# Fingerprint segmentation benchmark\n"
            "Private evaluation interface. Reported metrics always use the frozen "
            "validation-selected threshold."
        )
        with gr.Tabs():
            with gr.Tab("Explore test set"):
                with gr.Row():
                    examiner = gr.Dropdown(
                        examiner_values, value="All", label="Examiner value"
                    )
                    example = gr.Dropdown(
                        example_choices,
                        value=example_choices[0][1],
                        label="Private test example",
                    )
                selected_methods = gr.CheckboxGroup(
                    methods, value=default_methods, label="Methods"
                )
                exploration = gr.Slider(
                    0.0,
                    1.0,
                    value=0.5,
                    step=0.05,
                    label="Exploration threshold (reported metrics remain fixed)",
                )
                explore_run = gr.Button("Render overlays", variant="primary")
                explore_results = gr.HTML(
                    "<p>Select a private test example and one or more methods.</p>"
                )
                examiner.change(
                    lambda value: filter_examples(value, examples),
                    examiner,
                    example,
                    queue=False,
                    api_visibility="private",
                )
                explore_run.click(
                    lambda *values: render_explore(backend, *values),
                    [example, selected_methods, exploration],
                    explore_results,
                    api_visibility="private",
                    concurrency_limit=1,
                )

            with gr.Tab("Compare two methods"):
                compare_example = gr.Dropdown(
                    example_choices,
                    value=example_choices[0][1],
                    label="Private test example",
                )
                with gr.Row():
                    compare_a = gr.Dropdown(methods, value=methods[0], label="Method A")
                    compare_b = gr.Dropdown(methods, value=method_b, label="Method B")
                compare_run = gr.Button("Compare", variant="primary")
                compare_results = gr.HTML(
                    "<p>Choose two methods to view a correctness difference map.</p>"
                )
                compare_run.click(
                    lambda *values: render_compare(backend, *values),
                    [compare_example, compare_a, compare_b],
                    compare_results,
                    api_visibility="private",
                    concurrency_limit=1,
                )

            with gr.Tab("Live-scan (SD302g)"):
                hull_example = gr.Dropdown(
                    hull_choices,
                    value=hull_choices[0][1],
                    label="Private live-scan example",
                )
                hull_method = gr.Dropdown(methods, value=methods[0], label="Method")
                hull_run = gr.Button("Render hull evaluation", variant="primary")
                hull_results = gr.HTML(
                    "<p>Select an example to view minutiae-hull lower-bound metrics.</p>"
                )
                hull_run.click(
                    lambda *values: render_hull(backend, *values),
                    [hull_example, hull_method],
                    hull_results,
                    api_visibility="private",
                    concurrency_limit=1,
                )

            with gr.Tab("Upload"):
                gr.Markdown(
                    "Uploads are processed in memory and are not saved. Reported method "
                    "thresholds are fixed; ground-truth metrics are unavailable."
                )
                gr.HTML(UPLOAD_HTML, js_on_load=UPLOAD_JS, elem_id="memory-uploader")
                upload_data = gr.Textbox(
                    label="Upload data", elem_id="upload-data", elem_classes="sr-only"
                )
                upload_ppi = gr.Number(
                    value=500, minimum=100, maximum=2000, label="Image PPI"
                )
                upload_methods = gr.CheckboxGroup(
                    upload_methods_available,
                    value=default_upload_methods,
                    label="Methods",
                )
                upload_run = gr.Button("Run in-memory segmentation", variant="primary")
                upload_results = gr.HTML("<p>Choose an image and methods.</p>")
                upload_run.click(
                    lambda *values: render_upload(backend, *values),
                    [upload_data, upload_ppi, upload_methods],
                    upload_results,
                    api_visibility="private",
                    concurrency_limit=1,
                )

            with gr.Tab("Report"):
                # The report is embedded as escaped srcdoc in a sandboxed iframe.
                gr.HTML(
                    '<iframe class="report-frame" sandbox srcdoc="'
                    + escape(report_html, quote=True)
                    + '"></iframe>'
                )
    return app.queue(max_size=8, default_concurrency_limit=1, api_open=False)


def launch(app: gr.Blocks, settings: DemoSettings) -> None:
    """Launch with fixed safety arguments; there is no public-share path."""
    validate_bind_host(settings.host)
    app.launch(
        server_name=settings.host,
        server_port=settings.port,
        share=False,
        auth=(settings.username, settings.password),
        inbrowser=False,
        show_error=False,
        allowed_paths=[],
        blocked_paths=["/"],
        max_file_size=MAX_UPLOAD_BYTES,
        enable_monitoring=False,
        strict_cors=True,
        css=CSS,
        footer_links=[],
        max_threads=8,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    settings = load_settings()
    report_html = settings.report_path.read_text(encoding="utf-8")
    backend = load_backend(settings.backend_factory)
    app = create_app(backend, report_html)
    launch(app, settings)


if __name__ == "__main__":
    main()
