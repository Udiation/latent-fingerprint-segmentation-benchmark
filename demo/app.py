#!/usr/bin/env python3
"""Authenticated private segmentation demo with in-memory browser uploads.

The app intentionally has no Gradio File/Image upload component. A browser
FileReader converts an image to a data URL, which the callback decodes in
memory. Results are returned as HTML data images, avoiding Gradio's file cache.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from html import escape
import ipaddress
import os
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2
import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.backend import DemoBackend, Example, LoadedExample, Prediction, load_backend  # noqa: E402
from segbench.metrics import image_metrics  # noqa: E402


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_PIXELS = 25_000_000
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
        "DEMO_HOST", "DEMO_PORT", "DEMO_USER", "DEMO_PASS", "DEMO_REPORT_PATH",
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
    if '<meta name="segbench-private-report" content="1">' not in report_head \
            or 'name="segbench-public-safe"' in report_head:
        raise ValueError("DEMO_REPORT_PATH is not a private report produced by the builder")
    return DemoSettings(host, port, username, password, report.resolve(), env["DEMO_BACKEND_FACTORY"])


def decode_upload(data_url: str) -> np.ndarray:
    """Decode and validate a browser data URL without filesystem I/O."""
    match = _DATA_URL.fullmatch((data_url or "").strip())
    if not match:
        raise ValueError("Upload a PNG, JPEG, or WebP image.")
    try:
        payload = base64.b64decode(match.group(2), validate=True)
    except ValueError as exc:
        raise ValueError("The uploaded image encoding is invalid.") from exc
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("The uploaded image exceeds the 10 MB limit.")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2 or image.size > MAX_UPLOAD_PIXELS:
        raise ValueError("The uploaded image is invalid or its dimensions are too large.")
    return np.ascontiguousarray(image)


def _normalise_source(value: LoadedExample) -> tuple[np.ndarray, float, np.ndarray | None]:
    image = np.asarray(value.image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("backend returned an invalid source image")
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise ValueError("backend returned a non-finite source image")
        if float(image.max()) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    ppi = float(value.ppi)
    if not math_is_finite(ppi) or not 100 <= ppi <= 2000:
        raise ValueError("backend returned an invalid ppi")
    ground_truth = None if value.ground_truth is None else np.asarray(value.ground_truth).astype(bool)
    if ground_truth is not None and ground_truth.shape != image.shape:
        raise ValueError("backend ground truth shape differs from its image")
    return np.ascontiguousarray(image), ppi, ground_truth


def math_is_finite(value: float) -> bool:
    return not (np.isnan(value) or np.isinf(value))


def _png_data(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("could not encode an in-memory result")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (33, 155, 117), 2, lineType=cv2.LINE_AA)
    return rgb


def render_predictions(
    backend: DemoBackend,
    source_mode: str,
    example_key: str,
    upload_data_url: str,
    upload_ppi: float,
    methods: Sequence[str],
) -> str:
    """Run selected methods and return an entirely in-memory HTML result."""
    allowed_methods = list(backend.method_names())
    selected = list(methods or [])
    if not selected or any(method not in allowed_methods for method in selected):
        raise gr.Error("Select at least one available method.")
    try:
        if source_mode == "Private test example":
            allowed_keys = {item.key for item in backend.examples()}
            if example_key not in allowed_keys:
                raise ValueError("Select a valid test example.")
            image, ppi, ground_truth = _normalise_source(backend.load_example(example_key))
        elif source_mode == "In-memory upload":
            image = decode_upload(upload_data_url)
            ppi = float(upload_ppi)
            if not math_is_finite(ppi) or not 100 <= ppi <= 2000:
                raise ValueError("PPI must be between 100 and 2000.")
            ground_truth = None
        else:
            raise ValueError("Select an input source.")
        cards, metric_rows = [], []
        for method in selected:
            prediction = backend.predict(method, image.copy(), ppi)
            if not isinstance(prediction, Prediction):
                raise ValueError(f"{method} returned an invalid prediction record")
            mask = np.asarray(prediction.mask).astype(bool)
            if mask.shape != image.shape:
                raise ValueError(f"{method} returned a mask with the wrong shape")
            milliseconds = float(prediction.milliseconds)
            if not math_is_finite(milliseconds) or milliseconds < 0:
                raise ValueError(f"{method} returned an invalid runtime")
            cards.append(
                '<figure><img alt="Segmentation overlay" src="' + _png_data(_overlay(image, mask))
                + '"><figcaption>' + escape(method) + "</figcaption></figure>"
            )
            if ground_truth is None:
                metric_rows.append((method, "—", "—", "—", f"{milliseconds:.1f}"))
            else:
                metrics = image_metrics(mask, ground_truth)
                metric_rows.append((method, f"{metrics['dice']:.4f}", f"{metrics['mdr']:.4f}",
                                    f"{metrics['fdr']:.4f}", f"{milliseconds:.1f}"))
        rows = "".join("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>"
                       for row in metric_rows)
        return (
            '<div class="result-grid">' + "".join(cards) + "</div>"
            '<div class="result-table"><table><thead><tr><th>Method</th><th>Dice</th>'
            '<th>MDR</th><th>FDR</th><th>ms</th></tr></thead><tbody>' + rows + "</tbody></table></div>"
        )
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def filter_examples(examiner_value: str, examples: Sequence[Example]) -> gr.Dropdown:
    chosen = [item for item in examples if examiner_value == "All" or item.examiner_value == examiner_value]
    choices = [(item.label, item.key) for item in chosen]
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)


UPLOAD_HTML = """
<div class="memory-upload">
  <label for="memory-file-input">Upload fingerprint image</label>
  <input id="memory-file-input" type="file" accept="image/png,image/jpeg,image/webp">
  <p>Decoded in memory; maximum 10 MB.</p>
</div>
"""

UPLOAD_JS = r"""
const input = element.querySelector('input[type="file"]');
input.addEventListener('change', () => {
  const file = input.files && input.files[0];
  if (!file) return;
  if (!['image/png','image/jpeg','image/webp'].includes(file.type) || file.size > 10485760) {
    input.value = '';
    window.alert('Choose a PNG, JPEG, or WebP image no larger than 10 MB.');
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
.gradio-container{max-width:1220px!important}.memory-upload{border:1px dashed #94a3b8;border-radius:10px;padding:18px}
.memory-upload label{display:block;font-weight:700;margin-bottom:8px}.memory-upload p{color:#64748b;margin:8px 0 0}
.sr-only{position:absolute!important;left:-10000px!important;width:1px!important;height:1px!important;overflow:hidden!important}
.result-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}.result-grid figure{margin:0;border:1px solid #dbe3eb;border-radius:10px;overflow:hidden}
.result-grid img{display:block;width:100%}.result-grid figcaption{padding:8px 10px}.result-table table{width:100%;border-collapse:collapse;margin-top:16px}.result-table th,.result-table td{padding:8px;border-bottom:1px solid #dbe3eb;text-align:right}.result-table th:first-child,.result-table td:first-child{text-align:left}
.report-frame{width:100%;min-height:76vh;border:1px solid #dbe3eb;border-radius:10px}
"""


def create_app(backend: DemoBackend, report_html: str) -> gr.Blocks:
    """Construct the private UI without starting a server."""
    examples = list(backend.examples())
    methods = list(backend.method_names())
    examiner_values = ["All", *sorted({item.examiner_value for item in examples})]
    initial_choices = [(item.label, item.key) for item in examples]
    with gr.Blocks(title="Fingerprint segmentation benchmark", analytics_enabled=False) as app:
        gr.Markdown("# Fingerprint segmentation benchmark\nPrivate evaluation interface")
        with gr.Tabs():
            with gr.Tab("Interactive demo"):
                source = gr.Radio(
                    ["Private test example", "In-memory upload"], value="Private test example",
                    label="Input source",
                )
                with gr.Row():
                    examiner = gr.Dropdown(examiner_values, value="All", label="Examiner value")
                    example = gr.Dropdown(initial_choices, value=initial_choices[0][1], label="Test example")
                gr.HTML(UPLOAD_HTML, js_on_load=UPLOAD_JS, elem_id="memory-uploader")
                upload_data = gr.Textbox(label="Upload data", elem_id="upload-data", elem_classes="sr-only")
                upload_ppi = gr.Number(value=500, minimum=100, maximum=2000, label="Upload PPI")
                selected_methods = gr.CheckboxGroup(methods, value=methods[: min(3, len(methods))], label="Methods")
                run = gr.Button("Run segmentation", variant="primary")
                results = gr.HTML('<p class="empty">Select an input and methods.</p>')
                examiner.change(
                    lambda value: filter_examples(value, examples), examiner, example,
                    queue=False, api_visibility="private",
                )
                run.click(
                    lambda *values: render_predictions(backend, *values),
                    [source, example, upload_data, upload_ppi, selected_methods], results,
                    api_visibility="private", concurrency_limit=1,
                )
            with gr.Tab("Report"):
                # srcdoc is sandboxed and escaped, so the report cannot navigate
                # or access the surrounding application.
                gr.HTML(
                    '<iframe class="report-frame" sandbox srcdoc="'
                    + escape(report_html, quote=True) + '"></iframe>'
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
