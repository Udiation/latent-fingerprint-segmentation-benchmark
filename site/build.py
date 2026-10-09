#!/usr/bin/env python3
"""Build the public static project page from aggregate and synthetic inputs."""
from __future__ import annotations

import csv
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segbench.synthetic_demo import validate_synthetic_demo_receipt

SITE = ROOT / "site"
RESULTS = ROOT / "results"
SYNTHETIC = ROOT / "examples" / "synthetic"

APPROVED_RESULT_COLUMNS = {
    "table", "view", "group", "year", "method", "approach", "family",
    "training_data", "comment", "status", "seeds_completed", "n_images",
    "n_subjects", "accuracy", "error_rate", "dice", "jaccard", "miou", "mdr",
    "fdr", "mc", "hr", "par", "ohf", "params_m", "ms_per_img", "device",
    "accuracy_seed_std", "error_rate_seed_std", "dice_seed_std",
    "jaccard_seed_std", "miou_seed_std", "mdr_seed_std", "fdr_seed_std",
    "mc_seed_std", "hr_seed_std", "par_seed_std", "ohf_seed_std",
    "params_m_seed_std", "ms_per_img_seed_std",
    "ci_lower", "ci_upper",
}
METRICS = {
    "accuracy", "error_rate", "dice", "jaccard", "miou", "mdr", "fdr",
    "mc", "hr", "ohf", "ci_lower", "ci_upper", "accuracy_seed_std",
    "error_rate_seed_std", "dice_seed_std", "jaccard_seed_std",
    "miou_seed_std", "mdr_seed_std", "fdr_seed_std", "mc_seed_std",
    "hr_seed_std", "ohf_seed_std",
}
NONNEGATIVE = {
    "par", "params_m", "ms_per_img", "n_images", "n_subjects",
    "par_seed_std", "params_m_seed_std", "ms_per_img_seed_std",
}

# One card per Fig. 2 entry. Result rows are intentionally not used as a method
# catalogue because GMFS/SUFS and AlexNet/LeNet each expand to multiple rows.
METHOD_CATALOG = (
    {
        "entry": "CNN & autoencoders",
        "family": "Patch CNN",
        "source": "Chhabra et al. (2023)",
        "implementation": "Stacked convolutional autoencoder, then a centred patch classifier.",
        "deviation": "Train-subject patches only; shared 16×16 output blocks.",
    },
    {
        "entry": "GMFS & SUFS",
        "family": "Algorithmic / released network",
        "source": "Cappelli (2023)",
        "implementation": "Released GMFS algorithm and SUFS weights; separate tuned and fine-tuned rows.",
        "deviation": "No architecture substitution.",
    },
    {
        "entry": "U-Net",
        "family": "Encoder–decoder",
        "source": "Jaafar et al. (2022); Ronneberger et al. (2015)",
        "implementation": "Four levels, base width 32, BatchNorm/ReLU and bilinear upsampling.",
        "deviation": "Faithful architecture reimplementation.",
    },
    {
        "entry": "Adaptive Gaussian transformer",
        "family": "Transformer",
        "source": "Liu et al. (2022); Xie et al. (2021)",
        "implementation": "SegFormer-B2 with ImageNet initialisation and a one-channel adapter.",
        "deviation": "† Declared 2-D substitute for the reviewed volumetric OCT model.",
    },
    {
        "entry": "DRUnet",
        "family": "Encoder–decoder",
        "source": "Joshi et al.",
        "implementation": "R2U-Net-style recurrent blocks with two steps and dropout 0.2.",
        "deviation": "Deterministic table inference; MC dropout is an auxiliary analysis.",
    },
    {
        "entry": "Recurrent adversarial network",
        "family": "Domain adaptation",
        "source": "Joshi et al. (2021)",
        "implementation": "Recurrent backbone with bottleneck gradient-reversal alignment.",
        "deviation": "Table A disables alignment; RA-RUnet is restricted to the cross-domain setting.",
    },
    {
        "entry": "Deep Nested UNets",
        "family": "Encoder–decoder",
        "source": "Liu & Qian (2020); Zhou et al. (2018)",
        "implementation": "Native UNet++ with nested dense skips and four supervision heads.",
        "deviation": "Final head only at inference.",
    },
    {
        "entry": "Support Vector Machine",
        "family": "Classical",
        "source": "Cortes & Vapnik (1995); Bazen & Gerez (2001)",
        "implementation": "Nyström RBF approximation over one shared eight-feature block descriptor.",
        "deviation": "Kernel approximation is disclosed for tractability.",
    },
    {
        "entry": "AlexNet & LeNet",
        "family": "Patch CNN",
        "source": "Serafim et al. (2019)",
        "implementation": "Centred 64×64 and 32×32 patches predict a 16×16 output block.",
        "deviation": "AlexNet uses an output-equivalent deterministic fixed pool.",
    },
    {
        "entry": "Deep CNN",
        "family": "Released network",
        "source": "Tang et al. (2017)",
        "implementation": "Official FingerNet weights and segmentation head, evaluated zero-shot.",
        "deviation": "No replacement when the legacy runtime is unavailable.",
    },
    {
        "entry": "Random decision forest",
        "family": "Classical",
        "source": "Sankaran et al. (2017)",
        "implementation": "Balanced random forest over the shared block descriptor.",
        "deviation": "Common validation grid and post-processing.",
    },
    {
        "entry": "Weighted extreme learning machine",
        "family": "Classical",
        "source": "Ezeobiejesi & Bhanu (2016)",
        "implementation": "Single-hidden-layer ELM with inverse-class-frequency W1 weights.",
        "deviation": "Common block descriptor and post-processing.",
    },
)


def read_csv(path: Path, approved: set[str]) -> list[dict[str, str]]:
    """Read a public aggregate CSV with an exact, numeric-safe schema."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != approved:
            raise ValueError(f"unexpected columns in {path.name}: {reader.fieldnames}")
        rows = list(reader)
    for row in rows:
        for key in METRICS:
            if key in row and row[key] != "":
                value = float(row[key])
                if not 0 <= value <= 1:
                    raise ValueError(f"{path.name} {key} outside [0,1]")
        for key in NONNEGATIVE:
            if key in row and row[key] != "":
                value = float(row[key])
                if value < 0:
                    raise ValueError(f"{path.name} {key} is negative")
    return rows


def _approved_assets(receipt: dict[str, object]) -> dict[str, dict[str, str]]:
    assets = receipt.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError("synthetic receipt assets must be a list")
    approved: dict[str, dict[str, str]] = {}
    for item in assets:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("synthetic receipt has a malformed asset")
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".png":
            raise ValueError("synthetic receipt has an unsafe asset path")
        key = relative.as_posix()
        if key in approved:
            raise ValueError("synthetic receipt has a duplicate asset path")
        approved[key] = {str(field): str(value) for field, value in item.items()}
    return approved


def _copy_receipt_asset(
    approved: dict[str, dict[str, str]], relative: str, destination: Path
) -> None:
    source = SYNTHETIC / relative
    expected = approved[relative].get("sha256", "")
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"synthetic receipt mismatch for {source.name}")
    shutil.copy2(source, destination)


def synthetic_examples() -> list[dict[str, object]]:
    """Copy receipt-bound synthetic sources, GT, and optional method outputs.

    A method prediction is accepted only when its receipt row has role
    ``method_prediction`` plus ``method`` and ``source_stem``. This lets the
    browser demo consume genuine synthetic inference artifacts later without
    ever fabricating an unavailable method output.
    """
    receipt_path = SYNTHETIC / "synthetic_receipt.json"
    if not receipt_path.is_file():
        return []
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    demo_index = validate_synthetic_demo_receipt(
        SYNTHETIC,
        receipt,
        verify_files=True,
        verify_builder=True,
        builder_path=ROOT / "scripts" / "build_synthetic_site_demo.py",
    )
    approved = _approved_assets(receipt)
    image_paths = []
    for stem in demo_index["display_source_stems"]:
        candidates = [
            path
            for path, item in approved.items()
            if item.get("role") == "example_image" and Path(path).stem == stem
        ]
        if len(candidates) != 1:
            raise ValueError(f"synthetic display source {stem} is not uniquely receipt-bound")
        image_paths.append(candidates[0])
    destination = SITE / "assets" / "synthetic"
    destination.mkdir(parents=True, exist_ok=True)
    for stale in destination.glob("prediction-*.png"):
        stale.unlink()
    output: list[dict[str, object]] = []
    for index, image_rel in enumerate(image_paths, start=1):
        stem = Path(image_rel).stem
        mask_rel = next(
            (
                path
                for path, item in approved.items()
                if item.get("role") == "ground_truth_mask" and Path(path).stem == stem
            ),
            "",
        )
        if not mask_rel:
            raise ValueError(f"synthetic image {stem} lacks a receipt-bound mask")
        source_destination = destination / f"source-{index}.png"
        mask_destination = destination / f"mask-{index}.png"
        _copy_receipt_asset(approved, image_rel, source_destination)
        _copy_receipt_asset(approved, mask_rel, mask_destination)
        predictions: list[dict[str, str]] = []
        methods: list[dict[str, str]] = []
        for method, record in demo_index["methods"].items():
            methods.append(
                {
                    "method": method,
                    "label": str(record["label"]),
                    "status": str(record["status"]),
                    "reason": str(record.get("reason", "")),
                }
            )
            matching = [
                item
                for item in demo_index["predictions"][method]
                if item.get("source_stem") == stem
            ]
            if record["status"] != "complete":
                continue
            if len(matching) != 1:
                raise ValueError(f"complete synthetic method {method} lacks source {stem}")
            item = matching[0]
            prediction_rel = str(item["path"])
            prediction_destination = destination / f"prediction-{index}-{method}.png"
            _copy_receipt_asset(approved, prediction_rel, prediction_destination)
            predictions.append(
                {
                    "method": method,
                    "label": item.get("label", method),
                    "mask": f"assets/synthetic/{prediction_destination.name}",
                }
            )
        output.append(
            {
                "key": f"synthetic-{index}",
                "label": f"Synthetic sample {index}",
                "image": f"assets/synthetic/{source_destination.name}",
                "mask": f"assets/synthetic/{mask_destination.name}",
                "predictions": sorted(predictions, key=lambda row: row["label"]),
                "methods": methods,
            }
        )
    return output


def method_cards() -> str:
    """Render exactly one factual card for each of the twelve review entries."""
    cards = []
    for method in METHOD_CATALOG:
        cards.append(
            '<article class="card method">'
            f'<p class="meta">{escape(method["family"])}</p>'
            f'<h3>{escape(method["entry"])}</h3>'
            f'<p><strong>Source:</strong> {escape(method["source"])}</p>'
            f'<p><strong>Here:</strong> {escape(method["implementation"])}</p>'
            f'<p class="deviation">{escape(method["deviation"])}</p>'
            "</article>"
        )
    return "".join(cards)


def _synthetic_panel(synthetic: list[dict[str, object]]) -> str:
    if not synthetic:
        return (
            '<p class="notice">Synthetic assets have not been generated yet. '
            "Run the quick-start command and rebuild.</p>"
        )
    return '''<div class="synthetic">
<div class="synthetic-controls"><label for="synthetic-select">Synthetic sample</label><select id="synthetic-select"></select><label for="synthetic-method">Output</label><select id="synthetic-method"></select><p id="synthetic-note" class="notice">These deterministic CPU micro-runs exercise repository code on generated patterns only. They are not benchmark evidence. Unavailable external-weight methods are named explicitly and never receive fabricated masks.</p><ul id="synthetic-availability" class="availability"></ul></div>
<div class="grid"><figure class="card synthetic-figure"><img id="synthetic-source" alt="Synthetic ridge-pattern source"><figcaption>Synthetic source</figcaption></figure><figure class="card synthetic-figure"><img id="synthetic-mask" alt="Selected synthetic foreground mask"><figcaption id="synthetic-caption">Synthetic ground truth</figcaption></figure></div>
</div>'''


def main() -> None:
    """Rebuild the static page and its source-digest manifest."""
    results = read_csv(RESULTS / "public_results.csv", APPROVED_RESULT_COLUMNS)
    tables = read_csv(
        RESULTS / "table_status.csv", {"table", "title", "status", "note"}
    )
    stats = read_csv(
        RESULTS / "statistics.csv",
        {
            "table", "metric", "comparison", "estimate", "ci_lower", "ci_upper",
            "p_holm", "significant", "status",
        },
    )
    synthetic = synthetic_examples()
    public_result_assets = SITE / "assets" / "results"
    public_result_assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        RESULTS / "public_results.csv", public_result_assets / "public_results.csv"
    )
    payload = {
        "schema_version": 1,
        "results": results,
        "tables": tables,
        "statistics": stats,
        "synthetic": synthetic,
    }
    payload_json = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":")
    ).replace("</", "<\\/")
    tabs = "".join(
        f'<button id="tab-{escape(row["table"])}" class="tab" role="tab" '
        f'data-table="{escape(row["table"])}" aria-controls="result-panel" '
        f'aria-selected="{"true" if row["table"] == "A" else "false"}">'
        f'{escape(row["table"].replace("_prime", "′").replace("_value", "-value"))}'
        "</button>"
        for row in tables
    )
    abstract = (
        "Fingerprint foreground segmentation affects every downstream operation that depends "
        "on ridge-bearing area. We compare the twelve segmentation entries catalogued in a "
        "recent review under one subject-disjoint split, one physical resolution, and one "
        "metric implementation. Examiner ROI polygons define the primary latent endpoint, "
        "while usable-ridge maps provide a second ground-truth definition. Expert-minutiae "
        "hulls support a lower-bound exemplar analysis without being misrepresented as full "
        "foreground masks. Learning rates, stopping epochs, thresholds, and post-processing "
        "are selected using validation subjects only; each frozen model is scored once on the "
        "test subjects. Uncertainty uses subject-cluster bootstrap intervals and paired tests "
        "with Holm correction. The public release contains aggregate tables, reproducible code, "
        "and receipt-bound synthetic illustrations only. Licensed biometric samples, per-image "
        "records, trained weights, and private infrastructure details are excluded."
    )
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="segbench-public-safe" content="1"><title>Latent Fingerprint Segmentation Benchmark</title><meta name="description" content="A reproducible subject-disjoint benchmark of fingerprint foreground segmentation methods."><link rel="icon" href="assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="assets/style.css"></head>
<body><header class="top"><nav class="wrap nav" aria-label="Main"><a class="brand" href="#top">SegBench</a><div class="nav-links"><a href="#problem">Problem</a><a href="#pipeline">Pipeline</a><a href="#datasets">Data</a><a href="#methods">Methods</a><a href="#results">Results</a><a href="#demo">Demo</a><a href="#reproduce">Reproduce</a></div><button id="theme-toggle" class="theme" aria-label="Toggle colour theme">Theme</button></nav></header>
<main id="top"><section class="hero"><div class="wrap"><p class="eyebrow">Fingerprint segmentation · reproducible evaluation</p><h1>One split. One evaluator. Twelve method entries.</h1><p class="lede">A validation-selected comparison of classical, patch, encoder–decoder, transformer, and released segmentation methods on examiner-annotated latent fingerprints.</p><p class="authors">Udit Narayan · TODO: co-authors · Affiliation TODO</p><div class="buttons"><span class="button" aria-disabled="true">Paper pending</span><a class="button primary" href="https://github.com/Udiation/latent-fingerprint-segmentation-benchmark">Code</a><a class="button" href="assets/results/public_results.csv">Results CSV</a><a class="button" href="#citation">Cite</a></div><p class="status"><strong>Results pending:</strong> public metrics remain blank until the preregistered seed-zero completion gate.</p></div></section>
<section class="section" id="abstract"><div class="wrap"><h2>Abstract</h2><p class="section-intro">{escape(abstract)}</p></div></section>
<section id="problem" class="section"><div class="wrap"><h2>The problem</h2><p class="section-intro">Background texture can create false ridge evidence; overly tight masks can remove usable minutiae. The benchmark measures both missed foreground and false background detection while preserving the distinction between an examiner ROI and a usable-ridge map.</p><div class="synthetic-problem"><img src="assets/synthetic/source-1.png" alt="Generated ridge-pattern example"><img src="assets/synthetic/mask-1.png" alt="Generated foreground mask"><p class="meta">Generated illustration only; no biometric data.</p></div></div></section>
<section id="pipeline" class="section"><div class="wrap"><h2>Pipeline</h2><p class="section-intro">All methods share the same study units, split, physical resolution, validation boundary, and evaluator.</p><img class="pipeline" src="assets/pipeline.svg" alt="Benchmark pipeline from licensed data through examiner GT, subject split, methods, evaluation and statistics"></div></section>
<section id="datasets" class="section"><div class="wrap"><h2>Datasets and ground truth</h2><p class="section-intro">The public repository ships no source fingerprint. Counts and definitions below describe the authorised local evaluation protocol.</p><div class="grid"><article class="card dataset"><h3>SD302-Latent</h3><p>10,092 annotated impressions from 200 subjects. Examiner ROI polygons define Table A; 10,024 complete ridge-quality maps define Table A-prime.</p><p class="meta">136 / 24 / 40 subjects · train / validation / test</p></article><article class="card dataset"><h3>SD302-Exemplar</h3><p>Expert minutiae form a lower-bound convex hull. B1 reports MC, HR, PAR, and OHF—never Dice against the hull.</p></article><article class="card dataset"><h3>Exclusions</h3><p>68 incomplete quality annotations affect A-prime only; two NONPRINT annotations are excluded from test scoring.</p></article></div><figure class="hull-explainer"><img src="assets/hull-explainer.svg" alt="Synthetic ridge pattern with expert minutiae points and their convex hull"><figcaption>A synthetic explanation of the exemplar endpoint: the minutiae hull is known foreground, while ridge area outside it remains unknown.</figcaption></figure></div></section>
<section id="methods" class="section"><div class="wrap"><h2>Methods</h2><p class="section-intro">Exactly one card represents each Fig. 2 entry. Multi-row entries expand only in the result tables. The dagger marks the declared two-dimensional substitute.</p><div class="grid">{method_cards()}</div></div></section>
<section id="results" class="section"><div class="wrap"><h2>Interactive results</h2><p class="section-intro">Every value, interval, and significance row below is generated from the same tracked aggregate CSV/JSON payload used by the paper tables. Empty cells are deliberately rendered as pending.</p><div class="tabs" role="tablist" aria-label="Result tables">{tabs}</div><div id="result-panel" role="tabpanel" aria-live="polite"><div class="controls"><label>Metric <select id="metric"><option value="dice">Dice</option><option value="jaccard">Jaccard</option><option value="mdr">MDR</option><option value="fdr">FDR</option><option value="accuracy">Accuracy</option><option value="mc">MC</option><option value="hr">HR</option><option value="par">PAR</option><option value="ohf">OHF</option><option value="ms_per_img">ms/img</option></select></label><label id="group-filter-wrap" hidden>Device / capture group <select id="group-filter"><option value="all">All groups</option></select></label><button id="sort-toggle" class="button" aria-label="High to low">High to low</button></div><div class="table-scroll"><table><thead><tr><th>Approach</th><th>Family</th><th>Status</th><th>Metric and 95% CI</th><th>Seeds</th><th>Params M</th><th>ms/img</th><th>Device</th></tr></thead><tbody id="result-body"></tbody></table></div><h3>Dice–runtime trade-off</h3><div id="pareto" class="chart"></div><h3>Corrected significance</h3><div class="table-scroll significance"><table><thead><tr><th>Metric</th><th>Comparison</th><th>Estimate</th><th>95% CI</th><th>Holm p</th><th>Significant</th><th>Status</th></tr></thead><tbody id="significance-body"></tbody></table></div></div></div></section>
<section id="demo" class="section"><div class="wrap"><h2>Synthetic browser demo</h2><p class="section-intro">Choose a generated print and a receipt-bound output. Ground truth is always available; each completed output comes from the named repository implementation after synthetic-train-only fitting where required. These micro-runs demonstrate executable code paths, not relative accuracy or biometric performance.</p>{_synthetic_panel(synthetic)}</div></section>
<section id="findings" class="section"><div class="wrap"><h2>Findings</h2><ul id="findings-list"><li>Pending aggregate finalisation.</li></ul></div></section>
<section id="reproduce" class="section"><div class="wrap"><h2>Reproducibility</h2><p class="section-intro">No script downloads biometric data. Supply locally authorised paths and frozen receipts.</p><pre>python -m pip install -r requirements.txt
python scripts/smoke_synthetic.py --work-dir .smoke
bash scripts/reproduce_table_A.sh --config local/benchmark.env
python site/build.py
python scripts/check_site_numbers.py
python scripts/check_release_requirements.py
python scripts/scan_release.py --phase working --allow-limited-synthetic</pre><p>Seeds, thresholds, immutable split assignment, and one-command table entry points are documented in <a href="https://github.com/Udiation/latent-fingerprint-segmentation-benchmark/blob/main/docs/PROTOCOL.md">PROTOCOL.md</a>.</p></div></section>
<section id="limitations" class="section"><div class="wrap"><h2>Limitations</h2><p class="section-intro">Reimplementations can differ from original code; SegFormer is a declared substitute; released zero-shot models have external training provenance; ROI and usable-ridge GT answer different questions; exemplar hulls are lower bounds; and one coordinated data source cannot establish every deployment domain.</p></div></section>
<section id="citation" class="section"><div class="wrap"><h2>Citation</h2><pre id="bibtex">@software{{narayan2026latentseg,
  author = {{Narayan, Udit and {{TODO: co-authors}}}},
  title = {{Latent Fingerprint Segmentation Benchmark}},
  year = {{2026}},
  url = {{https://github.com/Udiation/latent-fingerprint-segmentation-benchmark}}
}}</pre><button id="copy-cite" class="button">Copy BibTeX</button><h3>Acknowledgements</h3><p>NIST SD302; pyfing; the FingerNet authors; and the Hugging Face SegFormer implementation.</p></div></section></main><footer class="footer"><div class="wrap">Aggregate results and synthetic examples only · no tracking or analytics</div></footer>
<script id="public-data" type="application/json">{payload_json}</script><script src="assets/app.js"></script></body></html>'''
    (SITE / "index.html").write_text(html, encoding="utf-8")
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            RESULTS / "public_results.csv",
            RESULTS / "table_status.csv",
            RESULTS / "statistics.csv",
        )
    }
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "source_sha256": source_hashes,
        "result_rows": len(results),
        "synthetic_examples": len(synthetic),
        "synthetic_complete_methods": (
            sum(method["status"] == "complete" for method in synthetic[0]["methods"])
            if synthetic else 0
        ),
        "synthetic_unavailable_methods": (
            sum(method["status"] == "unavailable" for method in synthetic[0]["methods"])
            if synthetic else 0
        ),
        "method_cards": len(METHOD_CATALOG),
    }
    (SITE / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
