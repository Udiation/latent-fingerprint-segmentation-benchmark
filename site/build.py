#!/usr/bin/env python3
"""Build the public static project page from aggregate and synthetic inputs."""
from __future__ import annotations

import csv
import hashlib
from html import escape
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
RESULTS = ROOT / "results"
SYNTHETIC = ROOT / "examples" / "synthetic"
APPROVED_RESULT_COLUMNS = {
    "table", "year", "method", "approach", "family", "training_data", "comment",
    "status", "seeds_completed", "dice", "jaccard", "mdr", "fdr", "accuracy",
    "params_m", "ms_per_img", "device", "ci_lower", "ci_upper",
}
METRICS = {"dice", "jaccard", "mdr", "fdr", "accuracy", "ci_lower", "ci_upper"}


def read_csv(path: Path, approved: set[str]) -> list[dict[str, str]]:
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
    return rows


def synthetic_examples() -> list[dict[str, str]]:
    receipt_path = SYNTHETIC / "synthetic_receipt.json"
    if not receipt_path.is_file():
        return []
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    approved = {item["path"]: item for item in receipt.get("assets", [])}
    image_paths = sorted(path for path, item in approved.items() if item.get("role") == "example_image" and "finger_latent" in path)[:3]
    destination = SITE / "assets" / "synthetic"
    destination.mkdir(parents=True, exist_ok=True)
    output = []
    for index, image_rel in enumerate(image_paths, start=1):
        image_path = SYNTHETIC / image_rel
        stem = image_path.stem
        mask_rel = next((p for p, item in approved.items() if item.get("role") == "ground_truth_mask" and Path(p).stem == stem), "")
        if not mask_rel:
            raise ValueError(f"synthetic image {stem} lacks a receipt-bound mask")
        mask_path = SYNTHETIC / mask_rel
        for path, rel in ((image_path, f"source-{index}.png"), (mask_path, f"mask-{index}.png")):
            expected = approved[path.relative_to(SYNTHETIC).as_posix()]["sha256"]
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"synthetic receipt mismatch for {path.name}")
            shutil.copy2(path, destination / rel)
        output.append({"key": f"synthetic-{index}", "label": f"Synthetic sample {index}", "image": f"assets/synthetic/source-{index}.png", "mask": f"assets/synthetic/mask-{index}.png"})
    return output


def method_cards(results: list[dict[str, str]]) -> str:
    seen: set[str] = set()
    cards = []
    for row in results:
        if row["method"] in {"all_background", "all_foreground"} or row["method"] in seen:
            continue
        seen.add(row["method"])
        cards.append(f'<article class="card method"><p class="meta">{escape(row["family"])}</p><h3>{escape(row["approach"])}</h3><p>{escape(row["comment"])}</p><p class="meta">{escape(row["training_data"])}</p></article>')
    return "".join(cards)


def main() -> None:
    results = read_csv(RESULTS / "public_results.csv", APPROVED_RESULT_COLUMNS)
    tables = read_csv(RESULTS / "table_status.csv", {"table", "title", "status", "note"})
    stats = read_csv(RESULTS / "statistics.csv", {"table", "metric", "comparison", "estimate", "ci_lower", "ci_upper", "p_holm", "significant", "status"})
    synthetic = synthetic_examples()
    public_result_assets = SITE / "assets" / "results"
    public_result_assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RESULTS / "public_results.csv", public_result_assets / "public_results.csv")
    payload = {"schema_version": 1, "results": results, "tables": tables, "statistics": stats, "synthetic": synthetic}
    payload_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    tabs = "".join(f'<button id="tab-{escape(row["table"])}" class="tab" role="tab" data-table="{escape(row["table"])}" aria-controls="result-panel" aria-selected="{"true" if row["table"] == "A" else "false"}">{escape(row["table"].replace("_prime", "′").replace("_value", "-value"))}</button>' for row in tables)
    synthetic_panel = '<p class="notice">Synthetic assets have not been generated yet. Run the quick-start command and rebuild.</p>' if not synthetic else '<div class="synthetic"><div><label for="synthetic-select">Synthetic sample</label><select id="synthetic-select"></select><p class="notice">Method outputs remain pending; this panel shows only a generated source and its generated GT.</p></div><div class="grid"><figure class="card" style="grid-column:span 6"><img id="synthetic-source" alt="Synthetic ridge-pattern source"><figcaption>Synthetic source</figcaption></figure><figure class="card" style="grid-column:span 6"><img id="synthetic-mask" alt="Synthetic foreground mask"><figcaption>Synthetic GT</figcaption></figure></div></div>'
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="segbench-public-safe" content="1"><title>Latent Fingerprint Segmentation Benchmark</title><meta name="description" content="A reproducible subject-disjoint benchmark of fingerprint foreground segmentation methods."><link rel="icon" href="assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="assets/style.css"></head>
<body><header class="top"><nav class="wrap nav" aria-label="Main"><a class="brand" href="#top">SegBench</a><div class="nav-links"><a href="#problem">Problem</a><a href="#pipeline">Pipeline</a><a href="#methods">Methods</a><a href="#results">Results</a><a href="#reproduce">Reproduce</a></div><button id="theme-toggle" class="theme" aria-label="Toggle colour theme">Theme</button></nav></header>
<main id="top"><section class="hero"><div class="wrap"><p class="eyebrow">Fingerprint segmentation · reproducible evaluation</p><h1>One split. One evaluator. Twelve method entries.</h1><p class="lede">A validation-selected comparison of classical, patch, encoder–decoder, transformer and released segmentation methods on examiner-annotated latent fingerprints.</p><p class="authors">Udit Narayan · TODO: co-authors · Affiliation TODO</p><div class="buttons"><a class="button primary" href="https://github.com/Udiation/latent-fingerprint-segmentation-benchmark">Code</a><a class="button" href="assets/results/public_results.csv">Results CSV</a><a class="button" href="#citation">Cite</a><span class="button" aria-disabled="true">Paper pending</span></div><p class="status"><strong>Results pending:</strong> public metrics remain blank until the preregistered seed-zero completion gate.</p></div></section>
<section class="section"><div class="wrap"><h2>Abstract</h2><p class="section-intro">Fingerprint foreground segmentation affects every downstream operation that depends on ridge-bearing area. We benchmark the segmentation methods catalogued in a recent review using a single subject-disjoint split, common resolution and common metric implementation. Examiner ROI polygons define the primary latent endpoint; usable-ridge maps provide a second GT definition; expert-minutiae hulls support a lower-bound exemplar analysis without pretending to be full masks. Hyperparameters and thresholds are selected on validation data, final test scoring is locked, and uncertainty is estimated by subject-cluster bootstrap with multiplicity-corrected paired tests. This public site contains aggregate results and synthetic illustrations only. Licensed biometric samples, per-image records and trained weights are not distributed.</p></div></section>
<section id="problem" class="section"><div class="wrap"><h2>The problem</h2><p class="section-intro">Background texture can create false ridge evidence; overly tight masks can remove usable minutiae. The benchmark measures both missed foreground and false background detection while preserving the distinction between an examiner ROI and a usable-ridge map.</p>{synthetic_panel}</div></section>
<section id="pipeline" class="section"><div class="wrap"><h2>Pipeline</h2><p class="section-intro">All methods share the same study units, split, physical resolution, validation boundary and evaluator.</p><img class="pipeline" src="assets/pipeline.svg" alt="Benchmark pipeline from licensed data through examiner GT, subject split, methods, evaluation and statistics"></div></section>
<section class="section"><div class="wrap"><h2>Datasets and ground truth</h2><div class="grid"><article class="card dataset"><h3>SD302-Latent</h3><p>Examiner ROI polygons define Table A; complete ridge-quality maps define Table A-prime.</p><p class="meta">136 / 24 / 40 subjects · train / validation / test</p></article><article class="card dataset"><h3>SD302-Exemplar</h3><p>Expert minutiae form a lower-bound convex hull. B1 reports MC, HR, PAR and OHF—never Dice against the hull.</p></article><article class="card dataset"><h3>Exclusions</h3><p>68 incomplete quality annotations affect A-prime only; two NONPRINT annotations are excluded from test scoring.</p></article></div></div></section>
<section id="methods" class="section"><div class="wrap"><h2>Methods</h2><p class="section-intro">Sixteen explicit rows represent the twelve Fig. 2 entries. The dagger marks the declared two-dimensional substitute.</p><div class="grid">{method_cards(results)}</div></div></section>
<section id="results" class="section"><div class="wrap"><h2>Interactive results</h2><p class="section-intro">The table and plots are generated from the tracked aggregate CSV. Empty cells are deliberately rendered as pending.</p><div class="tabs" role="tablist" aria-label="Result tables">{tabs}</div><div id="result-panel" role="tabpanel" aria-live="polite"><div class="controls"><label>Sort metric <select id="metric"><option value="dice">Dice</option><option value="jaccard">Jaccard</option><option value="mdr">MDR</option><option value="fdr">FDR</option><option value="accuracy">Accuracy</option><option value="ms_per_img">ms/img</option></select></label><span>Confidence intervals appear after aggregate finalisation.</span></div><div class="table-scroll"><table><thead><tr><th>Approach</th><th>Family</th><th>Status</th><th>Metric</th><th>Seeds</th><th>Params M</th><th>ms/img</th><th>Device</th></tr></thead><tbody id="result-body"></tbody></table></div><h3>Dice–runtime trade-off</h3><div id="pareto" class="chart"></div><h3>Corrected significance</h3><p class="notice">Holm-corrected paired comparisons remain pending the final aggregate bundle.</p></div></div></section>
<section class="section"><div class="wrap"><h2>Findings</h2><ul><li>Pending: deep-versus-classical Dice difference and subject-cluster CI.</li><li>Pending: examiner-value effect.</li><li>Pending: runtime–Dice Pareto frontier.</li><li>Pending: exemplar cross-domain hull-recall change.</li></ul></div></section>
<section id="reproduce" class="section"><div class="wrap"><h2>Reproducibility</h2><p class="section-intro">No script downloads biometric data. Supply locally authorised paths and frozen receipts.</p><pre>python -m pip install -r requirements.txt
python scripts/smoke_synthetic.py --work-dir .smoke
bash scripts/reproduce_table_A.sh --config local/benchmark.env
python site/build.py
python scripts/check_site_numbers.py
python scripts/scan_release.py</pre></div></section>
<section class="section"><div class="wrap"><h2>Limitations</h2><p class="section-intro">Reimplementations can differ from original code; SegFormer is a declared substitute; released zero-shot models have external training provenance; ROI and usable-ridge GT answer different questions; exemplar hulls are lower bounds; and one coordinated data source cannot establish every deployment domain.</p></div></section>
<section id="citation" class="section"><div class="wrap"><h2>Citation</h2><pre id="bibtex">@software{{narayan2026latentseg,
  author = {{Narayan, Udit and {{TODO: co-authors}}}},
  title = {{Latent Fingerprint Segmentation Benchmark}},
  year = {{2026}},
  url = {{https://github.com/Udiation/latent-fingerprint-segmentation-benchmark}}
}}</pre><button id="copy-cite" class="button">Copy BibTeX</button><h3>Acknowledgements</h3><p>NIST SD302; pyfing; the FingerNet authors; and the Hugging Face SegFormer implementation.</p></div></section></main><footer class="footer"><div class="wrap">Aggregate results and synthetic examples only · no tracking or analytics</div></footer>
<script id="public-data" type="application/json">{payload_json}</script><script src="assets/app.js"></script></body></html>'''
    (SITE / "index.html").write_text(html, encoding="utf-8")
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (RESULTS / "public_results.csv", RESULTS / "table_status.csv", RESULTS / "statistics.csv")}
    manifest = {"schema_version": 1, "status": "PASS", "source_sha256": source_hashes, "result_rows": len(results), "synthetic_examples": len(synthetic)}
    (SITE / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
