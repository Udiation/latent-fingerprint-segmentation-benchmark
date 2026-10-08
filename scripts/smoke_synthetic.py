#!/usr/bin/env python3
"""Run the end-to-end classical pipeline on generated, non-biometric data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    if work.exists() and not args.keep:
        shutil.rmtree(work)
    synthetic = work / "synthetic"
    split = work / "splits.json"
    output = work / "results"
    masks = work / "masks"
    run([sys.executable, "scripts/make_synthetic_demo.py", "--out", str(synthetic), "--subjects", "12", "--per-subject", "1", "--seed", "7"])
    run([sys.executable, "scripts/make_split.py", "--manifest", str(synthetic / "manifest.csv"), "--out", str(split), "--seed", "42"])
    run([sys.executable, "scripts/run_classical.py", "--manifest", str(synthetic / "manifest.csv"), "--splits", str(split), "--train-subset", "finger_latent", "--methods", "rf", "--out", str(output), "--blocks-per-class", "8", "--max-train-blocks", "2000", "--max-val-blocks", "500", "--tune-blocks", "500", "--n-jobs", "2", "--seed", "0"])
    model = output / "models" / "rf_finger_latent.joblib"
    training_card = output / "cards" / "rf_finger_latent.json"
    prediction_root = masks / "rf" / "finger_latent"
    run([sys.executable, "scripts/predict_classical.py", "--manifest", str(synthetic / "manifest.csv"), "--splits", str(split), "--model", str(model), "--training-card", str(training_card), "--method", "rf", "--display-name", "RF synthetic smoke", "--seed", "0", "--output-root", str(prediction_root), "--run-card", str(output / "rf_prediction.json"), "--warmup", "0", "--progress-every", "1"])
    run([sys.executable, "scripts/evaluate_masks.py", "--manifest", str(synthetic / "manifest.csv"), "--splits", str(split), "--subsets", "finger_latent", "--method", "rf_external", "--display-name", "RF synthetic smoke", "--pred-template", str(prediction_root / "{image_id}.png"), "--pred-ppi", "500", "--out", str(output)])
    run([sys.executable, "scripts/make_tables.py", "--summary", str(output / "summary.csv"), "--out", str(output), "--with-std"])
    summary = output / "summary.csv"
    if not summary.is_file() or not (output / "tables.md").is_file():
        raise SystemExit("synthetic smoke test did not produce tables")
    print(json.dumps({"status": "PASS", "summary": summary.name, "synthetic_only": True}, sort_keys=True))


if __name__ == "__main__":
    main()
