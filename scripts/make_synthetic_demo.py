"""SYNTHETIC data only - for smoke-testing the pipeline anywhere (safe for a public
repo). Draws ridge-like patterns inside random ellipses on cluttered backgrounds."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_print(rng, h, w, noisy):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = rng.uniform(0.35, 0.65) * h, rng.uniform(0.35, 0.65) * w
    ay, ax = rng.uniform(0.2, 0.35) * h, rng.uniform(0.18, 0.3) * w
    mask = ((yy - cy) / ay) ** 2 + ((xx - cx) / ax) ** 2 <= 1.0
    theta = np.arctan2(yy - cy, xx - cx)
    r = np.hypot(yy - cy, xx - cx)
    phase = 2 * np.pi * r / rng.uniform(8, 11) + 0.6 * np.sin(2 * theta)
    ridges = 0.5 + 0.5 * np.cos(phase)
    bg = rng.normal(0.75, 0.05, (h, w))
    if noisy:  # latent-like: low contrast, smudges, stray lines
        bg = cv2.GaussianBlur(rng.normal(0.6, 0.2, (h, w)), (0, 0), 6) + rng.normal(0, 0.06, (h, w))
        for _ in range(4):
            p1 = tuple(int(v) for v in rng.integers(0, [w, h]))
            p2 = tuple(int(v) for v in rng.integers(0, [w, h]))
            cv2.line(bg, p1, p2, float(rng.uniform(0.1, 0.4)), int(rng.integers(2, 6)))
    contrast = 0.25 if noisy else 0.6
    img = np.where(mask, bg * (1 - contrast) + contrast * (1 - ridges) * 0.5 + 0.1, bg)
    img = np.clip(img + rng.normal(0, 0.03, (h, w)), 0, 1)
    return (img * 255).astype(np.uint8), mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", type=int, default=30)
    ap.add_argument("--per-subject", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    rows = []
    specs = {"finger_latent": (420, 380, True, 1000),
             "finger_livescan": (300, 280, False, 500),
             "palm": (520, 480, False, 500)}
    for s in range(a.subjects):
        for subset, (h, w, noisy, ppi) in specs.items():
            for k in range(a.per_subject if subset != "palm" else 1):
                img, mask = make_print(rng, h, w, noisy)
                iid = f"syn_{subset}_{s:03d}_{k}"
                (out / "images").mkdir(parents=True, exist_ok=True)
                (out / "masks").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / "images" / f"{iid}.png"), img)
                cv2.imwrite(str(out / "masks" / f"{iid}.png"), mask.astype(np.uint8) * 255)
                rows.append({"image_id": iid, "image_path": str(out / "images" / f"{iid}.png"),
                             "mask_path": str(out / "masks" / f"{iid}.png"),
                             "subject_id": f"S{s:03d}", "subset": subset, "ppi": ppi})
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    assets = []
    for role, directory in (("example_image", "images"), ("ground_truth_mask", "masks")):
        for path in sorted((out / directory).glob("*.png")):
            assets.append({
                "path": path.relative_to(out).as_posix(),
                "sha256": sha256_file(path),
                "mime_type": "image/png",
                "role": role,
            })
    receipt = {
        "schema_version": 1,
        "kind": "segbench_synthetic_dataset",
        "provenance": "generated_by_make_synthetic_demo",
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "seed": a.seed,
        "subjects": a.subjects,
        "per_subject": a.per_subject,
        "assets": assets,
    }
    (out / "synthetic_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} synthetic images to {out}")


if __name__ == "__main__":
    main()
