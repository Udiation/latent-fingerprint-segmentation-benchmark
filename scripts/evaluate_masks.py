"""Score masks produced by any other method (SUFS, GMFS, U-Net, FingerNet, ...)
with exactly the same split, resolution and metrics as run_classical.py.

--pred-template is a path pattern with {image_id} (and optionally {subset}), e.g.
  private_data/seg_out/sufs/{image_id}.png
Predicted masks are assumed to be at the image's native resolution (manifest ppi)
unless --pred-ppi is given. A missing prediction is scored as an all-background
mask (counted in n_missing) so a method cannot look better by skipping hard images.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import _path  # noqa: F401
from segbench.io import (
    attach_split,
    load_manifest,
    manifest_target_shape,
    mask_ppi_for,
    match_shape,
    read_mask,
    to_target_ppi,
)
from segbench.metrics import image_metrics, summarise
from segbench.summary import MASK_EVALUATION_KEY_COLUMNS, upsert_summary


def evaluation_identity_suffix(run_id, training_data):
    """Namespace per-image evidence for alternate runs/training regimes."""
    if not run_id and not training_data:
        return ""
    encoded = json.dumps(
        {"run_id": run_id, "training_data": training_data},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "_identity-" + hashlib.sha256(encoded).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--subsets", nargs="+", required=True)
    ap.add_argument("--method", required=True, help="short key, e.g. sufs")
    ap.add_argument("--display-name", required=True, help='e.g. "SUFS (Cappelli 2023)"')
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--pred-template")
    source.add_argument(
        "--reference",
        choices=("all-background", "all-foreground"),
        help="generate a trivial shape-only reference inside the common evaluator",
    )
    ap.add_argument("--pred-ppi", type=float, default=0.0)
    ap.add_argument("--ms-per-img", type=float, default=float("nan"),
                    help="mean runtime measured when the masks were produced, if known")
    ap.add_argument("--target-ppi", type=float, default=500.0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--gt-column", default="mask_path",
                    help="manifest column containing the ground-truth mask")
    ap.add_argument("--group-by", default="",
                    help="optional manifest column for additional aggregate rows")
    ap.add_argument("--exclude", action="append", default=[], metavar="COLUMN=VALUE",
                    help="exclude rows matching a manifest value; repeatable")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--year", default="")
    ap.add_argument("--family", default="")
    ap.add_argument("--training-data", default="")
    ap.add_argument("--comment", default="")
    ap.add_argument("--params-m", type=float, default=float("nan"))
    ap.add_argument("--device", choices=("CPU", "GPU"), default="CPU")
    ap.add_argument("--threshold", type=float, default=float("nan"))
    ap.add_argument("--run-id", default="")
    ap.add_argument("--table", default="", help="table key, e.g. A or A_prime")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.splits) as f:
        splits = json.load(f)["splits"]
    df = attach_split(load_manifest(a.manifest), splits)
    if a.gt_column not in df.columns:
        raise ValueError(f"manifest has no ground-truth column {a.gt_column!r}")
    df[a.gt_column] = df[a.gt_column].fillna("").astype(str)
    df = df[(df[a.gt_column].str.len() > 0) & (df.split == a.split)]
    for expression in a.exclude:
        if "=" not in expression:
            raise ValueError(f"--exclude must be COLUMN=VALUE, got {expression!r}")
        column, value = expression.split("=", 1)
        if column not in df.columns:
            raise ValueError(f"manifest has no exclusion column {column!r}")
        df = df[df[column].fillna("").astype(str) != value]
    if a.group_by and a.group_by not in df.columns:
        raise ValueError(f"manifest has no grouping column {a.group_by!r}")

    def safe(value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "none"

    def metadata(group_value="overall"):
        return {
            "method": a.method,
            "display_name": a.display_name,
            "split": a.split,
            "train_subset": "pretrained/none",
            "ground_truth": a.gt_column,
            "table": a.table,
            "group_by": a.group_by,
            "group_value": group_value,
            "seed": a.seed,
            "year": a.year,
            "family": a.family,
            "training_data": a.training_data,
            "comment": a.comment,
            "params_m": a.params_m,
            "device": a.device,
            "threshold": a.threshold,
            "run_id": a.run_id,
        }

    for subset in a.subsets:
        recs, missing = [], 0
        subset_df = df[df.subset == subset]
        if a.limit:
            subset_df = subset_df.head(a.limit)
        for r in subset_df.to_dict("records"):
            gt = to_target_ppi(
                read_mask(r[a.gt_column]),
                mask_ppi_for(r, a.gt_column),
                a.target_ppi,
                is_mask=True,
            )
            target_shape = manifest_target_shape(r, a.target_ppi) or gt.shape
            gt = match_shape(gt, target_shape)
            p = None
            if a.reference == "all-background":
                pred = np.zeros(target_shape, dtype=bool)
            elif a.reference == "all-foreground":
                pred = np.ones(target_shape, dtype=bool)
            else:
                p = Path(a.pred_template.format(image_id=r["image_id"], subset=subset))
                if p.exists():
                    pred = read_mask(p)
                    pred = to_target_ppi(
                        pred, a.pred_ppi or r["ppi"], a.target_ppi, is_mask=True
                    )
                    pred = match_shape(pred, target_shape)
                else:
                    missing += 1
                    pred = np.zeros_like(gt)
            rec = image_metrics(pred, gt)
            rec.update({
                "image_id": r["image_id"],
                "subject_id": r["subject_id"],
                "missing": p is not None and not p.exists(),
            })
            if a.group_by:
                rec[a.group_by] = r[a.group_by]
            recs.append(rec)
        if not recs:
            print(f"{subset}: no test images with ground truth")
            continue
        per = pd.DataFrame(recs)
        seed_suffix = f"_seed{a.seed}" if a.seed is not None else ""
        evaluation_key = safe(a.table or a.gt_column)
        identity_suffix = evaluation_identity_suffix(a.run_id, a.training_data)
        per_path = out_dir / (
            f"per_image_{safe(a.method)}_{safe(subset)}_{evaluation_key}"
            f"{seed_suffix}{identity_suffix}.csv"
        )
        per.to_csv(per_path, index=False)
        row = summarise(per)
        row.update(metadata())
        row.update({"subset": subset, "n_missing": missing, "ms_per_img": a.ms_per_img})
        upsert_summary(
            out_dir / "summary.csv",
            row,
            key_columns=MASK_EVALUATION_KEY_COLUMNS,
        )
        if a.group_by:
            for group_value, group in per.groupby(a.group_by, dropna=False):
                group_row = summarise(group)
                group_row.update(metadata(str(group_value)))
                group_row.update({
                    "subset": subset,
                    "n_missing": int(group["missing"].sum()),
                    "ms_per_img": a.ms_per_img,
                })
                upsert_summary(
                    out_dir / "summary.csv",
                    group_row,
                    key_columns=MASK_EVALUATION_KEY_COLUMNS,
                )
        print(f"{a.method} {subset}: n={len(per)} missing={missing} "
              f"Dice {row['dice_mean']:.4f} Acc {row['accuracy_mean']:.4f}")


if __name__ == "__main__":
    main()
