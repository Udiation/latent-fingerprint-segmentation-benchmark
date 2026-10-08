"""Pixel-level segmentation metrics, matching the columns of Table 1 in
Dhaneshwar & Wadhera (2026): Accuracy, Error rate, Dice, Jaccard, MIoU, MDR, FDR.

Foreground = fingerprint (positive class).
  MDR (missed detection rate) = foreground pixels labelled background / all foreground pixels
  FDR (false detection rate)  = background pixels labelled foreground / all background pixels
These follow the segmentation literature (e.g. Ezeobiejesi & Bhanu 2016); note that
this FDR is NOT the statistical "false discovery rate".
"""
import numpy as np

METRIC_KEYS = ["accuracy", "error_rate", "dice", "jaccard", "miou", "mdr", "fdr"]


def confusion(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch pred {pred.shape} vs gt {gt.shape}")
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    tn = int(pred.size - tp - fp - fn)
    return tp, fp, fn, tn


def _ratio(num, den, empty):
    return num / den if den else empty


def metrics_from_confusion(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    acc = (tp + tn) / n
    jac_fg = _ratio(tp, tp + fp + fn, 1.0)  # both empty -> perfect agreement
    jac_bg = _ratio(tn, tn + fp + fn, 1.0)
    return {
        "accuracy": acc,
        "error_rate": 1.0 - acc,
        "dice": _ratio(2 * tp, 2 * tp + fp + fn, 1.0),
        "jaccard": jac_fg,
        "miou": (jac_fg + jac_bg) / 2.0,
        "mdr": _ratio(fn, tp + fn, float("nan")),
        "fdr": _ratio(fp, fp + tn, float("nan")),
    }


def image_metrics(pred, gt):
    tp, fp, fn, tn = confusion(pred, gt)
    out = metrics_from_confusion(tp, fp, fn, tn)
    out.update({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return out


def summarise(per_image_df):
    """Mean and std over images (macro) plus pooled-pixel (micro) values."""
    row = {"n_images": int(len(per_image_df))}
    for k in METRIC_KEYS:
        row[f"{k}_mean"] = float(per_image_df[k].mean())
        row[f"{k}_std"] = float(per_image_df[k].std(ddof=0))
    pooled = metrics_from_confusion(
        int(per_image_df.tp.sum()), int(per_image_df.fp.sum()),
        int(per_image_df.fn.sum()), int(per_image_df.tn.sum()))
    for k in METRIC_KEYS:
        row[f"{k}_pooled"] = float(pooled[k])
    if "ms_per_img" in per_image_df.columns:
        row["ms_per_img"] = float(per_image_df["ms_per_img"].mean())
    return row
