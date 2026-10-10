"""Authenticated aggregate statistics for the SD302-Exemplar lower-bound endpoint.

This module operates only on already-scored ``evaluate_hull.py`` CSVs.  It
never reads an exemplar, hull, minutia, probability, mask, or checkpoint.  The
per-image CSVs remain private; only aggregate rows and paired statistics leave
the restricted report transaction.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


METRICS = ("mc", "hr", "par", "ohf")
REFERENCE_METHODS = ("all_background", "all_foreground", "hull_as_prediction")
PRODUCTION_PROTOCOL = "cureblock_sd302g_b1_v1"
PRODUCTION_METHODS = frozenset({
    "gmfs_default", "gmfs_tuned", "sufs", "sufs_ft", "fingernet",
    "svm", "rf", "welm", "unet", "unetpp", "drunet", "ra_runet",
    "lenet", "alexnet", "cnn_ae", "segformer_sub", "ra_runet_aligned",
})
PRODUCTION_UNSEEDED_METHODS = frozenset({
    "gmfs_default", "gmfs_tuned", "sufs", "fingernet",
})
REQUIRED_PER_IMAGE = {
    "method", "display_name", "image_id", "subject_id", "split", "group",
    "device", "capture", "verification_status", "n_minutiae",
    "minutiae_covered", "hull_pixels", "hull_covered", "predicted_pixels",
    "outside_hull_pixels", "mc", "hr", "par", "ohf", "missing",
}


class B1StatisticsError(ValueError):
    """B1 evidence cannot support the requested aggregate claim."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise B1StatisticsError(message)


def validate_universe(
    payload: Mapping[str, Any], *, require_production: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Validate the authenticated, full applicable B1 presentation universe."""
    require(
        payload.get("schema_version") == 1
        and payload.get("kind") == "sd302g_b1_method_universe_v1",
        "B1 method universe schema/kind changed",
    )
    methods = payload.get("methods")
    require(isinstance(methods, list) and methods, "B1 method universe is empty")
    result: Dict[str, Dict[str, Any]] = {}
    for raw in methods:
        require(isinstance(raw, Mapping), "B1 method-universe row is malformed")
        key = str(raw.get("table_method", "")).strip()
        status = str(raw.get("status", "")).strip()
        reason = str(raw.get("reason", "")).strip()
        seed_regime = str(raw.get("seed_regime", "")).strip()
        require(key and key not in result, "B1 method universe has an empty/duplicate key")
        require(key not in REFERENCE_METHODS,
                f"{key}: deterministic reference rows are not B1 method-universe members")
        require(status in {"score", "N/A"}, f"{key}: invalid B1 universe status")
        require(status == "score" or reason, f"{key}: N/A requires an exact reason")
        require(status != "score" or not reason, f"{key}: scored row must not carry an N/A reason")
        require(seed_regime in {"unseeded", "seeded"},
                f"{key}: B1 seed_regime must be unseeded or seeded")
        result[key] = {
            "table_method": key,
            "display_name": str(raw.get("display_name", key)).strip() or key,
            "status": status,
            "reason": reason,
            "year": raw.get("year", ""),
            "family": str(raw.get("family", "Unclassified")),
            "training_data": str(raw.get("training_data", "unspecified")),
            "comment": str(raw.get("comment", "")),
            "seed_regime": seed_regime,
        }
    protocol = str(payload.get("protocol", ""))
    if require_production or protocol == PRODUCTION_PROTOCOL:
        require(protocol == PRODUCTION_PROTOCOL,
                "B1 delivery does not declare the production protocol")
        require(set(result) == PRODUCTION_METHODS,
                "B1 production method universe is incomplete or contains extra rows")
        require(
            {key for key, row in result.items() if row["seed_regime"] == "unseeded"}
            == PRODUCTION_UNSEEDED_METHODS,
            "B1 production seed regimes differ from the frozen method policy",
        )
    else:
        require(payload.get("synthetic_test_only") is True,
                "non-production B1 universes are permitted only in synthetic tests")
    statistics = payload.get("statistics")
    require(isinstance(statistics, Mapping), "B1 method universe lacks statistics policy")
    require(
        type(statistics.get("bootstrap_seed")) is int
        and statistics.get("bootstrap_replicates") == 1000
        and statistics.get("bootstrap_unit") == "subject"
        and statistics.get("paired") is True,
        "B1 subject-bootstrap policy changed",
    )
    require(
        statistics.get("leader_metric") == "hr"
        and statistics.get("leader_direction") == "maximum"
        and statistics.get("wilcoxon_alternative") == "two-sided"
        and statistics.get("holm_alpha") == 0.05
        and statistics.get("references_in_family") is False,
        "B1 Wilcoxon/Holm policy changed",
    )
    require(
        statistics.get("fixed_comparisons") == len(result) - 1,
        "B1 fixed Holm family differs from the method universe",
    )
    return result


def validate_per_image(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    """Validate and canonicalize one adopted method's private per-image rows."""
    missing_columns = sorted(REQUIRED_PER_IMAGE - set(frame.columns))
    require(not missing_columns, f"{method}: B1 per-image columns missing: {missing_columns}")
    selected = frame[frame.method.astype(str) == method].copy()
    require(not selected.empty, f"{method}: no method-specific B1 rows")
    require(set(selected.split.astype(str)) == {"test"}, f"{method}: non-test B1 rows")
    require(set(selected.verification_status.astype(str)) == {"REFERENCE_VERIFIED"},
            f"{method}: coordinates are not reference verified")
    require(not pd.to_numeric(selected.missing, errors="coerce").fillna(1).astype(bool).any(),
            f"{method}: missing B1 prediction")
    require(
        selected.image_id.notna().all() and selected.subject_id.notna().all(),
        f"{method}: missing B1 image/subject identity",
    )
    selected["image_id"] = selected.image_id.astype(str)
    selected["subject_id"] = selected.subject_id.astype(str)
    require(
        selected.image_id.str.strip().ne("").all()
        and selected.subject_id.str.strip().ne("").all(),
        f"{method}: empty B1 image/subject identity",
    )
    require(selected.image_id.is_unique, f"{method}: duplicate B1 image identity")
    count_columns = (
        "n_minutiae", "minutiae_covered", "hull_pixels", "hull_covered",
        "predicted_pixels", "outside_hull_pixels",
    )
    counts: Dict[str, np.ndarray] = {}
    for column in count_columns:
        values = pd.to_numeric(selected[column], errors="coerce").to_numpy(float)
        require(
            np.isfinite(values).all() and np.all(values >= 0)
            and np.equal(values, np.floor(values)).all(),
            f"{method}: invalid B1 count column {column}",
        )
        counts[column] = values
    require(
        np.all(counts["n_minutiae"] > 0)
        and np.all(counts["hull_pixels"] > 0)
        and np.all(counts["minutiae_covered"] <= counts["n_minutiae"])
        and np.all(counts["hull_covered"] <= counts["hull_pixels"])
        and np.all(counts["outside_hull_pixels"] <= counts["predicted_pixels"])
        and np.array_equal(
            counts["predicted_pixels"],
            counts["hull_covered"] + counts["outside_hull_pixels"],
        ),
        f"{method}: inconsistent B1 evaluator counts",
    )
    for metric in METRICS:
        values = pd.to_numeric(selected[metric], errors="coerce").to_numpy(float)
        require(np.isfinite(values).all() and np.all(values >= 0),
                f"{method}: invalid B1 {metric}")
        if metric != "par":
            require(np.all(values <= 1), f"{method}: out-of-range B1 {metric}")
        selected[metric] = values
    expected = {
        "mc": counts["minutiae_covered"] / counts["n_minutiae"],
        "hr": counts["hull_covered"] / counts["hull_pixels"],
        "par": counts["predicted_pixels"] / counts["hull_pixels"],
        "ohf": np.divide(
            counts["outside_hull_pixels"], counts["predicted_pixels"],
            out=np.zeros_like(counts["outside_hull_pixels"]),
            where=counts["predicted_pixels"] != 0,
        ),
    }
    for metric, values in expected.items():
        require(np.allclose(selected[metric].to_numpy(float), values, rtol=0, atol=1e-12),
                f"{method}: B1 {metric} does not match evaluator counts")
    return selected.sort_values("image_id", kind="stable").reset_index(drop=True)


def _draws(seed: int, replicates: int, n_subjects: int) -> np.ndarray:
    require(n_subjects >= 1, "B1 subject bootstrap needs at least one subject")
    if n_subjects == 1:
        # A one-cluster subgroup has a degenerate, explicitly non-informative
        # interval.  Keeping it avoids borrowing subjects outside that device
        # or capture subgroup, which would be a worse statistical claim.
        return np.zeros((replicates, 1), dtype=np.int64)
    return np.random.Generator(np.random.PCG64(seed)).integers(
        0, n_subjects, size=(replicates, n_subjects), dtype=np.int64
    )


def _derived_seed(seed: int, aggregate: str, value: str) -> int:
    """Derive a stable seed for subgroup bootstraps without Python hash state."""
    if aggregate == "overall" and value == "all":
        return seed
    digest = hashlib.sha256(
        f"{seed}\0{aggregate}\0{value}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _cluster_interval(
    frame: pd.DataFrame,
    metric: str,
    subjects: Sequence[str],
    draws: np.ndarray,
) -> tuple[float, float]:
    index = {subject: offset for offset, subject in enumerate(subjects)}
    sums = np.zeros(len(subjects), dtype=np.float64)
    counts = np.zeros(len(subjects), dtype=np.int64)
    for subject, value in zip(frame.subject_id.astype(str), frame[metric].astype(float)):
        offset = index[subject]
        sums[offset] += value
        counts[offset] += 1
    denominators = counts[draws].sum(axis=1)
    require(np.all(denominators > 0), f"B1 bootstrap has empty {metric} replicate")
    values = sums[draws].sum(axis=1) / denominators
    low, high = np.quantile(values, [0.025, 0.975], method="linear")
    return float(low), float(high)


def aggregate_seed_frames(
    frames: Mapping[str, Sequence[tuple[Optional[int], pd.DataFrame]]],
    universe: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> tuple[list[Dict[str, Any]], Dict[str, pd.DataFrame], np.ndarray, list[str]]:
    """Average seeds per image, then aggregate with paired subject draws."""
    successful = {key for key, runs in frames.items() if runs}
    expected = {key for key, row in universe.items() if row["status"] == "score"}
    require(successful == expected,
            f"B1 scored method universe differs: expected {sorted(expected)}, got {sorted(successful)}")
    canonical: Dict[str, pd.DataFrame] = {}
    common_images: Optional[tuple[str, ...]] = None
    common_subjects: Optional[list[str]] = None
    common_row_metadata: Optional[Dict[str, tuple[str, ...]]] = None
    seed_ids: Dict[str, list[Optional[int]]] = {}
    for method in sorted(successful):
        runs = list(frames[method])
        seeds = [seed for seed, _ in runs]
        require(len(seeds) == len(set(seeds)), f"{method}: duplicate B1 seed")
        if universe[method]["seed_regime"] == "unseeded":
            require(seeds == [None],
                    f"{method}: frozen unseeded B1 endpoint requires exactly one unseeded run")
        else:
            require(
                all(type(seed) is int and seed in {0, 1, 2} for seed in seeds)
                and 0 in seeds,
                f"{method}: frozen seeded B1 endpoint requires a unique subset of seeds 0, 1, 2 including 0",
            )
        base = runs[0][1][
            ["image_id", "subject_id", "group", "device", "capture", "n_minutiae",
             "hull_pixels"]
        ].copy()
        identity = tuple(base.image_id.astype(str))
        subjects_for_rows = tuple(base.subject_id.astype(str))
        for _, frame in runs[1:]:
            require(tuple(frame.image_id.astype(str)) == identity,
                    f"{method}: B1 seed image universe changed")
            require(tuple(frame.subject_id.astype(str)) == subjects_for_rows,
                    f"{method}: B1 seed subject pairing changed")
            for column in ("group", "device", "capture", "n_minutiae", "hull_pixels"):
                require(tuple(frame[column].astype(str)) == tuple(base[column].astype(str)),
                        f"{method}: B1 seed metadata changed: {column}")
        if common_images is None:
            common_images = identity
            common_subjects = sorted(set(subjects_for_rows))
            common_row_metadata = {
                column: tuple(base[column].astype(str))
                for column in (
                    "subject_id", "group", "device", "capture", "n_minutiae",
                    "hull_pixels",
                )
            }
        require(identity == common_images, f"{method}: B1 image universe differs across methods")
        require(sorted(set(subjects_for_rows)) == common_subjects,
                f"{method}: B1 subject universe differs across methods")
        assert common_row_metadata is not None
        for column, expected_values in common_row_metadata.items():
            require(tuple(base[column].astype(str)) == expected_values,
                    f"{method}: B1 paired metadata differs across methods: {column}")
        for metric in METRICS:
            matrix = np.stack([frame[metric].to_numpy(float) for _, frame in runs])
            base[metric] = matrix.mean(axis=0)
        canonical[method] = base
        seed_ids[method] = seeds
    require(common_subjects is not None, "B1 has no authenticated per-image evidence")
    draws = _draws(
        int(policy["bootstrap_seed"]), int(policy["bootstrap_replicates"]), len(common_subjects)
    )
    rows: list[Dict[str, Any]] = []
    for method, meta in universe.items():
        if meta["status"] == "N/A":
            row: Dict[str, Any] = {
                "method": method, "display_name": meta["display_name"], "split": "test",
                "verification_status": "NOT_SCORED", "aggregate": "overall", "value": "all",
                "status": "N/A", "reason": meta["reason"], "n_seeds": 0, "seed_ids": "",
                "n_images": len(common_images or ()), "n_subjects": len(common_subjects),
                "n_minutiae": 0, "n_missing": 0,
                "bootstrap_seed": int(policy["bootstrap_seed"]),
                "bootstrap_replicates": int(policy["bootstrap_replicates"]),
                "bootstrap_unit": "subject", "bootstrap_status": "unavailable",
            }
            for metric in METRICS:
                for suffix in ("mean", "std", "seed_std", "ci_low", "ci_high", "pooled"):
                    row[f"{metric}_{suffix}"] = np.nan
            rows.append(row)
            continue
        frame = canonical[method]
        selectors = [("overall", "all", frame)]
        for column in ("group", "device", "capture"):
            selectors.extend(
                (column, value, frame[frame[column].astype(str) == value])
                for value in sorted(set(frame[column].astype(str)))
            )
        runs = list(frames[method])
        for aggregate, value, part in selectors:
            part_subjects = sorted(set(part.subject_id.astype(str)))
            part_seed = _derived_seed(int(policy["bootstrap_seed"]), aggregate, str(value))
            part_draws = _draws(
                part_seed, int(policy["bootstrap_replicates"]), len(part_subjects)
            )
            row = {
                "method": method, "display_name": meta["display_name"], "split": "test",
                "verification_status": "REFERENCE_VERIFIED", "aggregate": aggregate,
                "value": value, "status": "ok", "reason": "", "n_seeds": len(runs),
                "seed_ids": ",".join("unseeded" if seed is None else str(seed) for seed, _ in runs),
                "n_images": len(part), "n_subjects": int(part.subject_id.nunique()),
                "n_minutiae": int(part.n_minutiae.sum()), "n_missing": 0,
                "bootstrap_seed": part_seed,
                "bootstrap_replicates": int(policy["bootstrap_replicates"]),
                "bootstrap_unit": "subject", "bootstrap_status": "ok",
            }
            part_ids = set(part.image_id.astype(str))
            for metric in METRICS:
                values = part[metric].to_numpy(float)
                low, high = _cluster_interval(part, metric, part_subjects, part_draws)
                seed_means = [
                    float(run[run.image_id.astype(str).isin(part_ids)][metric].mean())
                    for _, run in runs
                ]
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=0))
                row[f"{metric}_seed_std"] = (
                    float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else np.nan
                )
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
                # Pooled counts are intentionally not reconstructed after seed
                # averaging.  A macro endpoint is the registered estimand.
                row[f"{metric}_pooled"] = np.nan
            rows.append(row)
    return rows, canonical, draws, common_subjects


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if not len(nonzero):
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    return float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())


def paired_hr_statistics(
    canonical: Mapping[str, pd.DataFrame],
    universe: Mapping[str, Mapping[str, Any]],
    subjects: Sequence[str],
    draws: np.ndarray,
    policy: Mapping[str, Any],
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Compute the fixed B1 leader comparison family or explain unavailability."""
    available = sorted(canonical)
    if len(available) < 2:
        reason = "fewer than two B1 methods have authenticated paired per-image evidence"
        return ([{
            "leader": available[0] if available else "", "comparator": "",
            "status": "UNAVAILABLE", "reason": reason, "metric": "hr",
            "n_images": 0, "n_subjects": len(subjects), "statistic": np.nan,
            "raw_p": np.nan, "holm_p": np.nan, "significant": False,
            "mean_paired_difference": np.nan, "difference_ci_low": np.nan,
            "difference_ci_high": np.nan, "rank_biserial": np.nan,
        }], {"status": "UNAVAILABLE", "reason": reason})
    means = {method: float(frame.hr.mean()) for method, frame in canonical.items()}
    maximum = max(means.values())
    leader = min(method for method, value in means.items() if value == maximum)
    leader_frame = canonical[leader]
    records: list[Dict[str, Any]] = []
    for comparator in sorted(method for method in universe if method != leader):
        if comparator not in canonical:
            records.append({
                "leader": leader, "comparator": comparator, "status": "N/A",
                "reason": universe[comparator]["reason"], "metric": "hr",
                "n_images": 0, "n_subjects": len(subjects), "statistic": np.nan,
                "raw_p": np.nan, "holm_p": np.nan, "significant": False,
                "mean_paired_difference": np.nan, "difference_ci_low": np.nan,
                "difference_ci_high": np.nan, "rank_biserial": np.nan,
            })
            continue
        frame = canonical[comparator]
        require(tuple(frame.image_id) == tuple(leader_frame.image_id),
                f"{comparator}: B1 Wilcoxon pairing changed")
        differences = leader_frame.hr.to_numpy(float) - frame.hr.to_numpy(float)
        if np.all(differences == 0):
            statistic, raw_p, status = 0.0, 1.0, "all_zero"
        else:
            result = wilcoxon(
                differences, alternative="two-sided", zero_method="wilcox",
                correction=False, method="approx",
            )
            statistic, raw_p, status = float(result.statistic), float(result.pvalue), "ok"
        diff = leader_frame[["image_id", "subject_id"]].copy()
        diff["hr"] = differences
        low, high = _cluster_interval(diff, "hr", subjects, draws)
        records.append({
            "leader": leader, "comparator": comparator, "status": status, "reason": "",
            "metric": "hr", "n_images": len(differences), "n_subjects": len(subjects),
            "statistic": statistic, "raw_p": raw_p, "holm_p": np.nan,
            "significant": False, "mean_paired_difference": float(differences.mean()),
            "difference_ci_low": low, "difference_ci_high": high,
            "rank_biserial": _rank_biserial(differences),
        })
    family = int(policy["fixed_comparisons"])
    finite = sorted(
        (row for row in records if math.isfinite(float(row["raw_p"]))),
        key=lambda row: (float(row["raw_p"]), str(row["comparator"])),
    )
    running = 0.0
    for index, row in enumerate(finite):
        adjusted = min(1.0, (family - index) * float(row["raw_p"]))
        running = max(running, adjusted)
        row["holm_p"] = running
        row["significant"] = running <= float(policy["holm_alpha"])
    return records, {"status": "COMPLETE", "reason": "", "leader": leader}
