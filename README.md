# Latent Fingerprint Segmentation Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-2563eb.svg)](LICENSE)
[![CI](https://github.com/Udiation/latent-fingerprint-segmentation-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Udiation/latent-fingerprint-segmentation-benchmark/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v1.0_provisional-f59e0b.svg)](CHANGELOG.md)
[![DOI](https://img.shields.io/badge/DOI-pending-64748b.svg)](#citation)

A reproducible, subject-disjoint comparison of classical, patch-based,
encoder–decoder, transformer, and released fingerprint foreground segmenters.

> **Release status — v1.0 provisional snapshot.** Authenticated aggregate
> results are published where available. Any row marked `pending` is incomplete
> and is never inferred from validation metrics or another seed. A v1.1 update
> will regenerate the tables and website when the remaining frozen run finishes.

## TL;DR

- Twelve method entries from Fig. 2 of Dhaneshwar & Wadhera (2026) are mapped
  to sixteen explicit benchmark rows plus non-ranked trivial references.
- The main endpoint uses NIST SD302 latent images and examiner ROI annotations;
  usable-ridge and exemplar-hull endpoints test the sensitivity to GT choice.
- Subjects are disjoint across train, validation, and test. Learning rates,
  stopping epochs, thresholds, and post-processing are selected on validation
  data only; the test transaction is single-use for each frozen model.
- **Headline status:** this provisional snapshot reports completed aggregate
  rows and labels unfinished rows explicitly; no claim is made from a pending row.

## Headline results

Table A values are generated from `results/public_results.csv`. Completed cells
come from the authenticated aggregate snapshot; pending cells are intentionally
blank and will be replaced atomically by the v1.1 aggregate update.

<!-- BEGIN GENERATED HEADLINE RESULTS -->
| Family | Approach | Status | Dice | Jaccard | MDR | FDR | Seeds | Params (M) | ms/img (device) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Classical | GMFS (default) | complete | 75.91% | 62.61% | 25.06% | 43.72% | 1 | 0.00 | 15.62 (CPU) |
| Classical | GMFS (val-tuned) | complete | 79.58% | 67.19% | 15.56% | 60.93% | 1 | 0.00 | 15.48 (CPU) |
| Pretrained | SUFS (zero-shot) | complete | 74.95% | 61.70% | 27.00% | 39.13% | 1 | 7.08 | 21.22 (GPU) |
| Pretrained | SUFS (fine-tuned) | complete | 81.85% ± 0.72% | 70.91% ± 0.76% | 7.71% ± 1.57% | 74.34% ± 10.31% | 3 | 7.08 | 26.17 (GPU) |
| Pretrained | FingerNet (pretrained) | complete | 58.93% | 46.45% | 49.94% | 11.54% | 1 | 4.71 | 35.51 (GPU) |
| Classical | SVM (RBF via Nystrom) | complete | 75.85% ± 0.05% | 62.26% ± 0.07% | 27.91% ± 0.07% | 35.35% ± 0.15% | 3 | 1.01 | 41.57 (CPU) |
| Classical | Random forest | complete | 76.82% ± 0.04% | 63.40% ± 0.06% | 26.13% ± 0.09% | 37.35% ± 0.15% | 3 | 52.70 | 58.32 (CPU) |
| Classical | Weighted ELM | complete | 75.27% ± 0.19% | 61.57% ± 0.20% | 28.70% ± 0.22% | 35.10% ± 0.07% | 3 | 0.01 | 32.28 (CPU) |
| Encoder-decoder | U-Net | complete | 79.50% ± 1.18% | 68.21% ± 0.96% | 20.95% ± 1.68% | 32.78% ± 1.94% | 3 | 7.24 | 29.57 (GPU) |
| Encoder-decoder | UNet++ | complete | 78.22% ± 0.58% | 67.49% ± 0.90% | 21.92% ± 3.24% | 33.21% ± 9.46% | 3 | 9.16 | 42.36 (GPU) |
| Encoder-decoder | DRUnet | complete | 84.85% ± 0.57% | 74.39% ± 0.83% | 5.17% ± 1.45% | 64.75% ± 9.92% | 3 | 8.38 | 33.40 (GPU) |
| Encoder-decoder | RUnet (no alignment) | complete | 84.46% ± 0.33% | 73.97% ± 0.53% | 4.25% ± 1.74% | 70.20% ± 7.11% | 3 | 8.38 | 34.03 (GPU) |
| Patch CNN | LeNet | complete | 84.93% ± 0.18% | 74.46% ± 0.27% | 7.00% ± 0.83% | 58.00% ± 2.42% | 3 | 0.06 | 16.76 (GPU) |
| Patch CNN | AlexNet | complete | 86.46% ± 0.02% | 76.62% ± 0.03% | 8.68% ± 0.14% | 41.73% ± 0.40% | 3 | 2.91 | 19.22 (GPU) |
| Patch CNN | CNN + stacked AE | complete | 83.97% ± 0.07% | 73.08% ± 0.07% | 6.32% ± 0.86% | 69.41% ± 3.04% | 3 | 0.16 | 19.44 (GPU) |
| Transformer | SegFormer-B2 (substitute) | complete | 85.59% ± 1.96% | 75.93% ± 2.71% | 5.48% ± 2.49% | 60.77% ± 21.79% | 3 | 27.34 | 32.69 (GPU) |
| Reference | All background | complete | 0.00% | 0.00% | 100.00% | 0.00% | 1 | 0.00 | pending (CPU) |
| Reference | All foreground | complete | 82.98% | 71.84% | 0.00% | 100.00% | 1 | 0.00 | pending (CPU) |
<!-- END GENERATED HEADLINE RESULTS -->

### Table 4 — SD302-Exemplar live-scan

This table uses expert-minutiae hulls as a lower-bound reference, so Dice is
not reported. Only reference-verified aggregate rows contain values; every
unscored cross-domain row remains explicitly pending.

<!-- BEGIN GENERATED B1 RESULTS -->
| Approach | Status | MC | HR | PAR | OHF | Seeds | Params (M) | ms/img (device) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GMFS (default) | complete | 99.62% | 99.41% | 1.45 | 27.83% | 1 | 0.00 | 28.35 (CPU) |
| GMFS (val-tuned) | complete | 99.89% | 99.68% | 1.48 | 28.95% | 1 | 0.00 | 27.91 (CPU) |
| SUFS (zero-shot) | complete | 99.90% | 99.77% | 1.51 | 30.32% | 1 | 7.08 | 1142.45 (CPU) |
| SUFS (fine-tuned) | complete | 95.87% ± 1.85% | 96.11% ± 1.56% | 3.84 ± 0.41 | 71.10% ± 3.01% | 3 | 7.08 | 31.80 (GPU) |
| FingerNet (pretrained) | complete | 99.80% | 99.95% | 1.79 | 41.56% | 1 | 4.71 | 49.66 (GPU) |
| SVM (RBF via Nystrom) | complete | 5.59% ± 3.71% | 5.76% ± 3.76% | 0.13 ± 0.07 | 35.56% ± 4.44% | 3 | 1.01 | 292.81 (CPU) |
| Random forest | complete | 98.88% ± 0.44% | 98.98% ± 0.43% | 1.46 ± 0.02 | 28.53% ± 0.66% | 3 | 52.70 | 131.90 (CPU) |
| Weighted ELM | complete | 27.87% ± 7.29% | 28.88% ± 7.79% | 0.52 ± 0.09 | 42.32% ± 3.56% | 3 | 0.01 | 105.31 (CPU) |
| U-Net | complete | 91.74% ± 3.44% | 93.17% ± 3.14% | 3.39 ± 0.47 | 68.22% ± 3.21% | 3 | 7.24 | 54.61 (GPU) |
| UNet++ | complete | 91.77% ± 4.24% | 92.91% ± 3.74% | 2.60 ± 0.70 | 59.60% ± 7.32% | 3 | 9.16 | 73.13 (GPU) |
| DRUnet | complete | 98.41% ± 2.47% | 99.12% ± 1.40% | 2.48 ± 1.14 | 49.32% ± 23.40% | 3 | 8.38 | 62.89 (GPU) |
| RUnet (no alignment) | complete | 98.83% ± 1.63% | 99.08% ± 1.30% | 3.13 ± 1.23 | 59.92% ± 18.14% | 3 | 8.38 | 63.44 (GPU) |
| LeNet | complete | 99.94% ± 0.02% | 99.93% ± 0.04% | 1.68 ± 0.02 | 36.76% ± 0.56% | 3 | 0.06 | 23.85 (GPU) |
| AlexNet | complete | 99.33% ± 0.21% | 99.69% ± 0.12% | 4.26 ± 0.01 | 73.02% ± 0.05% | 3 | 2.91 | 37.69 (GPU) |
| CNN + stacked AE | complete | 99.09% ± 0.25% | 99.43% ± 0.24% | 4.25 ± 0.01 | 73.09% ± 0.03% | 3 | 0.16 | 35.56 (GPU) |
| SegFormer-B2 (substitute) | complete | 99.80% ± 0.31% | 99.83% ± 0.26% | 3.88 ± 0.22 | 70.55% ± 1.58% | 3 | 27.34 | 79.54 (GPU) |
| RA-RUnet (latent-to-live-scan alignment) | n/a | pending | pending | pending | pending | 0 | pending | pending |
| All background | complete | 0.00% | 0.00% | 0.00 | 0.00% | 0 | 0.00 | 0.00 (CPU) |
| All foreground | complete | 100.00% | 100.00% | 4.30 | 73.23% | 0 | 0.00 | 0.00 (CPU) |
| Hull as prediction | complete | 100.00% | 100.00% | 1.00 | 0.00% | 0 | 0.00 | 0.00 (CPU) |
<!-- END GENERATED B1 RESULTS -->

[Full result schema](results/README.md) · [Protocol](docs/PROTOCOL.md) ·
[Limitations](docs/LIMITATIONS.md)

## Overview

![Benchmark pipeline](site/assets/pipeline.svg)

The same subject assignment and resolution-normalised image representation feed
every method. All final masks pass through one evaluator, followed by paired
subject-aware uncertainty and significance analyses.

## Methods mapped from the review

| Fig. 2 entry | Benchmark row(s) | Source | Implementation note |
|---|---|---|---|
| CNN & autoencoders | CNN + stacked AE | Chhabra et al. (2023) | Train-only unsupervised patch pretraining |
| GMFS & SUFS | GMFS default/tuned; SUFS zero-shot/fine-tuned | Cappelli (2023) | Released pyfing algorithms and weights |
| U-Net | U-Net | Jaafar et al. (2022); Ronneberger et al. (2015) | Four-level reimplementation |
| Adaptive Gaussian transformer | SegFormer-B2 (substitute†) | Liu et al. (2022); Xie et al. (2021) | †The reviewed OCT model is not a 2-D fingerprint model |
| DRUnet | DRUnet | Joshi et al. | Recurrent blocks with deterministic table inference |
| Recurrent adversarial network | RUnet; RA-RUnet | Joshi et al. (2021) | Alignment is used only for the declared cross-domain setting |
| Deep Nested UNets | UNet++ | Liu & Qian (2020); Zhou et al. (2018) | Nested skips and deep supervision |
| Support Vector Machine | SVM | Cortes & Vapnik (1995) | Nyström RBF approximation over shared block features |
| AlexNet & LeNet | AlexNet; LeNet | Serafim et al. (2019) | Centred patch-to-block classifiers |
| Deep CNN | FingerNet | Tang et al. (2017) | Official released segmentation head, zero-shot |
| Random decision forest | Random forest | Sankaran et al. (2017) | Shared block features and validation grid |
| Weighted extreme learning machine | Weighted ELM | Ezeobiejesi & Bhanu (2016) | W1 class weighting |

Architecture and deviation details are in [METHODS.md](docs/METHODS.md).

## Datasets

The benchmark uses NIST Special Database 302, available from the
[official NIST SD302 page](https://www.nist.gov/itl/iad/btg/nist-special-database-302).
No fingerprint image, annotation, derived mask, per-image metric, or trained
weight is distributed here.

- **SD302-Latent:** latent images plus examiner ROI and usable-ridge annotations.
- **SD302-Exemplar:** live-scan exemplars evaluated with expert-minutiae hulls,
  explicitly treated as a lower-bound region rather than full pixel GT.
- Manually annotated live-scan and palm endpoints remain conditional on human GT.

See [DATASETS.md](docs/DATASETS.md) for acquisition and reconstruction steps.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/make_synthetic_demo.py --out examples/synthetic --subjects 4 --per-subject 1 --seed 7
python scripts/build_synthetic_site_demo.py --methods rf --steps 1 --threads 1
python scripts/smoke_synthetic.py --work-dir .smoke
python site/build.py
python scripts/check_site_numbers.py
python scripts/scan_release.py --phase working --allow-limited-synthetic
```

The smoke path is synthetic-only and is designed to finish in under five
minutes on a CPU. The same flow is available as the output-free
[`examples/synthetic/quickstart.ipynb`](examples/synthetic/quickstart.ipynb)
notebook. Its displayed metrics are pipeline checks on generated patterns, not
paper results. The core-only command genuinely fits and runs the random-forest
implementation and marks every unrequested method as `not_built`. To rebuild
all locally executable website outputs, install `requirements-deep.txt` and
`requirements-pretrained.txt`, then run:

```bash
CUDA_VISIBLE_DEVICES='' KERAS_BACKEND=torch \
  python scripts/build_synthetic_site_demo.py --methods all --steps 30 --threads 2
```

This slower command uses tiny synthetic-only CPU training budgets. It is an
executable-code demonstration, not benchmark evidence. FingerNet and
ImageNet-initialized SegFormer remain explicitly unavailable because their
pinned external snapshots are not distributed in this repository. Once an
authorised local SD302 copy is available, each paper
table has one entry point:

```bash
bash scripts/reproduce_table_A.sh --plan "$BENCH_ROOT/plans/A.json" --output "$BENCH_ROOT/out"
bash scripts/reproduce_table_A_prime.sh --plan "$BENCH_ROOT/plans/A_prime.json" --output "$BENCH_ROOT/out"
bash scripts/reproduce_table_B1.sh --plan "$BENCH_ROOT/plans/B1.json" --output "$BENCH_ROOT/out"
bash scripts/reproduce_table_D.sh --plan "$BENCH_ROOT/plans/D.json" --output "$BENCH_ROOT/out"
```

The shell entry points validate private manifests, artifact hashes, subject
splits and frozen selection receipts, invoke the real inference/evaluation
programs, and reject incomplete prediction coverage. They never download
biometric data or select a result on test. See the complete private-plan schema
and dry-run workflow in [REPRODUCTION.md](docs/REPRODUCTION.md).

Final aggregate outputs enter the public tree through one fail-closed command.
Run its dry mode first; it validates the hash-bound aggregate-only bundle and
rebuilds the results documentation and website in isolation:

```bash
python scripts/update_from_aggregate.py \
  --bundle /path/to/public-aggregate-bundle \
  --dry-run \
  --report /path/to/dry-run-report.json
python scripts/check_release_requirements.py
```

The accepted schema and apply command are documented in
[`results/README.md`](results/README.md).

The final publication bridge consumes the canonical Stage-5 package directly,
with its receipt digest supplied independently. Run a dry conversion first;
the final metadata flag refuses publication unless the same receipt also binds
the chosen public configs and anonymous subject split:

```bash
python scripts/import_stage5_public.py \
  --package "$STAGE5_PACKAGE" \
  --expected-receipt-sha256 "$STAGE5_RECEIPT_SHA256" \
  --require-release-metadata --dry-run \
  --report /path/outside/the/repository/stage5-import-dry-run.json
```

After controlled apply, use the working scan while the generated files are
deliberately uncommitted. Commit those exact files, then run the final clean-tree
scan with its report outside the repository. See
[`PUBLICATION.md`](docs/PUBLICATION.md).

The receipt-authenticated private review interface is documented separately in
[`PRIVATE_DEMO.md`](docs/PRIVATE_DEMO.md) and
[`PRIVATE_DEMO_RUNBOOK.md`](docs/PRIVATE_DEMO_RUNBOOK.md). It is not part of
the public website and accepts only digest-pinned local artifacts.

Validation-only handoff, full seed-zero launch, and one-time seed delivery are
described in
[`AUTONOMOUS_VALIDATION_HANDOFF.md`](docs/AUTONOMOUS_VALIDATION_HANDOFF.md),
[`AUTONOMOUS_SEED0_HANDOFF.md`](docs/AUTONOMOUS_SEED0_HANDOFF.md), and
[`STAGE3_INCREMENTAL_RUNBOOK.md`](docs/STAGE3_INCREMENTAL_RUNBOOK.md). The
ground-truth-blind B1 delivery path is documented in
[`B1_CROSSDOMAIN_RUNBOOK.md`](docs/B1_CROSSDOMAIN_RUNBOOK.md).

## Protocol summary

- Frozen 136/24/40-subject train/validation/test assignment.
- Train-only normalisation; validation-only learning-rate, epoch, threshold,
  and post-processing selection.
- Three seeds for trainable methods; released methods retain their declared
  training regime.
- Full-image, 500-ppi pixel evaluation through one implementation.
- Subject-cluster bootstrap confidence intervals and paired Wilcoxon tests with
  Holm correction.
- Every rerun and protocol amendment is retained in an immutable audit trail.

## Project links

- [Project website](https://udiation.github.io/latent-fingerprint-segmentation-benchmark/)
- [Methodology](docs/METHODOLOGY.md)
- [Protocol](docs/PROTOCOL.md)
- [Results](docs/RESULTS.md)
- Paper: pending

## Citation

```bibtex
@software{narayan2026latentseg,
  author  = {Narayan, Udit and {TODO: co-authors}},
  title   = {Latent Fingerprint Segmentation Benchmark},
  year    = {2026},
  version = {v1.0-benchmark},
  url     = {https://github.com/Udiation/latent-fingerprint-segmentation-benchmark}
}
```

## Licence, acknowledgements, and contact

The benchmark software is MIT licensed. Dataset use remains governed by the
terms supplied with NIST SD302. We acknowledge NIST SD302, Cappelli's pyfing,
the FingerNet authors, and the Hugging Face SegFormer implementation.

Contact: open a GitHub issue for code and reproducibility questions. Requests
for weights must be reviewed separately and remain subject to dataset terms.
