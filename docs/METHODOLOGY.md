# Methodology

> **Release status:** protocol-complete; v1.0 is a provisional aggregate
> snapshot. Completed rows contain authenticated test aggregates, while every
> unfinished row is explicitly marked pending. No pending value is inferred.

## Study design

We compare the fingerprint-foreground segmentation entries listed in Fig. 2 of
Dhaneshwar and Wadhera (2026) under one subject-disjoint protocol. The unit of
splitting is the NIST anonymous subject rather than an image or impression.
The fixed assignment contains 136 training subjects, 24 validation subjects,
and 40 test subjects. Every image, annotation and capture associated with one
subject inherits the same assignment. Exact-content hashes are checked for
cross-split duplicates before training.

## Ground truth and endpoints

Table A uses the examiner ROI polygon from EFS field 9.300 as the latent-print
foreground. Table A-prime evaluates the same frozen predictions against the
usable-ridge area derived from field 9.308. Of 10,092 annotated impressions,
10,024 have complete quality maps; 68 are excluded from A-prime only. Two
NONPRINT annotations are excluded from test scoring. Low ROI/quality agreement
is retained rather than filtered.

For SD302 exemplars, a full foreground outline is unavailable. Expert minutiae
from field 9.331 are converted from physical units using the recorded image
resolution and rasterised as a convex hull. The hull is a lower bound: pixels
outside it can still be valid fingerprint. We therefore report minutiae
coverage, hull recall, predicted-area ratio and outside-hull fraction, never
Dice, Jaccard or precision against the hull.

## Preprocessing and training

Images are converted to grayscale, reduced to 500 ppi with area interpolation,
and normalised to [0,1]. Masks use nearest-neighbour interpolation. Pixel models
use 512-pixel random crops with one shared augmentation distribution. Patch
models predict centred 16-pixel blocks from their declared context window and
are upsampled to full-resolution pixel masks before scoring.

Every trainable deep family receives the same two-rate validation sweep,
followed by fresh full runs for three seeds. Full training uses BCE plus Dice,
AdamW, cosine decay and validation-Dice early stopping. Thresholds are selected
on validation probability maps from the frozen grid 0.1 through 0.7 in steps of
0.1. Zero-shot rows use the released inference procedure unless explicitly
stated otherwise. Classical models select their grid entry by validation
balanced block accuracy.

## Evaluation and statistics

All binary predictions are scored by the same evaluator at 500 ppi. Primary
values are image-macro means; three-seed rows report the mean and seed standard
deviation. Timing includes pre- and post-processing at batch size one. Dice
confidence intervals use 1,000 subject-cluster bootstrap resamples. Pairwise
Dice comparisons use the Wilcoxon signed-rank test with Holm correction.

Final test scoring occurs only after the configuration, checkpoint and
validation-selected threshold are frozen. Missing predictions are counted and
scored as all-background. All-background and all-foreground rows are descriptive
references and are excluded from method ranking.

## Declared deviations

- SegFormer-B2 is labelled as a substitute for the reviewed OCT-specific
  transformer, which is not a directly applicable two-dimensional fingerprint
  segmentation architecture.
- AlexNet uses a fixed average-pooling operation for its fixed-size input when
  the operation is numerically equivalent to adaptive pooling; the equivalence
  is unit tested.
- SVM uses a Nyström approximation to the RBF kernel for tractable block-scale
  training.
- The classical methods are faithful reimplementations rather than executions
  of unavailable original source code.
- Released SUFS and FingerNet models are zero-shot and were trained externally,
  not on the benchmark subjects.

Metric values and chosen hyperparameters enter this repository only through an
authenticated aggregate release bundle. The v1.1 refresh will replace pending
cells after the remaining frozen run and locked scoring transaction complete.
