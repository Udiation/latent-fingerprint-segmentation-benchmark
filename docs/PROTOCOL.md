# Frozen evaluation protocol

This is the public, infrastructure-free form of the preregistered protocol.
The private execution log is authoritative for process-level provenance; only
shareable amendments are copied here.

## Invariants

1. Subjects, not images, are split into 136 train, 24 validation and 40 test.
2. Training and validation may be repeated under the declared equal tuning
   budget. Test data never influences a choice.
3. Images are evaluated at 500 ppi, with area interpolation for images and
   nearest-neighbour interpolation for masks.
4. Full-image pixel metrics are produced by one evaluator revision. Patch block
   maps are upsampled before this evaluator sees them.
5. Deep models share learning-rate candidates `{3e-4, 1e-3}` and threshold
   candidates `{0.1, 0.2, ..., 0.7}`. Each threshold is chosen on validation
   probability maps.
6. Classical grids are selected by validation balanced block accuracy. GMFS's
   separate tuned row is selected by validation Dice from its declared grid.
7. Three seeds are required for trainable final rows. A released or deterministic
   algorithmic row is reported once without invented seed variation.
8. Each final model has one locked test-scoring transaction. A bug-forced rerun
   must retain the earlier artifact and disclose the reason.

## Common pixel metrics

For fingerprint foreground as the positive class:

- `Accuracy = (TP + TN) / (TP + FP + FN + TN)`
- `Error = 1 - Accuracy`
- `Dice = 2TP / (2TP + FP + FN)`
- `Jaccard = TP / (TP + FP + FN)`
- `MIoU = (Jaccard + TN/(TN+FP+FN))/2`
- `MDR = FN / (TP + FN)`
- `FDR = FP / (FP + TN)`

FDR is the background false-detection rate, not statistical false discovery.

## Exemplar lower-bound metrics

Let `M` be expert minutiae, `H` their convex-hull pixels and `P` predicted
foreground:

- `MC = minutiae inside P / all minutiae`
- `HR = |H intersect P| / |H|`
- `PAR = |P| / |H|`
- `OHF = |P outside H| / |P|`

MC and HR must be interpreted together with PAR. The hull is not full GT, so no
pixel-overlap metric is reported against it.

## Shareable amendment log

| Date | Amendment | Prospective justification |
|---|---|---|
| 2026-10-08 | Extended the neural validation threshold grid from 0.3–0.7 to 0.1–0.7 | A validation-only sweep selected the old lower boundary; no affected test result had been opened for Stage 3 selection |
| 2026-10-08 | Required retained validation and test probability maps for neural rows | Makes validation re-thresholding auditable without changing frozen weights |
| 2026-10-09 | Permitted fixed average pooling for AlexNet's fixed-size input after numerical equivalence testing | Avoids a nondeterministic backward implementation without changing the mathematical mapping |

## Leakage assertions

- subject sets are pairwise disjoint;
- patch training samples originate only from train subjects;
- validation Dice is computed on complete validation images;
- normalisation statistics are fit on train only;
- cross-domain alignment may use unlabelled train-subject target images but no
  validation or test target image;
- all test masks remain sealed until the validation-selected model is frozen.

## Validation handoff and one-time scoring

Validation probability replay is controlled by an identity-bound handoff that
accepts only a preregistered source process, immutable queue bytes, and a clean
code revision. Its device and resource policy is supplied at deployment time;
the public implementation contains no machine topology. The controller can
co-schedule validation-only work but cannot signal the source controller,
modify its ledger, open the test split, or launch the selected full-training
queue. An ineligible historical full run is replaced in a new versioned
namespace rather than overwritten.

After validation selection is frozen, each final seed may receive its one
held-out score through an append-only journal. Table A and Table A-prime reuse
the same authenticated prediction bytes, and the common evaluator is invoked
exactly once per endpoint. This operational G3 delivery does not satisfy the
multi-seed G4 analysis gate: final statistical analysis still requires at
least two successful seeds for every trainable native family. Import into the
combined delivery verifies hashes and copies evidence byte-for-byte without
rescoring.

## Public aggregate publication boundary

The public repository never imports evaluator-level or per-image artifacts.
Stage 4 must first emit a separate public aggregate bundle containing only the
three exact CSV schemas documented in `results/README.md`, bound by a SHA-256
manifest. `scripts/update_from_aggregate.py` rejects any other file or column,
screens every aggregate cell for private identifiers and infrastructure/data
references, and rebuilds `docs/RESULTS.md` and the website from those same CSV
bytes. The README headline table is replaced only inside explicit generated
markers from the same rows. A dry run stages and number-checks all outputs
without replacing tracked files; apply mode installs only the already-validated
staged output set.
