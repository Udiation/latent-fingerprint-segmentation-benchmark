# SD302-Exemplar cross-domain delivery runbook

This runbook publishes one already-selected Dataset-1 model seed on Table B1.
It does not train on SD302g and does not use its minutiae hull during inference.
The output metrics are MC, HR, PAR and OHF only. Dice, Jaccard, MIoU, MDR and
FDR are invalid for this lower-bound endpoint.

## Preconditions

- The Dataset-1 final seed has a clean run card, immutable `best.pt`, and a
  full validation-only selector on thresholds 0.1 through 0.7.
- `build_sd302g_hulls.py` and the independent coordinate check produced a
  `REFERENCE_VERIFIED` manifest bound to the authoritative 136/24/40 subject
  split.
- `prepare_sd302g_inference_manifest.py` produced a complete manifest with an
  empty `mask_path` and a source SHA-256 for every image.
- The inference code is a clean pinned commit. RA-RUnet alignment is not
  covered here; it remains pending while the required unlabelled SD302b target
  is unavailable.

## 1. Prepare without opening the held-out endpoint

Copy `configs/b1_crossdomain.template.json` to restricted storage and replace
every path and the expected commit. Then run:

```bash
python segbench/scripts/prepare_b1_crossdomain.py --spec /private/b1/unet_seed0/spec.json
```

The preparer authenticates the training run, selector, split, coordinate
verification, GT-blind manifest and implementation bytes. It emits an
immutable authorization and a plan containing exact commands. It does not
decode an exemplar, open a hull/minutiae file, launch a GPU process or compute
a result.

## 2. Execute the inference command once

Submit `plan.json`'s `inference.argv` through the guarded project queue. The
common runner reads only the exemplar image path, PPI and source digest. The
Dataset-1 selector remains byte-identical; the cross-domain authorization
permits its threshold to be reused without pretending it was selected on the
target manifest.

The complete card must retain:

- one binary 0/255 mask at 500 ppi per test exemplar;
- one pre-threshold float16 probability map per exemplar;
- per-image timing including decode, resampling and post-processing after the
  declared warm-up; and
- clean process-start provenance plus all selector/checkpoint/source hashes.

## 3. Arm and score exactly once

After inference completes, run `plan.json`'s `score_once.argv`:

```bash
python segbench/scripts/finalize_b1_crossdomain.py \
  --authorization /private/b1/unet_seed0/authorization.json
```

The finalizer authenticates every retained artifact before creating the
exclusive score-once journal. Only after that journal exists does it invoke
`evaluate_hull.py`, the sole process that opens hull/minutiae files. It requires
Hull-as-prediction MC = 100% and HR = 100%, zero missing predictions, and
overall plus group/device/capture aggregates. A journal without a completed
bundle blocks automatic retry; remediation must use a newly documented,
versioned destination.

The resulting bundle uses the same adoption-receipt contract as the existing
zero-shot B1 rows, so `finalize_sd302g_b1.py` can merge it without regenerating
or modifying their predictions.
