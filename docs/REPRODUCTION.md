# One-command table reproduction

The four `scripts/reproduce_table_*.sh` entry points run real mask inference
and the locked evaluators. They do not download licensed data and they do not
select a learning rate, checkpoint, or threshold. Those frozen inputs are
declared in a private JSON plan whose schema is
`segbench.reproduction-plan.v1`.

## Safety model

- Plans and outputs must remain outside the public repository.
- Artifact-backed runners require a caller-supplied SHA-256. A mismatch stops
  before inference.
- Test inference for neural models requires the immutable validation threshold
  receipt and the protocol amendment already enforced by the model runner.
- Every prediction must exist for every selected test row before evaluation.
  This prevents the evaluator's documented missing-as-background fallback from
  hiding an incomplete reproduction run.
- All commands are assembled from a fixed runner allowlist and invoked without
  a shell. A plan cannot inject an arbitrary command.
- Table A and A-prime are fixed at 500 ppi and exclude `NONPRINT`. A-prime
  requires the successful, hash-bound Table A receipt, reuses its exact
  prediction templates, and changes only the GT column.
- B1 uses only `evaluate_hull.py`; Dice, Jaccard and FDR are never computed on
  the exemplar lower-bound hull.
- Table D requires each entry to declare whether its target is a pixel mask or
  a lower-bound hull.

Dry-run validates paths, hashes, split syntax and scientific dispatch, then
prints the exact commands without creating outputs:

```bash
bash scripts/reproduce_table_A.sh \
  --plan "$BENCH_ROOT/plans/table_A.json" \
  --output "$BENCH_ROOT/reproduced" \
  --dry-run
```

Remove `--dry-run` to execute. A resumable receipt is written as
`reproduce_<table>.json` under the private output root. Evaluator-native files
are under `results/<table>/<run_key>/`; masks, probability maps and run cards
remain under `runs/<run_key>/`.

The same paths can be supplied through a private configuration file:

```text
REPRO_PLAN=plans/table_A.json
REPRO_OUTPUT=outputs/reproduced
```

```bash
bash scripts/reproduce_table_A.sh --config "$BENCH_ROOT/table_A.env"
```

Alternatively, set `SEGBENCH_REPRO_PLAN` and `SEGBENCH_REPRO_OUTPUT`.
CLI arguments take precedence. Plan-relative paths are resolved relative to
the plan file; config and environment paths are resolved by the caller.

## Plan shape

This minimal plan evaluates already-generated masks. Values shown here are
paths relative to the private plan, not files shipped by this repository.

```json
{
  "schema": "segbench.reproduction-plan.v1",
  "table": "A",
  "source_manifest": "manifests/latent.csv",
  "splits": "splits/subjects.json",
  "subset": "finger_latent",
  "split": "test",
  "target_ppi": 500,
  "entries": [
    {
      "run_key": "method_seed0",
      "method": "method",
      "display_name": "Method",
      "runner": {
        "kind": "existing_predictions",
        "pred_template": "predictions/method_seed0/{image_id}.png",
        "pred_ppi": 500
      },
      "metadata": {
        "seed": 0,
        "device": "GPU",
        "run_id": "method_seed0"
      }
    }
  ]
}
```

No threshold appears in this example because the release must not invent a
final value. When final evidence exists, `metadata.threshold` records the
already-frozen validation-selected value; it does not control inference.

A Table A-prime plan additionally declares `table_a_receipt` and
`table_a_receipt_sha256`. Each A-prime `run_key`, method, prediction template
and prediction PPI must match that successful Table A receipt exactly. Its
runner kind is therefore always `existing_predictions`.

### Frozen neural checkpoint

Use `deep_checkpoint` for U-Net, UNet++, DRUnet, RUnet, patch networks and the
SegFormer substitute. The configuration, checkpoint, immutable selector and
amendment are mandatory. The checkpoint digest is supplied by the operator.

```json
{
  "kind": "deep_checkpoint",
  "config": "configs/frozen.yaml",
  "checkpoint": "checkpoints/frozen.pt",
  "checkpoint_sha256": "<64 lowercase hexadecimal characters>",
  "threshold_receipt": "receipts/threshold.json",
  "threshold_amendment": "receipts/amendment.json",
  "device": "cpu",
  "seed": 0
}
```

Optional keys are `cache_dir`, `cache_compatibility_receipt`,
`cross_domain_authorization`, and `memory_fraction`. Cross-domain test
inference remains subject to the authorization checks in `infer_masks.py`.

Fine-tuned SUFS uses `sufs_checkpoint` with the same checkpoint, digest,
configuration, selector and amendment fields. Its own runner performs the
released-weight identity checks.

### Other audited runners

| `runner.kind` | Required plan fields | Repository program |
|---|---|---|
| `classical_model` | `model`, `model_sha256`, `training_card`, `method`, `display_name`, `seed` | `predict_classical.py` |
| `pyfing` | `method`, `display_name`; optional frozen `parameters_json` plus `parameters_sha256` | `run_pyfing.py generate` |
| `fingernet` | `config`, `source_repo`, `weights`, `weights_sha256`, `device` | `run_fingernet.py` |
| `existing_predictions` | `pred_template`, explicit `pred_ppi` | evaluator only |
| `reference` | `reference` equal to `all-background` or `all-foreground` | common mask evaluator |

The FingerNet runner independently verifies the pinned source revision and
released weights. The classical runner independently binds its fitted model to
the training card. GMFS-tuned plans point `parameters_json` to the frozen
validation selection; the dispatcher never tunes on test data.
Every receipt records the SHA-256 of the source manifest, evaluation manifest,
subject split and plan in addition to artifact-specific hashes recorded by the
underlying inference run cards.

## Per-table entry points

```bash
bash scripts/reproduce_table_A.sh --plan "$BENCH_ROOT/plans/A.json" --output "$BENCH_ROOT/out"
bash scripts/reproduce_table_A_prime.sh --plan "$BENCH_ROOT/plans/A_prime.json" --output "$BENCH_ROOT/out"
bash scripts/reproduce_table_B1.sh --plan "$BENCH_ROOT/plans/B1.json" --output "$BENCH_ROOT/out"
bash scripts/reproduce_table_D.sh --plan "$BENCH_ROOT/plans/D.json" --output "$BENCH_ROOT/out"
```

For B1, `source_manifest` is the image manifest used for inference and
`evaluation_manifest` is the verified hull-builder manifest with its adjacent
provenance receipt. The evaluator automatically emits the all-background,
all-foreground and hull-as-prediction references for each method invocation.

For D, declare `evaluation.kind` as `mask` or `hull`. Mask entries must also
declare `gt_column`; hull entries must use a verified hull manifest. An entry
may override `source_manifest`, `evaluation_manifest`, `splits` and `subset`,
allowing both cross-domain directions without changing code.

## Synthetic verification

`tests/test_reproduce_entrypoints.py` exercises all four shell entry points in
dry-run mode, verifies checkpoint hash binding, and runs a complete synthetic
Table A transaction through `evaluate_masks.py`. It uses no biometric data:

```bash
CUDA_VISIBLE_DEVICES='' pytest -q tests/test_reproduce_entrypoints.py
```
