# Incremental G3 held-out delivery

This runbook implements the protocol's split between G3 and G4. It lets each
final seed receive its single held-out score as soon as its validation-only
selection evidence is frozen. It does not relax the original G4 requirement of
at least two successful seeds for every native model.

## 1. Freeze the seed-0 plan

Create a restricted JSON specification with kind
`stage3_incremental_seed0_plan_spec_v1`. Bind the manifest, subject split,
threshold amendment, preprocessing-cache receipt, and the amended Stage-2
U-Net seed-0 delivery by absolute path and SHA-256. Supply exactly one seed-0
record for each native method except U-Net. Every record binds:

- configuration, run card, checkpoint, expanded-grid selector receipt, and raw
  plus selected validation probability cards;
- either legacy-completion evidence or the amended per-epoch profile;
- mutually disjoint mask, probability, inference-card, work, and per-seed
  delivery output directories; and
- the fixed training-lineage label.

The specification also binds the Python executable, code root, inference and
scoring commits, runtime-authorized device indices, per-job RAM reservations,
and expected prediction count. Hash the completed specification, then run:

```bash
python scripts/prepare_stage3_seed0_incremental.py \
  --spec <seed0-spec.json> --expected-spec-sha256 <sha256> \
  --restricted-root <restricted-root> \
  --gate-out <seed0-gate.json> --plan-out <seed0-plan.json> \
  --queue-out <seed0-inference-queue.yaml>
```

The emitted gate must say `satisfies_stage4_multi_seed_gate: false`. Run the
queue with the guarded scheduler and the queue's exact registered device list,
`--max-per-gpu 1`, and memory fraction `0.3`. The one-job timing isolation is
required for comparable `ms/img`; it is independent of training concurrency.

## 2. Publish each seed exactly once

After a planned inference card is complete, run the finalizer from a clean
checkout at the scoring commit:

```bash
python scripts/finalize_stage3_incremental_seed.py \
  --plan <seed0-plan.json> --expected-plan-sha256 <sha256> \
  --restricted-root <restricted-root> --method <method> --seed 0 \
  --bundle-out <the-records-frozen-seed-delivery-out> \
  --repo-root <git-top-level> --expected-git-commit <scoring-commit>
```

The finalizer requires the bundle path frozen in the plan, creates a durable
score-once journal before evaluation, and invokes `score_threshold_run.py`
once. That shared scorer authenticates one retained probability/mask set and
passes the same frozen mask directory to the common evaluator for Table A and
Table A-prime. A completed delivery is verification-only on subsequent calls.
Any partial bundle or existing journal without a complete delivery requires a
manual audit; automatic rescoring is forbidden.

## 3. Add seeds 1 and 2

For each newly completed seed, create a restricted specification with kind
`stage3_incremental_additional_plan_spec_v1`. It uses the same frozen data,
threshold, cache, code, device, and resource fields, binds all prior per-seed
deliveries plus the Stage-2 U-Net seed-0 delivery, and contains only
not-yet-delivered seed-1 or seed-2 records. All seven non-U-Net seed-0
deliveries must already be present. Run:

```bash
python scripts/prepare_stage3_additional_incremental.py \
  --spec <additional-spec.json> --expected-spec-sha256 <sha256> \
  --restricted-root <restricted-root> \
  --gate-out <additional-gate.json> --plan-out <additional-plan.json> \
  --queue-out <additional-inference-queue.yaml>
```

Run its guarded inference queue and finalize each planned seed with the command
in Section 2, substituting the additional plan and its frozen delivery path.
The planner rejects any method/seed already present in a prior delivery.

## 4. Import into the G4 bundle

Do not import until the original `stage3_native_training_readiness_gate` proves
at least two valid seeds, including seed 0, for every native method. Create a
restricted import specification of kind `stage3_incremental_import_spec_v1`
that binds the manifest, split, readiness gate, Stage-2 U-Net seed-0 delivery,
and exactly every successful non-U-Net-0 per-seed delivery named by the gate.
Then run:

```bash
python scripts/import_stage3_incremental.py \
  --spec <import-spec.json> --expected-spec-sha256 <sha256> \
  --restricted-root <restricted-root> --bundle-out <base-stage3-delivery>
```

The importer cannot run inference or evaluation. It authenticates training and
test-universe identities, copies per-image evidence byte-for-byte, and marks
every imported seed `imported_not_rescored`. Feed that base delivery and the
same per-run threshold records to `build_stage4_deliveries.py stage3`; the
existing Stage-4 finalizer then applies its unchanged multi-seed gate.

## 5. Required checks

- The seed-0 and additional gates are operational only, never G4 passes.
- Each per-seed receipt has exactly two invocations: `A` and `A_prime`.
- Both invocations derive from one frozen probability/mask set.
- A delivery's plan, journal, receipt, summary, and per-image CSV hashes still
  verify before it is admitted to a later plan or import.
- The combined delivery contains every and only the successful seeds named by
  the original readiness gate; U-Net seed 0 is reused from Stage 2.
- Importing an already complete bundle changes no bytes and invokes no scorer.
