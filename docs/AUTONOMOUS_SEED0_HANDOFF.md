# Autonomous full seed-0 handoff

`scripts/autonomous_seed0_handoff.py` closes the intentional gap between the
validation-only handoff and full seed-0 training. It can be started while the
validation handoff is still running. It launches nothing until the upstream
controller publishes its immutable completion receipt.

## Trust chain

The controller fails closed unless all of these identities agree:

1. the exact upstream handoff plan and its no-launch preparation proof;
2. the clean Git commit, tree and file hashes recorded in that plan;
3. the upstream completion receipt and its bound handoff-plan SHA-256;
4. the selected full seed-0 queue and the queue SHA-256 in the completion;
5. the adjacent `selected_full_seed0_preparation.json`, including both
   validation-selection summary hashes, the queue hash, the job list and each
   fresh/reuse decision;
6. every generated configuration or SUFS selection-adapter hash;
7. every guarded job identity and its declared completion artifact; and
8. this controller's own clean commit, tree and launcher file hashes.

The first accepted set is written once to `launch_authorization.json`. The
entire chain is replayed immediately before the guarded subprocess starts and
again after it returns.

## Safety boundaries

- Only `train_seg.py`, `train_patch.py` and `train_sufs.py` are accepted.
- `test`, `heldout`, `--test*` and `--heldout*` tokens are rejected.
- The controller does not open manifests, split files, images, masks,
  probabilities, checkpoints or result payloads. It reads control metadata
  and hashes only.
- Each new output must be exactly
  `runs_root/full_v3_expanded_grid/<method>/seed0`; legacy full-run paths are
  rejected.
- A pre-existing non-empty versioned output without the controller ledger is
  rejected. A normal controller restart uses the guarded ledger and
  `--resume auto`.
- Training runs only through `run_queue_guarded.py`, with the registered GPU
  allow-list, concurrency, process-memory fraction, CPU/RAM limits, foreign
  process guard and disk floor. The outer controller must run at nice level 10
  or lower priority so every child inherits it.
- This controller never performs inference, scoring or test selection.

## Prepare (no launch)

Write a JSON specification containing only control-plane paths and digests:

```json
{
  "schema_version": 1,
  "kind": "autonomous_seed0_handoff_spec",
  "source_handoff_plan": "/absolute/path/to/upstream/plan.json",
  "source_handoff_plan_sha256": "<sha256>",
  "source_handoff_preparation_proof": "/absolute/path/to/upstream/preparation.json",
  "source_handoff_preparation_proof_sha256": "<sha256>",
  "controller_root": "/absolute/private-results/seed0_handoff",
  "source_wait_seconds": 172800,
  "poll_seconds": 10,
  "runtime_policy": {
    "allowed_gpus": ["<private integer indices>"],
    "max_jobs_per_gpu": "<private integer>",
    "memory_fraction": "<private fraction>",
    "max_cpu_cores": "<private integer>",
    "max_project_ram_gib": "<private GiB>",
    "min_disk_free_fraction": "<private fraction>",
    "min_free_mib": "<private MiB>",
    "foreign_process_mib": "<private MiB>",
    "min_available_ram_gib": "<private GiB>",
    "host_ram_gib_per_job": "<measured reservation GiB>",
    "cpu_workers_per_job": "<configured loader workers>",
    "reservation_evidence": {
      "path": "/absolute/private/path/to/measurement-receipt",
      "sha256": "<receipt sha256>"
    }
  }
}
```

Then, from one clean immutable checkout:

```bash
python segbench/scripts/autonomous_seed0_handoff.py prepare \
  --spec /absolute/path/to/spec.json \
  --code-root /absolute/path/to/clean/repository \
  --out /absolute/private-results/seed0_handoff/plan.json \
  --proof-out /absolute/private-results/seed0_handoff/preparation.json
```

Preparation succeeds before the upstream `completion.json` exists and writes
`READY_NOT_LAUNCHED`; it never invokes a subprocess.

## Future launch

Use the literal plan digest emitted by preparation. Run the controller in its
own detached terminal session so it can poll safely across disconnects:

```bash
nice -n 10 python segbench/scripts/autonomous_seed0_handoff.py run \
  --plan /absolute/private-results/seed0_handoff/plan.json \
  --plan-sha256 <literal-plan-sha256>
```

The controller blocks until the exact upstream completion appears. It then
executes only the guarded policy recorded in the restricted specification.
The policy has no code defaults: it must explicitly provide the device
allow-list, concurrency, per-process memory fraction, CPU/RAM budgets, GPU
free-memory and foreign-process guards, per-job reservations and disk floor.
Its private measurement receipt is authenticated by path and SHA-256, and the
same identity must already be bound by the upstream selected-full policy.
Preparation rejects a policy whose per-job RAM or worker reservations cannot
fill every declared slot within the aggregate caps.

On success, `completion.json` binds the authorization, guarded ledger and all
completion artifact hashes. A non-zero guarded exit, an identity mismatch, a
held-out token or an unledgered output leaves no controller completion receipt.
