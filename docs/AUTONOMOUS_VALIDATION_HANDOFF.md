# Autonomous validation-only handoff runbook

This controller fills unused registered slots during the authenticated
AlexNet/SegFormer mixed-recovery queue, then completes the frozen validation
probability replays. It never opens the test split and never launches the
selected full seed-0 queue it prepares.

## Preparation

Preparation is results-blind and launches nothing:

```bash
python scripts/autonomous_validation_handoff.py prepare \
  --spec /absolute/path/handoff_spec.yaml \
  --code-root /absolute/path/clean/controller/worktree \
  --out /absolute/path/handoff_plan.json \
  --proof-out /absolute/path/handoff_preparation.json
```

The specification must bind the transition watcher by PID, process start
ticks, and argv SHA-256, plus every permitted post-`exec` mixed-controller argv.
Preparation accepts only that watcher or one of those exact post-`exec`
commands. It also binds the immutable initial validation replay plan and queue.
The specification supplies an explicit `runtime_policy`; no device index,
memory reservation, CPU limit, or disk guard is embedded in public code. The
policy contains `allowed_gpu_indices`, `max_jobs_per_gpu`, `min_free_mib`,
`foreign_process_mib`, `memory_fraction`, `min_disk_free_fraction`,
`max_project_ram_gib`, `max_cpu_cores`, `min_available_ram_gib`,
`seed0_host_ram_gib`, and `seed0_cpu_workers`.

## Launch after review

Use the plan digest printed in the preparation receipt:

```bash
nice -n 10 python scripts/autonomous_validation_handoff.py run \
  --plan /absolute/path/handoff_plan.json \
  --plan-sha256 PLAN_SHA256
```

Run this in its own detached `tmux` session and write stdout/stderr to the
restricted log directory. Before every launch decision, the guarded replay
queue authenticates the source controller and workers and counts them toward
the GPU-slot, CPU, RAM, memory-fraction, and disk guards. It cannot signal the
source or write its ledger. If the source queue has already completed, the
controller accepts its expected exit only after authenticating the full source
ledger and every declared completion artifact.

The controller performs these append-only actions:

1. run the immutable first validation replay queue;
2. finalize learning-rate choices from validation completion metadata;
3. replay eligible historical full checkpoints on validation;
4. wait for both sweep run cards for DRUnet, RUnet, AlexNet, and SegFormer;
5. compile and run a separate second validation replay queue;
6. finalize those validation-only choices; and
7. prepare, but do not launch, a full seed-0 queue. Any historical full run
   made ineligible by the new LR choice is replaced only in the versioned
   `full_v3_expanded_grid` namespace; SUFS receives a trainer-compatible
   validation selection adapter rather than overwriting its legacy receipt.

Any dependency, identity, split, process, source-state, resource, disk, or
completion mismatch fails closed. Test inference and scoring remain a separate
sealed transaction.
