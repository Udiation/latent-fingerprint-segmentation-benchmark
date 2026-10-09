# Autonomous validation-only handoff runbook

This controller waits for the authenticated AlexNet/SegFormer mixed-recovery
queue to complete, then runs the frozen validation-probability replays. It
never opens the test split and never launches the selected full seed-0 queue it
prepares.

## Preparation

Preparation is results-blind and launches nothing:

```bash
python segbench/scripts/autonomous_validation_handoff.py prepare \
  --spec /absolute/path/handoff_spec.yaml \
  --code-root /absolute/path/clean/controller/worktree \
  --out /absolute/path/handoff_plan.json \
  --proof-out /absolute/path/handoff_preparation.json
```

The specification must bind the transition watcher by PID, process start
ticks, and argv SHA-256, plus every permitted post-`exec` mixed-controller argv.
Preparation accepts only that watcher or one of those exact post-`exec`
commands. It also binds the immutable initial validation replay plan and queue.
The restricted specification must provide, without defaults, the complete
guarded-launch runtime policy and the selected full-job RAM/worker reservation.
The full-job reservation includes the absolute path and SHA-256 of its private
measurement receipt; preparation and both downstream handoffs authenticate
that evidence without copying private measurements into committed code. Every
late replay candidate likewise declares its RAM and worker reservation. A
serial replay with no DataLoader or process-pool workers declares zero workers;
full training declares the configured DataLoader-worker count.
Preparation rejects a policy whose selected full-job reservations cannot fill
all declared device slots within the aggregate CPU and RAM caps.

## Launch after review

Use the plan digest printed in the preparation receipt:

```bash
nice -n 10 python segbench/scripts/autonomous_validation_handoff.py run \
  --plan /absolute/path/handoff_plan.json \
  --plan-sha256 PLAN_SHA256
```

Run this in its own detached `tmux` session and write stdout/stderr to the
restricted log directory. The guarded replay queue launches nothing while a
source worker is running. It accepts the source controller's expected exit
only after all source jobs are complete and it has authenticated the full
source ledger and every declared completion artifact.

This source-completion barrier is intentional. The recovery controller was not
launched with reciprocal ownership of replay workers, so concurrent replay
could make a valid replay process appear foreign to the recovery queue and
trigger cooperative source checkpointing. The rejected coexistence plan never
launched a replay worker. Once the barrier passes, the guarded replay queue
applies the ordinary process-slot, CPU, RAM, memory-fraction, and disk guards.
It cannot signal the source or write its ledger.

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
