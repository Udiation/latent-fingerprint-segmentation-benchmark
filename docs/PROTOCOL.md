# Segmentation benchmark execution protocol

This is the prespecified operational protocol for producing the segmentation
tables. It contains no benchmark results or biometric identifiers. The words
**must**, **must not**, and **stop** are normative.

## 1. Frozen study units

1. Create exactly one manifest row per unique source image with a stable image
   key, image path, ground-truth path, subject key, subset, and recorded native
   resolution. A source image must never be repeated merely because it has more
   than one annotation record.
2. For a latent source with multiple Type-9 annotations, union all associated
   examiner ROI masks. Union the associated usable-ridge masks only when every
   constituent annotation has a complete quality map; otherwise exclude that
   source from Table A-prime while retaining it in Table A. This aggregation is
   fixed before model training and uses no prediction.
3. Retain the constituent examiner assessments. Assign a source to VALUE,
   LIMITED, or NOVALUE only when all annotations agree; use an explicit MIXED
   group for conflicting assessments.
4. Exclude NONPRINT source images from final test scoring and report the two
   annotation-level exclusions in the restricted audit trail.
5. Never derive ground truth from GMFS, SUFS, a neural model, or a consensus of
   benchmark predictions.
6. Record input and manifest hashes before launching a learned method. Any data
   change creates a new benchmark version rather than silently updating a run.

## 2. Frozen partitions and leakage checks

The existing assignment is authoritative: 136 train subjects, 24 validation
subjects, and 40 test subjects, with recorded seed `20260908`. Conversion must:

- require exactly one split per subject;
- reject null, missing, or unknown assignments;
- assert the three expected subject counts; and
- verify that the subject sets are pairwise disjoint.

All samples from a subject must inherit its assignment. Before training, group
images by exact content hash and stop if a duplicate group crosses a split.
Finger/capture metadata are also audited for cross-split identities. FVC uses
set B for model development and set A once for final evaluation. The test split
must not influence crop policy, augmentation, architecture choices,
hyperparameters, thresholds, stopping, post-processing, or failure handling.

## 3. Evaluation tables

| Table | Model development | Final scoring |
|---|---|---|
| A: latent ROI | SD302e/SD302h train and validation subjects | SD302 test subjects, overall and by examiner assessment |
| A-prime: usable ridge | Same fitted models as Table A | Same test images, scored against quality masks |
| B1: SD302 exemplar | No supervised fitting to SD302g hulls; RA-RUnet uses labelled SD302h latent train subjects and unlabelled SD302b/SD302g train subjects | SD302g test subjects, expert-minutiae lower-bound metrics |
| B2: FVC | FVC set B | FVC set A |
| B3: SD302 live-scan | Labelled SD302b train/validation subjects | Labelled SD302b test subjects |
| C: palm | No palm fitting; reuse Table B models | Labelled SD302c test palms |
| D: cross-domain | No additional fitting | A models on B test data and B models on A test data |

If licensed FVC data or manual SD302b/SD302c annotations are unavailable, the
affected endpoint must read `pending data`. No unofficial data or proxy mask is
allowed.

For B1, field 9.300 full-canvas rectangles are not foreground masks. Convert
field 9.331 expert minutiae from 0.01 mm units with
`pixels = units * ppi / 2540` and form their convex hull. This hull is a
permitted lower-bound evaluation region, not pixel-accurate ground truth and not
a supervised training target. Only test subjects are scored. Any unlabelled
alignment target is restricted to SD302b/SD302g train subjects.

### 3.1 Cross-domain B1 scoring transaction

Each Dataset-1 seed is eligible for B1 only after its final checkpoint and
expanded-grid validation selector are frozen. A GT-blind inference manifest
contains exemplar image paths, PPI, subject assignments and source hashes but
has an empty `mask_path`; it exposes neither hull nor minutiae paths. An
immutable cross-domain authorization binds that manifest to the unchanged
Dataset-1 selector, checkpoint, run card, authoritative split and independently
verified SD302g coordinate build. The common neural runner retains one 500-ppi
binary mask, one pre-threshold float16 probability map and one timing record per
test exemplar. It never fits on SD302g or opens a hull/minutiae artifact.

Before scoring, all retained artifacts and the exact 475-image/40-subject test
universe are authenticated. The score wrapper then creates an exclusive
write-once journal and invokes `evaluate_hull.py` exactly once. Only that
evaluator may open the hull/minutiae artifacts. Publication requires the
Hull-as-prediction gate (MC = 100% and HR = 100%), no missing prediction, and
overall plus group/device/capture aggregation. MC, HR, PAR and OHF are the only
reported B1 metrics. A journal left without a completed bundle blocks automatic
retry and requires a documented versioned remediation.

## 4. Ground-truth validation

1. Parse SD302h Type-9 EFS annotations using declared ANSI/NIST separators and
   field tags.
2. Rasterise the field 9.300 polygon as ROI ground truth at native dimensions.
3. Rasterise field 9.308 cells above the declared no-ridge/background category
   as the quality ground truth. The annotation audit retains all 10,092 valid
   ROI annotations and 10,024 complete quality maps and lists all 68
   quality-only annotation exclusions. Aggregate these to 9,986 unique source
   images for Table A and 9,920 source images with complete constituent quality
   annotation for Table A-prime.
4. Preserve field 9.353 examiner assessment.
5. Convert physical coordinates with each image's recorded resolution and ROI
   offsets. Never adjust coordinates by visual trial.
6. Check dimensions, empty masks, polygon validity, area, and ROI/quality IoU;
   retain low-agreement cases and report them separately.
7. Visually inspect a fixed random overlay sample before model work. A systematic
   offset, orientation error, or failed reference validation is a stop condition.
8. Pass a GT-as-prediction test through the shared evaluator: Dice must equal 1,
   MDR 0, and FDR 0.

## 5. Shared preprocessing

- Convert to grayscale and normalise intensity to `[0, 1]`.
- Resample from manifest resolution to 500 ppi. Use area interpolation for
  reduction and nearest-neighbour interpolation for every mask operation.
  The 2026 SD302e full-resolution source filenames carry their actual capture
  PPI while SD302h masks use a 1,000-PPI annotation grid; record and apply
  `ppi` and `mask_ppi` separately, then align only the one-pixel rounding edge
  to the pre-audited 500-PPI image shape.
- Reject unresolved or contradictory resolution metadata; do not guess.
- Pixel training uses random `512 x 512` crops with padding where necessary.
- Shared augmentation is horizontal flip, rotation within 15 degrees, scale
  0.9--1.1, brightness/contrast within 20%, and light Gaussian noise.
- Full-image inference pads to a multiple of 32. If necessary, tile at 512 with
  64-pixel overlap.
- Apply the model's validation-selected sigmoid threshold and no test-time
  augmentation.
- Pre-cache every source image and available mask at 500 ppi as an atomic uint8
  NPY artifact in the restricted cache. The cache is usable only after a
  deterministic 20-image comparison is byte-identical to the existing
  on-the-fly preprocessing path; every entry is bound to the source, manifest,
  resolution, interpolation policy, and implementation hashes.
- New neural jobs use eight data-loader workers per process, pinned memory,
  persistent workers, and prefetch factor four. Aggregate workers across
  concurrent jobs must remain within the runtime-declared project budget while
  preserving capacity for the controller and reporting work.
- Save an unthresholded probability map (`uint8` 0--255 or `float16`) for every
  validation and test image from every final-table neural pixel or patch model
  and every successful seed, including released zero-shot networks when their
  official output exposes probabilities. Each NPY is finite, range checked,
  shape checked and bound to the manifest source-image SHA-256; its directory
  must contain exactly the manifest/split-derived image universe.
  For trainable models, select the threshold using validation probability maps
  only, then derive test binary masks from the stored test probabilities.
  Zero-shot SUFS retains its released decision rule; FingerNet is the declared
  zero-shot-weight threshold-calibration exception described below. A binary mask is never
  accepted as a probability map; an official runner that cannot expose its raw
  probability must be reported explicitly rather than reverse-engineered from
  the binary output.
- Save every evaluated prediction as a binary 0/255 PNG at 500 ppi.

### 5.1 Frozen comparison grid and evaluator

Every Dataset-1 table row is a full-image, pixel-level comparison on the same
canonical 500-ppi grid. Patch and block classifiers may operate internally on
16-by-16 blocks, but their thresholded block maps undergo the frozen shared
block cleanup and are expanded and cropped back to the complete image grid
*before* scoring. A block-grid PNG is never a valid evaluator input.

All Dataset-1 rows, including classical, pretrained, neural and trivial
references, pass through the same committed `evaluate_masks.py` bytes and the
same metric implementation. Production invocations enable
`--strict-prediction-grid`, set both prediction and target resolution to 500
ppi, and bind the exact evaluator commit, SHA-256, command line and output CSV
hashes in a write-once scoring receipt. A prose assertion is not a receipt:
the committed scoring wrapper must actually invoke the evaluator, record exit
status zero, bind manifest, split and prediction-set identities, and retain
stdout, stderr, per-image CSV and summary hashes. Strict mode rejects resizing, shape
mismatches and values other than 0/255. The prerelease comparability audit
requires the complete frozen method/seed inventory, independently verifies
every prediction shape, and checks that both Table A and Table A-prime use the
same prediction bytes. No table-specific inference or post-processing is
permitted.

## 6. Equal tuning policy

All selectable quantities must be fixed from training and validation data. The
test set is opened only after a run is frozen.

- Pixel models use BCE plus Dice, AdamW, cosine decay, weight decay `1e-4`, at
  most 100 epochs, and validation-Dice early stopping with patience 15 for full
  runs unless a documented method requirement conflicts.
- Every deep family receives the same learning-rate sweep: seed 0, exactly 30
  completed epochs, learning rates `3e-4` and `1e-3`. Generated native sweep
  configurations set patience to 30, so early stopping cannot fire during the
  fixed 30-epoch comparison. Validation Dice selects one rate before fresh full
  three-seed runs, which restore patience 15. Discarded sweep candidates never
  generate test predictions and are not final-table models. Their immutable run
  cards retain the complete validation threshold-score grid. The two preserved
  legacy-grid U-Net++ candidates additionally require retained raw validation
  probabilities and an expanded-grid selector receipt before rate selection.
- Every trainable neural pixel or patch family selects its mask threshold on validation data
  from `{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}`. This grid was prospectively
  extended on 2026-10-08 after a validation-only UNet++ sweep selected the old
  lower boundary of 0.3; no Stage 3 test prediction or metric had been opened.
  The same amended grid applies to every trainable pixel family, including
  completed Stage 1/2 models, whose frozen weights are re-evaluated on
  validation data before their test masks are regenerated. The chosen
  threshold is frozen before the corresponding final test inference and is
  reported in result metadata. The FingerNet weights and architecture remain
  zero-shot, but its segmentation threshold receives this same validation-only
  grid; the released 5-by-5 opening is applied after each candidate threshold.
  This threshold calibration is reported explicitly and does not update any
  network weight. The earlier fixed-0.5 FingerNet card remains immutable audit
  history but is superseded and selection-ineligible; the final row is pending
  replay from sealed validation and test probabilities through a validation
  selector and one fresh common-evaluator receipt. SUFS zero-shot retains its released decision rule, and
  algorithmic parameters are not silently moved into the neural grid.
- Pixel methods share the crop and augmentation distribution.
- Training remains fp32 by default. A method may use bf16 autocast plus
  channels-last only after a two-epoch validation-only comparison against fp32
  has absolute Dice difference below 0.005. The acceptance receipt is bound to
  every accelerated run card; absent or failed evidence forces fp32.
- Seeds 0, 1, and 2 are run for trainable methods; pilots use seed 0 only.
- Pretrained methods do not receive test-set fine-tuning.
- Classical models choose a configuration by validation balanced block
  accuracy (the arithmetic mean of foreground recall and background recall),
  with the predeclared candidate order breaking exact ties, and refit on
  training blocks. This family-specific selection criterion is frozen and is
  not replaced post hoc by validation image Dice. `run_classical.py` has no test-data
  or scoring path. Its resumable feature cache and frozen model cards are
  keyed by the manifest, split, sampling policy, seed, and implementation
  hashes. `predict_classical.py` subsequently generates masks without reading
  any ground-truth file.
- GMFS chooses its tuned configuration from a predeclared validation grid:
  the released default plus the 16 factorial corners formed by sigma
  `{3.0, 6.0}`, threshold `{0.15, 0.25}`, closing iterations `{4, 8}`, and
  opening iterations `{8, 16}`. These are symmetric low/high neighbours around
  the released defaults (`13/3`, `0.2`, `6`, `12`). Macro validation Dice is
  primary and a lexicographic parameter tuple breaks exact ties.
- SUFS fine-tuning starts from the SHA-256-pinned PyFing 0.6 weights and uses
  its unchanged Keras architecture through the Torch backend. It receives the
  same train-only normalisation, crops, augmentations, learning-rate sweep and
  threshold grid as the reimplemented pixel models. Because its final layer is
  sigmoid, BCE and soft Dice consume probabilities rather than logits.
- SUFS fine-tuning and all neural checkpoint selection use validation Dice.
- A selected pixel threshold and the fixed block post-processing are never
  retuned on a test domain.
- Any method-specific departure must be frozen before scoring and printed in
  its row label or run card.

### 2026-10-09 v3 threshold-policy supersession

The original 2026-10-08 threshold amendment remains immutable. Its statement
that released zero-shot thresholds were fixed is superseded only for
FingerNet, following the subsequently recorded v3 operator requirement:
FingerNet uses the same seven-value validation grid as every other applicable
neural row. This is threshold calibration only; the released weights remain
unchanged. Released SUFS zero-shot retains its released decision rule. A new
append-only amendment must bind the exact original amendment SHA-256 and the
append-only authorization log. Every v3 validation replay plan and FingerNet
selector binds both the superseding amendment and its parent SHA-256. Neither
the amendment publisher nor the replay planner opens test inputs, test labels,
probability maps, or result metrics.

Validation probability replay is two-phase for any legacy full run. Both
learning-rate sweep checkpoints are replayed and receive a threshold chosen on
the expanded grid before an existing full checkpoint becomes eligible. The
full checkpoint is replayed only when its training learning rate equals the
new validation-selected rate; otherwise it is marked ineligible and a fresh
full run is required. This applies equally to U-Net and SUFS fine-tuning.
Released SUFS retains its released threshold: its validation probability maps
are retained and authenticated without grid tuning, while the corresponding
test-export command remains sealed until the validation receipt exists.

If only the preprocessing-cache implementation identity changed, the cache is
not accepted on argument alone. A committed validator must compare the exact
cached and on-the-fly image/mask arrays for 50 deterministic validation images,
record the 50 private identifiers and artifact hashes, bind both implementation
identities, and establish byte identity. Any difference fails closed. This
compatibility proof never opens test or held-out inputs and cannot waive a
manifest, split, target-PPI, mask-column, or cached-artifact mismatch.

## 7. Method registry and declared deviations

| Row | Frozen implementation | Fidelity note |
|---|---|---|
| GMFS default | pyfing defaults | Pretrained/algorithmic baseline |
| GMFS tuned | Four native parameters chosen on validation | Separate from default row |
| SUFS | Released pyfing weights, zero-shot | No test adaptation |
| SUFS fine-tuned | Same network adapted on train, stopped on validation | Additional benchmark row |
| FingerNet | Official Tang et al. segmentation probabilities and post-processing | Released Keras 2.0.2 weights, inference-only compatibility graph; zero-shot weights with validation-only threshold calibration; N/A rather than substitution on failure |
| U-Net | Four levels, base 32, BatchNorm/ReLU, bilinear upsampling plus convolution | Reimplementation |
| UNet++ | Base 32, nested dense skips, four equally weighted training heads, final inference head | Reimplementation |
| DRUnet | R2U-Net blocks, two recurrences, dropout 0.2 | Deterministic table inference |
| Recurrent adversarial | Recurrent backbone plus gradient-reversal bottleneck discriminator | Table A is labelled RUnet without alignment |
| LeNet | LeNet-5-style, `32 x 32` context to centred block label | Reimplementation |
| AlexNet | Five convolution and three fully connected layers, dropout, `64 x 64` context | Adapted patch classifier |
| CNN plus autoencoder | Train-only unsupervised `64 x 64` patch pretraining, then block classifier | No validation/test pretraining |
| SegFormer-B2 | ImageNet-pretrained, one-channel adapter | Explicit 2-D substitute for the OCT transformer |
| SVM | Shared eight features, Nyström RBF approximation plus linear SVM | Not exact RBF SVC |
| Random forest | Shared eight features and validation-selected tree parameters | Reimplementation |
| WELM | Shared eight features, W1 weighting | Reimplementation |

For Table B1 recurrent alignment, the labelled source is restricted to SD302h
latent training subjects and the unlabelled target is restricted to SD302b and
SD302g training subjects. The SD302g hull is evaluation-only and is never a
training target. For Table A, no target-domain alignment is run. At the current
benchmark freeze, the team-supplied SD302b archive was unavailable after three
authorized transfer failures, so the aligned RA-RUnet row remains pending. The
target definition is not weakened to SD302g-only alignment. The same block
cleanup is applied to every block method: closing, opening, hole filling, then
removal of components under four blocks.

All-background and all-foreground references are generated and scored for each
table but are not ranked as learned methods.

**AlexNet deterministic-pooling remediation.** The original learning-rate
sweep stopped before epoch 1 because deterministic backward is unavailable for
`AdaptiveAvgPool2d`. With the protocol's fixed `64 x 64` input, the pre-pool
feature tensor is always `4 x 4`; adaptive pooling to `2 x 2` was therefore
replaced by `AvgPool2d(kernel_size=2, stride=2)`. A unit test on 100 seeded
inputs measured maximum absolute difference `0.0` between the two operators.
The remediation changes deterministic execution only and preserves the
computed architecture at the declared input shape. Original failure evidence
is retained and the replacement uses a fresh recovery ledger. This disclosure
does not claim learning-rate selection, validation performance, or a held-out
result; the AlexNet row remains provisional until the ordinary validation-only
selection and one-time scoring receipts exist.

For classical estimators, the required parameter-size column is labelled
**fitted representation scalars (M)**. This is an exact count of numeric values
needed by the fitted inference representation (scaler and Nyström state plus
classifier state, or every scalar field in the stored forest node records plus
their node-value arrays). It is not
presented as a neural trainable-parameter equivalent.

## 8. Mask scoring

Every method must be scored by the common mask evaluator. Missing masks are
converted to all-background predictions and counted. For foreground as the
positive class:

- `Accuracy = (TP + TN) / (TP + FP + FN + TN)`
- `Error = 1 - Accuracy`
- `Dice = 2TP / (2TP + FP + FN)`
- `Jaccard = TP / (TP + FP + FN)`
- `background IoU = TN / (TN + FP + FN)`
- `MIoU = (Jaccard + background IoU) / 2`
- `MDR = FN / (TP + FN)`
- `FDR = FP / (FP + TN)`

FDR is the background false-detection rate, not statistical false discovery.
The evaluator's empty-class conventions are immutable. Report image-macro mean
and population standard deviation as primary, pooled-pixel values as secondary,
and evaluated/missing counts for every row. Aggregate three trainable seeds as
mean plus standard deviation.

Before publication, `audit_dataset1_comparability.py` must issue a passing,
write-once receipt. It fails closed unless all 18 presentation rows and their
frozen seed universes are present; subjects are disjoint; every scored PNG is a
full-image binary mask at 500 ppi; every scoring receipt invokes the same
strict evaluator; patch masks independently reproduce the retained block
probabilities after shared post-processing and expansion; patch sample cards
bind only training subjects; and every neural selector score independently
recomputes from all validation images on the expanded grid. This receipt is a
release prerequisite, not a new selection signal.

The independent validation-sanity audit is also required for every pixel and
patch model before Stage 4 aggregation. It authenticates the selected
validation probability card and replays macro Dice on the complete amended
threshold grid `{0.1, 0.2, ..., 0.7}`. It reports the selected threshold,
validation Dice at every threshold, predicted foreground fraction,
predicted-area/ground-truth-area ratio, and training-curve endpoints and
slopes. Within each prespecified similarity family it flags an absolute
selected-validation-Dice gap greater than 0.04 for review; the flag is a
diagnostic and cannot change selection. Ten validation images are selected
deterministically by the hash rule recorded in the private receipt for later
side-by-side overlays. Missing, partial, binary-only, non-validation, or
unbound probability cards fail closed. This audit never reads test inputs,
predictions, labels, or metrics.

### SD302g lower-bound scoring

Never run the pixel-mask metrics above against a minutiae hull. For expert
minutiae set `M`, hull pixels `H`, and predicted foreground `P`, report:

- `MC = |{m in M: m lies in P}| / |M|`;
- `HR = |H intersect P| / |H|`;
- `PAR = |P| / |H|`; and
- `OHF = |P outside H| / |P|`.

The reference rows are all-background, all-foreground, and hull-as-prediction.
Hull-as-prediction must give MC and HR equal to one. All-foreground demonstrates
why MC or HR alone is gameable. The preregistered presentation is therefore MC
and HR together with PAR, with OHF descriptive; there is no scalar ranking and
no post-hoc dilation cutoff. Aggregate overall, by baseline/challenger group,
by device code, and by every capture label actually observed in the release
(currently roll and slap); do not rename a release label to fit a presumed list.

## 9. Uncertainty and comparisons

Use 1,000 paired subject-level bootstrap replicates for 95% confidence
intervals on headline metrics and method differences. A bootstrap draw samples
test subjects with replacement and includes all their images. Use the same draw
for both methods in a paired contrast. Freeze and record the bootstrap seed
before accessing results.

For the prespecified Table A comparison, identify the highest observed
image-macro Dice row, compare it with every other eligible method using paired
two-sided Wilcoxon signed-rank tests on per-image Dice, and apply Holm correction
over that family. Store raw p-values, adjusted p-values, paired effect sizes,
and subject-bootstrap intervals. Do not use significance to alter the selected
method or rerun its hyperparameter search.

The Stage 4 statistical producer accepts only an immutable input specification
bound to the preregistered plan, frozen input lock, and passing summary-universe
preflight receipt. It verifies confusion-derived per-image metrics and the
exact table, group, subject, image, method, and seed universes before averaging
successful seeds per source image. It then applies the one frozen subject draw
matrix to every method, metric, table, subgroup, and paired contrast. An
unavailable method remains an explicit `N/A` row and remains in the fixed Holm
family denominator. Aggregate outputs are write-once and undergo a scan for
source image identifiers, subject identifiers, and private filesystem paths.

### Incremental seed-0 held-out delivery

G3 delivery and G4 statistical eligibility are separate gates. As soon as all
validation-only selection evidence for a final seed-0 checkpoint is frozen, a
ground-truth-blind seed-0 plan may generate its one retained test probability
set. The common threshold scorer then evaluates that seed exactly once and
publishes a write-once per-seed delivery. The single frozen mask set is passed
unchanged to both the examiner-ROI (A) and usable-ridge (A-prime) evaluator
invocations. A scoring journal is created before the evaluator starts; an
incomplete transaction cannot be retried
automatically. U-Net seed 0 is imported from its separately reselected Stage 2
delivery and is never rescored by this path.

A seed-0 readiness pass is explicitly not a G4 pass. The monolithic Stage 3
importer remains fail-closed until the original readiness gate proves at least
two successful seeds, including seed 0, for every native method. It then authenticates each
method/seed against that gate, copies per-image evaluator files byte-for-byte,
and records every imported seed as `imported_not_rescored`. Later seeds can be
added only through their own one-time deliveries; importing them cannot invoke
inference, threshold selection, or evaluation. Thus the early seed-0 score is
reused immutably when the multi-seed Stage 4 bundle is assembled.

## 10. Timing protocol

1. Use batch size one and disable test-time augmentation.
2. Run ten untimed warm-up images.
3. Time every test image, including decode/resampling or feature extraction,
   inference, and post-processing; exclude model construction and checkpoint
   loading.
4. Report arithmetic mean milliseconds per image.
5. Restrict CPU methods to one inference thread.
6. Record only `CPU` or `GPU` in the public device column.
7. For every training epoch, separately record data-wait and model-compute
   seconds plus generic device-utilisation samples. These diagnostics are not
   selection metrics and contain no device model or host identifier.

Timing must be rerun when code that affects preprocessing, inference, or
post-processing changes.

## 10.1 Shared-GPU scheduling

Long-running neural jobs use only runtime-authorized devices. The scheduler may
run up to three project jobs concurrently on each authorized device, with a
per-process allocator fraction no greater than the registered runtime limit. It
may fill a slot only after checking measured device memory, aggregate declared
RAM and CPU-worker reservations against their runtime caps, the registered free
disk floor, and the absence of an unowned process above the registered foreign-
process threshold. A newly detected foreign process causes project jobs to
checkpoint at an epoch boundary and vacate that device. During a legacy-worker
adoption, the scheduler counts each captured worker's original loader count
rather than assuming the amended value, and preserves non-loader capacity for
the controller and reporting work. Every new launch is blocked unless the
immutable 500-ppi cache receipt, threshold-policy amendment, clean code commit,
and exact generated-config hashes still match the queue plan. Held-out timed inference is
scheduled separately at one process per device; training concurrency is never
used for the reported `ms/img` measurements.
Apart from the separately pre-registered uniform cache/loader and threshold
amendments above, concurrency changes scheduling only; model definitions,
sample order, seeds, selection rules, and metrics remain identical.

An OOM retry has one hash-bound form for every trainable runner: the base
micro-batch must be even, the sole retry halves that micro-batch and doubles
gradient accumulation, and the effective batch is unchanged. The queue ledger,
launcher environment, resolved config, checkpoints, and run card all bind the
same stable contract and active state. Only base-to-retry is allowed; arbitrary
scales, a second retry, or a retry-to-base resume fail closed.

### 2026-10-09 source-complete autonomous validation handoff

Validation-probability replay must not co-run with the authenticated
AlexNet/SegFormer mixed-recovery controller. Although the validation controller
could account for recovery workers, the already-running recovery controller
was not launched with reciprocal ownership of validation workers. A replay
worker could therefore be classified as foreign and trigger a cooperative
checkpoint of the source jobs. That asymmetric coexistence plan was rejected
before any replay worker launched.

The handoff now enforces a source-completion barrier. All four authenticated
recovery cells must complete and their source controller must exit normally
before the first replay process may start. The handoff authenticates the final
source ledger and every declared completion artifact; retry, pre-emption,
failure, identity change, a torn ledger snapshot, an unowned process, or a
resource/disk guard failure stops the handoff. It never signals a source worker
or controller and never writes the source ledger. After the barrier passes,
the replay controller may use the registered capacity under the ordinary
process, CPU, RAM, and disk guards.

The first replay wave contains only the already frozen validation candidates.
Its finalizer reads completion and validation-selector metadata only. A second,
append-only replay plan is compiled after both immutable sweep run cards exist
for DRUnet, RUnet, AlexNet and SegFormer; it never edits the first plan. Full
seed-0 training jobs are merely prepared from the validation-selected rates.
They are not launched by the handoff, and no test command is authorized.

The queue performs its device-telemetry dependency/NVML preflight before it
creates or updates a ledger. A stop signal is rechecked immediately before
each process spawn; all locally running jobs receive their cooperative
preemption marker, and a pre-launch ledger transition is rolled back without
consuming an attempt when the stop arrives. Infrastructure failures are never
repaired by editing the failed ledger. `prepare_queue_recovery.py` instead
creates a new, immutable queue and ledger that hash-bind the prior queue,
failed ledger, and incident note; completed artifact hashes are preserved and
only unfinished jobs receive a fresh retry budget.

The two preserved U-Net++ sweep workers retain their launch-era configuration
and therefore cannot be selected from their legacy edge-threshold run metric.
For each frozen best checkpoint, first run `infer_masks.py` on the validation
split with `--probability-out`, the verified preprocessing cache, and the exact
threshold amendment. Then run `select_threshold_from_probabilities.py` over the
raw validation probability card. `prepare_deep_jobs.py select-full` requires
both selector receipts via `--unetpp-selector-receipts` plus the same
`--threshold-amendment` (with its expected SHA-256) and verified cache via
`--cache-dir`; every newly
emitted full-training argv binds both inputs. It chooses the learning rate only
from the two amended validation Dice values. The old run-card metric is
retained for audit but is explicitly selection-ineligible. Test images and
labels are not consulted in this handoff.

Before held-out inference, a write-once threshold-selector manifest binds each
eligible amended run to its raw validation probability card, selected
validation card, expanded-grid selector receipt, cache receipt, and threshold
amendment. The Stage 3 inference planner requires exact coverage, carries
`--threshold-receipt`, `--threshold-amendment`, and `--cache-dir` into every
test job, and rejects extra or missing run records. The inference wrapper
propagates all three inputs to both its resumable completion pass and its fresh
timed pass.

## 11. Failure and QA rules

- A failed method does not stop independent jobs.
- After the declared implementation effort is exhausted, report `N/A` with the
  exact technical reason; do not silently replace it.
- A missing prediction is penalised as all-background.
- Save a fixed overlay sample and the three lowest-Dice test cases per method in
  the restricted review area.
- Keep low ROI/quality agreement cases; analyse rather than exclude them.
- Any test-driven code or configuration change invalidates the affected final
  score and requires a fresh, documented benchmark version.

## 12. Reproducibility record

Each run card must contain:

- source commit and dirty-state flag;
- manifest, split, ground-truth, configuration, and checkpoint hashes;
- method key, table, seed, selected validation parameters, and stopping epoch;
- package versions and exact command;
- prediction count, missing count, evaluator version, and elapsed time; and
- whether the implementation is official, pretrained, reimplemented, adapted,
  or substituted.

The reproducible stage order is:

1. build and validate examiner ground truth;
2. convert and validate the frozen subject split;
3. generate masks for one method and seed;
4. score masks through the common evaluator;
5. build aggregate tables from evaluator outputs; and
6. run the frozen statistical analysis and report builder.

Deep jobs are materialised by `prepare_deep_jobs.py prepare-sweep`. It emits a
separate two-job U-Net pilot queue before the remaining model queue; each cell
is seed 0, exactly 30 completed epochs, and exactly one of `3e-4` and `1e-3`.
`prepare_deep_jobs.py select-pilot` reads only those two fixed-path U-Net run
cards and emits the frozen 100-epoch U-Net seed-0 job, so the pilot gate does
not depend on any later method finishing its sweep. The selection artifact
records the run-card, sweep-config, checkpoint, manifest, split, and full-config
hashes and contains no test metric.
`prepare_deep_jobs.py select-full` refuses incomplete, debug-limited, mismatched,
or hash-invalid sweep cards, including any card whose requested and completed
epoch counts are not both 30, selects solely by best validation Dice (lower
predeclared learning rate breaks an exact tie), then emits separate queues for
the U-Net seed-0 gate, remaining seed-0 runs, and seeds 1--2. The full runs are
fresh 100-epoch schedules with patience 15 rather than continuations of a
30-epoch scheduler.
SUFS fine-tuning uses its dedicated audited Keras-Torch path. FingerNet remains
a separate pretrained legacy-inference workflow and is printed in every
generated plan rather than silently disappearing.

Commands accept manifests, splits, configurations, and output locations as
arguments. No private location, machine identity, or device product is embedded
in source or documentation. Biometric inputs, masks, manifests, checkpoints,
per-image outputs, and private qualitative reports remain outside version
control.

### Public one-command entry points

Tables A, A-prime, B1 and D are dispatched by the corresponding
`scripts/reproduce_table_*.sh` command and the private-plan schema documented
in `REPRODUCTION.md`. The dispatcher is execution-only: it cannot tune a model,
select a checkpoint, or select a threshold. Neural checkpoints are coupled to
their validation-only selector receipt and threshold-policy amendment; fitted
artifacts are hash-checked before use. All generated or supplied predictions
must cover the complete selected test partition before the locked evaluator is
called. This makes the evaluator's missing-as-background behavior an explicit
standalone policy rather than an accidental way for a reproduction run to
proceed with missing files.

The Table A-prime entry point hash-binds a successful Table A receipt and must
reuse its exact run key, method, prediction template and prediction PPI; only
`gt_column` changes. B1 always dispatches the lower-bound hull evaluator,
which emits its three references and never emits pixel-overlap metrics. Table D
entries declare their target evaluator explicitly because its cross-domain
directions have different available ground truth. The dry-run form validates
paths, artifact hashes, split syntax, target resolution and dispatch while
printing the exact commands; it does not open test images or write results.

## 13. Provisional-publication amendment (2026-10-10)

The v1.0 public release may contain a receipt-authenticated aggregate snapshot
before every planned seed is complete. This changes publication timing only;
it does not change model selection, threshold selection, test scoring, metrics,
or statistics. Completed test aggregates are preserved exactly, unfinished
rows are labelled `pending` with their completed-seed count, and no missing
value is inferred. The same controlled aggregate importer regenerates the
tables and website for v1.1 after the remaining frozen work completes.
