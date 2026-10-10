# Methods

This page maps every Fig. 2 entry to its source, implementation, optimisation
budget, and declared deviation. Final parameters below are read from the
committed method configurations. All learned methods use train subjects only;
selection and thresholding use validation subjects only. Pixel-model thresholds
share the grid 0.1–0.7 in increments of 0.1. Where thresholds or classical
parameters differ by seed, final applied values from the configuration's
per-seed selection map are listed in seed 0/1/2 order.

## CNN and autoencoders — Chhabra et al. (2023)

- **Architecture/input:** a stacked convolutional autoencoder on 64×64 centred
  patches, followed by its encoder and a block classifier for a 16×16 region.
- **Objectives:** train-only reconstruction pretraining, then supervised binary
  block classification with balanced sampling.
- **Search/selection:** learning rate in {3e-4, 1e-3}; validation full-image
  Dice selects the learning rate and threshold. Chosen learning rate: 1e-3;
  thresholds: 0.3/0.3/0.3.
- **Deviation:** the shared 16×16 output block and common post-processing make
  this row directly comparable to the other patch methods.

## GMFS and SUFS — Cappelli (2023)

- **GMFS input/objective:** the released algorithm operates on full 500-ppi
  grayscale images and has no learned loss. One row uses released defaults;
  another selects its four native parameters on validation only.
- **SUFS input/objective:** the released SUFS segmentation network is evaluated
  zero-shot; a separate row fine-tunes it with BCE plus Dice.
- **Search/selection:** the declared GMFS validation grid; SUFS fine-tuning uses
  learning rate in {3e-4, 1e-3} and the common threshold grid. GMFS defaults
  were sigma 4.333333333333333, threshold 0.2, 12 openings, 6 closings and
  percentile 95. The validation-tuned GMFS row used sigma 3.0, threshold 0.15,
  8 openings, 8 closings and percentile 95 at 500 ppi. Zero-shot SUFS retained
  its fixed threshold 0.5. Fine-tuned SUFS used learning rate 1e-3 and
  thresholds 0.1/0.1/0.1.
- **Deviation:** none for zero-shot SUFS/GMFS. Fine-tuning is explicitly a
  separate benchmark row. The zero-shot SUFS threshold is released-fixed; the
  fine-tuned threshold is validation-selected.

## U-Net — Jaafar et al. (2022); Ronneberger et al. (2015)

- **Architecture/input:** four down/up levels, base width 32, BatchNorm/ReLU,
  bilinear upsampling plus convolution, and random 512×512 training crops.
- **Objective:** BCE plus Dice; full-image inference with padded or overlapping
  tiles where necessary.
- **Search/selection:** learning rate in {3e-4, 1e-3}; common validation
  threshold grid. Chosen learning rate: 1e-3; thresholds: 0.1/0.1/0.1.
- **Deviation:** architecture reimplementation rather than source weights.

## Adaptive Gaussian transformer — Liu et al. (2022)

- **Architecture/input:** **SegFormer-B2 (substitute†)** from Xie et al. (2021),
  ImageNet initialisation, one-channel adapter, and 512×512 crops.
- **Objective:** BCE plus Dice.
- **Search/selection:** learning rate in {3e-4, 1e-3}; common validation
  threshold grid. Chosen learning rate: 3e-4; thresholds: 0.3/0.1/0.2.
- **Deviation:** the reviewed adaptive Gaussian transformer targets volumetric
  OCT layers and is not a 2-D fingerprint segmenter. This row is a declared
  substitute, never described as a reproduction.

## DRUnet — Joshi et al.

- **Architecture/input:** recurrent residual U-Net, two recurrent steps per
  block, dropout 0.2, and 512×512 crops.
- **Objective:** BCE plus Dice.
- **Search/selection:** learning rate in {3e-4, 1e-3}; common validation
  threshold grid. Chosen learning rate: 1e-3; thresholds: 0.2/0.1/0.1.
- **Deviation:** table inference disables dropout. MC-dropout variance maps are
  a separately labelled analysis and do not change table masks.

## Recurrent adversarial network — Joshi et al. (2021)

- **Architecture/input:** the recurrent backbone above, without dropout, plus
  a bottleneck discriminator connected by gradient reversal.
- **Objectives:** BCE plus Dice on labelled source images and binary domain loss
  on source/target bottleneck features.
- **Search/selection:** the common learning-rate and threshold grids. The
  no-alignment RUnet used learning rate 1e-3 and thresholds 0.4/0.1/0.1. The
  aligned RA-RUnet row is unavailable and therefore has no chosen checkpoint
  parameters in the committed configuration.
- **Deviation:** Table A reports `RUnet (no alignment)` because no target domain
  exists there. RA-RUnet was reserved for the declared latent-to-live-scan
  experiment with unlabelled train-subject target images and no test subjects;
  no aligned result is reported without an authenticated aligned checkpoint.

## Deep Nested UNets — Liu & Qian (2020); Zhou et al. (2018)

- **Architecture/input:** native four-level UNet++, base width 32, nested dense
  skips, four equally weighted deep-supervision heads, and 512×512 crops.
- **Objective:** BCE plus Dice averaged across the training heads.
- **Search/selection:** learning rate in {3e-4, 1e-3}; common validation
  threshold grid. Chosen learning rate: 1e-3; thresholds: 0.1/0.1/0.1.
- **Deviation:** only the final head is used at inference, as preregistered.

## Support vector machine — Cortes & Vapnik (1995)

- **Architecture/input:** Nyström RBF features followed by a balanced linear
  SVM over the shared eight-dimensional 32×32-context block descriptor.
- **Objective:** margin loss in the linear SVM approximation.
- **Search/selection:** RBF scale and regularisation are selected by validation
  balanced block accuracy. The selected C was 10.0 for every seed; gamma was
  0.05/0.2/0.05.
- **Deviation:** the exact RBF kernel is approximated for tractability; block
  features follow the orientation-analysis family of Bazen & Gerez (2001).

## AlexNet and LeNet — Serafim et al. (2019)

- **Architecture/input:** LeNet receives a 32×32 centred patch; the adapted
  five-convolution AlexNet receives 64×64. Each predicts one 16×16 block.
- **Objective:** balanced binary block classification.
- **Search/selection:** learning rate in {3e-4, 1e-3}; thresholds are selected
  from full validation images after block upsampling. AlexNet used learning
  rate 3e-4 and thresholds 0.3/0.3/0.3; LeNet used learning rate 1e-3 and
  thresholds 0.4/0.4/0.4.
- **Deviation:** AlexNet's fixed average pool replaces adaptive average pool at
  the fixed input size after an output-equivalence test. The shared 16×16 block
  differs from some source implementations and is explicitly disclosed.

## Deep CNN / FingerNet — Tang et al. (2017)

- **Architecture/input:** official released FingerNet weights and segmentation
  head in its isolated legacy-compatible runtime; full 500-ppi images.
- **Objective:** no training in this benchmark; external source training is
  recorded as provenance.
- **Search/selection:** only the common mask threshold is selected on validation.
  Chosen threshold: 0.1; released weights were not updated.
- **Deviation:** none. Runtime incompatibility yields N/A, never a substitute.

## Random decision forest — Sankaran et al. (2017)

- **Architecture/input:** balanced random forest over the same block descriptor
  used by SVM and WELM.
- **Objective:** supervised tree impurity reduction.
- **Search/selection:** depth and minimum leaf size use the common classical
  validation budget. All three seeds selected maximum depth 20 and minimum
  leaf size 5.
- **Deviation:** reimplementation in the source method's spirit with shared
  features and post-processing for controlled comparison.

## Weighted extreme learning machine — Ezeobiejesi & Bhanu (2016)

- **Architecture/input:** single random hidden layer over the common block
  descriptor; centred 16×16 output blocks.
- **Objective:** weighted closed-form output regression with inverse class
  frequency (W1) weights.
- **Search/selection:** hidden width and regularisation use the common classical
  validation budget. All three seeds selected C=10,000; hidden widths were
  500/1,000/1,000.
- **Deviation:** reimplementation with shared features and post-processing.

## Shared inference and evaluation

Pixel networks run on full images padded to a multiple of 32, or use 512-pixel
tiles with 64-pixel overlap when needed. Patch predictions are upsampled to
pixels before the same evaluator is called. No test-time augmentation is used.
Every final metric is produced by `scripts/evaluate_masks.py`; SD302-Exemplar
uses only the lower-bound metrics in `scripts/evaluate_hull.py`.
