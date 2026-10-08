# Methods

This page maps every Fig. 2 entry to its source, implementation, optimisation
budget, and declared deviation. Final validation-selected parameters are marked
**pending** until imported from frozen aggregate receipts. All learned methods
use train subjects only; selection and thresholding use validation subjects
only. Pixel-model thresholds share the grid 0.1–0.7 in increments of 0.1.

## CNN and autoencoders — Chhabra et al. (2023)

- **Architecture/input:** a stacked convolutional autoencoder on 64×64 centred
  patches, followed by its encoder and a block classifier for a 16×16 region.
- **Objectives:** train-only reconstruction pretraining, then supervised binary
  block classification with balanced sampling.
- **Search/selection:** learning rate in {3e-4, 1e-3}; validation full-image
  Dice selects the learning rate and threshold. Chosen values: **pending**.
- **Deviation:** the shared 16×16 output block and common post-processing make
  this row directly comparable to the other patch methods.

## GMFS and SUFS — Cappelli (2023)

- **GMFS input/objective:** the released algorithm operates on full 500-ppi
  grayscale images and has no learned loss. One row uses released defaults;
  another selects its four native parameters on validation only.
- **SUFS input/objective:** the released SUFS segmentation network is evaluated
  zero-shot; a separate row fine-tunes it with BCE plus Dice.
- **Search/selection:** the declared GMFS validation grid; SUFS fine-tuning uses
  learning rate in {3e-4, 1e-3} and the common threshold grid. Chosen values:
  **pending**.
- **Deviation:** none for zero-shot SUFS/GMFS. Fine-tuning is explicitly a
  separate benchmark row. Zero-shot threshold selection on validation is
  disclosed.

## U-Net — Jaafar et al. (2022); Ronneberger et al. (2015)

- **Architecture/input:** four down/up levels, base width 32, BatchNorm/ReLU,
  bilinear upsampling plus convolution, and random 512×512 training crops.
- **Objective:** BCE plus Dice; full-image inference with padded or overlapping
  tiles where necessary.
- **Search/selection:** learning rate in {3e-4, 1e-3}; common validation
  threshold grid. Chosen values: **pending**.
- **Deviation:** architecture reimplementation rather than source weights.

## Adaptive Gaussian transformer — Liu et al. (2022)

- **Architecture/input:** **SegFormer-B2 (substitute†)** from Xie et al. (2021),
  ImageNet initialisation, one-channel adapter, and 512×512 crops.
- **Objective:** BCE plus Dice.
- **Search/selection:** learning rate in {3e-4, 1e-3}; common validation
  threshold grid. Chosen values: **pending**.
- **Deviation:** the reviewed adaptive Gaussian transformer targets volumetric
  OCT layers and is not a 2-D fingerprint segmenter. This row is a declared
  substitute, never described as a reproduction.

## DRUnet — Joshi et al.

- **Architecture/input:** recurrent residual U-Net, two recurrent steps per
  block, dropout 0.2, and 512×512 crops.
- **Objective:** BCE plus Dice.
- **Search/selection:** learning rate in {3e-4, 1e-3}; common validation
  threshold grid. Chosen values: **pending**.
- **Deviation:** table inference disables dropout. MC-dropout variance maps are
  a separately labelled analysis and do not change table masks.

## Recurrent adversarial network — Joshi et al. (2021)

- **Architecture/input:** the recurrent backbone above, without dropout, plus
  a bottleneck discriminator connected by gradient reversal.
- **Objectives:** BCE plus Dice on labelled source images and binary domain loss
  on source/target bottleneck features.
- **Search/selection:** the common learning-rate and threshold grids. Chosen
  values: **pending**.
- **Deviation:** Table A reports `RUnet (no alignment)` because no target domain
  exists there. RA-RUnet is used only for the declared latent-to-live-scan
  experiment, with unlabelled train-subject target images and no test subjects.

## Deep Nested UNets — Liu & Qian (2020); Zhou et al. (2018)

- **Architecture/input:** native four-level UNet++, base width 32, nested dense
  skips, four equally weighted deep-supervision heads, and 512×512 crops.
- **Objective:** BCE plus Dice averaged across the training heads.
- **Search/selection:** learning rate in {3e-4, 1e-3}; common validation
  threshold grid. Chosen values: **pending**.
- **Deviation:** only the final head is used at inference, as preregistered.

## Support vector machine — Cortes & Vapnik (1995)

- **Architecture/input:** Nyström RBF features followed by a balanced linear
  SVM over the shared eight-dimensional 32×32-context block descriptor.
- **Objective:** margin loss in the linear SVM approximation.
- **Search/selection:** RBF scale and regularisation are selected by validation
  balanced block accuracy. Chosen values: **pending**.
- **Deviation:** the exact RBF kernel is approximated for tractability; block
  features follow the orientation-analysis family of Bazen & Gerez (2001).

## AlexNet and LeNet — Serafim et al. (2019)

- **Architecture/input:** LeNet receives a 32×32 centred patch; the adapted
  five-convolution AlexNet receives 64×64. Each predicts one 16×16 block.
- **Objective:** balanced binary block classification.
- **Search/selection:** learning rate in {3e-4, 1e-3}; thresholds are selected
  from full validation images after block upsampling. Chosen values: **pending**.
- **Deviation:** AlexNet's fixed average pool replaces adaptive average pool at
  the fixed input size after an output-equivalence test. The shared 16×16 block
  differs from some source implementations and is explicitly disclosed.

## Deep CNN / FingerNet — Tang et al. (2017)

- **Architecture/input:** official released FingerNet weights and segmentation
  head in its isolated legacy-compatible runtime; full 500-ppi images.
- **Objective:** no training in this benchmark; external source training is
  recorded as provenance.
- **Search/selection:** only the common mask threshold is selected on validation.
  Chosen threshold: **pending**.
- **Deviation:** none. Runtime incompatibility yields N/A, never a substitute.

## Random decision forest — Sankaran et al. (2017)

- **Architecture/input:** balanced random forest over the same block descriptor
  used by SVM and WELM.
- **Objective:** supervised tree impurity reduction.
- **Search/selection:** depth and minimum leaf size use the common classical
  validation budget. Chosen values: **pending**.
- **Deviation:** reimplementation in the source method's spirit with shared
  features and post-processing for controlled comparison.

## Weighted extreme learning machine — Ezeobiejesi & Bhanu (2016)

- **Architecture/input:** single random hidden layer over the common block
  descriptor; centred 16×16 output blocks.
- **Objective:** weighted closed-form output regression with inverse class
  frequency (W1) weights.
- **Search/selection:** hidden width and regularisation use the common classical
  validation budget. Chosen values: **pending**.
- **Deviation:** reimplementation with shared features and post-processing.

## Shared inference and evaluation

Pixel networks run on full images padded to a multiple of 32, or use 512-pixel
tiles with 64-pixel overlap when needed. Patch predictions are upsampled to
pixels before the same evaluator is called. No test-time augmentation is used.
Every final metric is produced by `scripts/evaluate_masks.py`; SD302-Exemplar
uses only the lower-bound metrics in `scripts/evaluate_hull.py`.
