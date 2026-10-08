# Methods

All selected values below remain **pending** until imported from validation-only
selection receipts. Architecture definitions are frozen in `configs/`.

## Classical and algorithmic

### GMFS

The released pyfing implementation is reported with its default parameters and
as a separate validation-tuned row. The tuned row receives only the declared
four-parameter grid. Chosen parameters: **pending**.

### Support vector machine

The SVM consumes the shared eight-dimensional block descriptor and approximates
an RBF kernel with Nyström features before a balanced linear SVM. Selection
metric: validation balanced block accuracy. Chosen configuration: **pending**.

### Random forest

The forest consumes the same block descriptor and receives the same validation
selection boundary as the other classical rows. Chosen configuration: **pending**.

### Weighted ELM

The single-hidden-layer ELM uses inverse-class-frequency W1 sample weighting.
Chosen hidden width and regularisation: **pending**.

## Patch CNNs

### LeNet and AlexNet

LeNet uses a 32-pixel context window; AlexNet uses a 64-pixel context window.
Both predict the label of the centred 16-pixel block, use balanced train-only
sampling and share the fixed block post-processing. AlexNet's fixed average
pool replaces adaptive average pool only after an output-equivalence test.

### CNN + stacked autoencoder

The convolutional autoencoder is pretrained without labels on train-subject
patches only. Its encoder is then fine-tuned with a block classifier. Validation
and test patches never enter unsupervised pretraining.

## Encoder–decoder models

### U-Net

Four encoder/decoder levels, base width 32, BatchNorm/ReLU blocks and bilinear
upsampling followed by convolution.

### UNet++

Base width 32 with nested dense skips. Four deep-supervision heads contribute
equally during training; the final head is used for inference.

### DRUnet

R2U-Net-style blocks with two recurrent steps and dropout 0.2. Table inference
is deterministic; uncertainty maps are a separately labelled analysis.

### RUnet and RA-RUnet

Table A uses the recurrent backbone without domain alignment. The exemplar
experiment adds a gradient-reversal feature discriminator trained with labelled
latent train subjects and unlabelled exemplar/live-scan train subjects only.

## Transformer substitute

### SegFormer-B2 (substitute†)

The adaptive Gaussian transformer reviewed by Dhaneshwar and Wadhera was built
for volumetric OCT layers and is not directly applicable to this 2-D task.
SegFormer-B2 with ImageNet initialisation and a one-channel adapter is therefore
an explicit substitute, never described as a reproduction.

## Released networks

### SUFS

The released pyfing network is evaluated zero-shot. A separate fine-tuned row
starts from the same released weights and receives the common train/validation
protocol. Its final selected learning rate and threshold are **pending**.

### FingerNet

The official segmentation head and released weights are evaluated zero-shot in
an isolated compatibility environment. Any incompatibility is reported as N/A
rather than replaced by a different network.

