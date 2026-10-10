<!-- GENERATED FILE. Source-to-claim mappings are in traceability.json. -->
# 5. Results

## 5.1 Experimental setting

The evaluation used SD302e latent impressions with examiner annotations from SD302h. The fixed subject-disjoint partition contained 136 training subjects, 24 validation subjects, and 40 test subjects. Table e comprised 2,010 test impressions evaluated at 500 ppi. Model and threshold selection used validation data only, and each frozen model was evaluated on the test set through the locked scoring transaction. Trainable rows reported three seeds; deterministic or pretrained rows reported one run. Metrics were macro-averaged over test images, and the 95% confidence intervals were estimated with 1,000 subject-level cluster-bootstrap resamples.

**Table e. SD302-Latent segmentation against examiner ROI ground truth.**

| Method | Family | Acc (%) | Prec (%) | Recall (%) | Dice (%) | Jaccard (%) | MIoU | MDR (%) | FDR (%) | ms/img | Params (M) | Seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GMFS (default) | Classical | 69.10 | 83.17 | 74.94 | 75.91 | 62.61 | 0.464 | 25.06 | 43.72 | 15.62 | 0.000 | 1 |
| GMFS (val-tuned) | Classical | 71.36 | 79.35 | 84.44 | 79.58 | 67.19 | 0.451 | 15.56 | 60.93 | 15.48 | 0.000 | 1 |
| Support Vector Machine | Classical | 69.12 ± 0.06 | 84.15 ± 0.04 | 72.09 ± 0.07 | 75.85 ± 0.05 | 62.26 ± 0.07 | 0.488 ± 0.001 | 27.91 ± 0.07 | 35.35 ± 0.15 | 41.57 ± 0.32 | 1.009 ± 0.000 | 3 |
| Random decision forest | Classical | 69.82 ± 0.06 | 83.67 ± 0.03 | 73.87 ± 0.09 | 76.82 ± 0.04 | 63.40 ± 0.06 | 0.492 ± 0.000 | 26.13 ± 0.09 | 37.35 ± 0.15 | 58.32 ± 1.03 | 52.697 ± 0.335 | 3 |
| Weighted extreme learning machine | Classical | 68.61 ± 0.14 | 84.18 ± 0.02 | 71.30 ± 0.22 | 75.27 ± 0.19 | 61.57 ± 0.20 | 0.483 ± 0.002 | 28.70 ± 0.22 | 35.10 ± 0.07 | 32.28 ± 0.41 | 0.008 ± 0.002 | 3 |
| LeNet | Patch CNN | 77.46 ± 0.22 | 79.65 ± 0.51 | 93.00 ± 0.83 | 84.93 ± 0.18 | 74.46 ± 0.27 | 0.530 ± 0.006 | 7.00 ± 0.83 | 58.00 ± 2.42 | 16.76 ± 3.40 | 0.061 ± 0.000 | 3 |
| AlexNet | Patch CNN | 80.33 ± 0.04 | 83.45 ± 0.10 | 91.32 ± 0.14 | 86.46 ± 0.02 | 76.62 ± 0.03 | 0.598 ± 0.001 | 8.68 ± 0.14 | 41.73 ± 0.40 | 19.22 ± 3.48 | 2.908 ± 0.000 | 3 |
| CNN + stacked AE | Patch CNN | 75.62 ± 0.21 | 77.54 ± 0.62 | 93.68 ± 0.86 | 83.97 ± 0.07 | 73.08 ± 0.07 | 0.483 ± 0.009 | 6.32 ± 0.86 | 69.41 ± 3.04 | 19.44 ± 3.79 | 0.165 ± 0.000 | 3 |
| U-Net | Encoder-decoder | 74.74 ± 0.57 | 85.11 ± 0.88 | 79.05 ± 1.68 | 79.50 ± 1.18 | 68.21 ± 0.96 | 0.549 ± 0.007 | 20.95 ± 1.68 | 32.78 ± 1.94 | 29.57 ± 2.46 | 7.240 ± 0.000 | 3 |
| UNet++ | Encoder-decoder | 74.49 ± 0.45 | 85.11 ± 3.03 | 78.08 ± 3.24 | 78.22 ± 0.58 | 67.49 ± 0.90 | 0.546 ± 0.016 | 21.92 ± 3.24 | 33.21 ± 9.46 | 42.36 ± 12.72 | 9.159 ± 0.000 | 3 |
| DRUnet | Encoder-decoder | 76.85 ± 1.36 | 78.19 ± 1.92 | 94.83 ± 1.45 | 84.85 ± 0.57 | 74.39 ± 0.83 | 0.508 ± 0.040 | 5.17 ± 1.45 | 64.75 ± 9.92 | 33.40 ± 4.96 | 8.383 ± 0.000 | 3 |
| RUnet (no alignment) | Encoder-decoder | 76.17 ± 1.05 | 76.98 ± 1.55 | 95.75 ± 1.74 | 84.46 ± 0.33 | 73.97 ± 0.53 | 0.486 ± 0.027 | 4.25 ± 1.74 | 70.20 ± 7.11 | 34.03 ± 7.51 | 8.383 ± 0.000 | 3 |
| SegFormer-B2 (substitute†) | Transformer | 78.84 ± 3.72 | 80.13 ± 4.98 | 94.52 ± 2.49 | 85.59 ± 1.96 | 75.93 ± 2.71 | 0.537 ± 0.097 | 5.48 ± 2.49 | 60.77 ± 21.79 | 32.69 ± 3.15 | 27.341 ± 0.000 | 3 |
| SUFS (zero-shot) | Pretrained | 68.62 | 83.53 | 73.00 | 74.95 | 61.70 | 0.471 | 27.00 | 39.13 | 21.22 | 7.084 | 1 |
| SUFS (fine-tuned) | Pretrained | 73.46 ± 1.39 | 75.96 ± 2.50 | 92.29 ± 1.57 | 81.85 ± 0.72 | 70.91 ± 0.76 | 0.447 ± 0.039 | 7.71 ± 1.57 | 74.34 ± 10.31 | 26.17 ± 0.14 | 7.084 ± 0.000 | 3 |
| FingerNet (pretrained) | Pretrained | 60.00 | 92.79 | 50.06 | 58.93 | 46.45 | 0.430 | 49.94 | 11.54 | 35.51 | 4.710 | 1 |
| All background | Reference | 28.16 | — | 0.00 | 0.00 | 0.00 | 0.141 | 100.00 | 0.00 | — | 0.000 | 1 |
| All foreground | Reference | 71.84 | 71.84 | 100.00 | 82.98 | 71.84 | 0.362 | 0.00 | 100.00 | — | 0.000 | 1 |

## 5.2 Latent segmentation against examiner ROI

Across the 16 method rows, AlexNet attained the highest macro accuracy (80.33%; 95% CI 79.95%–80.71%) and MIoU (0.598; 95% CI 0.591–0.604). FingerNet produced the lowest FDR (11.54%; 95% CI 10.27%–13.00%) and the highest precision (92.79%), while its recall was 50.06% and its Dice was 58.93%. Thus, the accuracy, overlap, and false-detection extrema were associated with different operating points.

Within the patch-CNN family, accuracy ranged from 75.62% for CNN + stacked AE to 80.33% for AlexNet, and MIoU ranged from 0.483 to 0.598. The encoder–decoder rows separated into lower-FDR operating points for U-Net and UNet++ (FDR 32.78% and 33.21%) and higher-recall operating points for DRUnet and RUnet (recall 94.83% and 95.75%; FDR 64.75% and 70.20%). SegFormer-B2, the declared 2-D substitute, yielded accuracy 78.84%, Dice 85.59%, and FDR 60.77%.

The SVM, random-forest, and weighted-ELM rows had MDR values between 26.13% and 28.70%, with precision between 83.67% and 84.18%. Validation-tuned GMFS raised recall from 74.94% to 84.44% relative to its default setting, alongside an FDR increase from 43.72% to 60.93%. Fine-tuning SUFS changed recall from 73.00% to 92.29%, precision from 83.53% to 75.96%, and Dice from 74.95% to 81.85%.

The paired Wilcoxon analysis used AlexNet as the Table-e Dice reference. All 15 of 15 Holm-corrected comparisons were significant. The smallest mean paired Dice difference was 0.88 percentage points against SegFormer-B2 (95% bootstrap CI 0.42 percentage points to 1.34 percentage points; Holm-adjusted p=4.66e-04).

## 5.3 Effect of ground-truth definition

**Table e-2. SD302-Latent segmentation against usable-ridge ground truth.**

| Method | Family | Acc (%) | Prec (%) | Recall (%) | Dice (%) | Jaccard (%) | MIoU | MDR (%) | FDR (%) | ms/img | Params (M) | Seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GMFS (default) | Classical | 75.36 | 77.95 | 86.71 | 79.65 | 67.87 | 0.578 | 13.29 | 41.31 | 15.62 | 0.000 | 1 |
| GMFS (val-tuned) | Classical | 71.31 | 70.91 | 93.20 | 78.36 | 66.11 | 0.508 | 6.80 | 58.98 | 15.48 | 0.000 | 1 |
| Support Vector Machine | Classical | 77.96 ± 0.14 | 79.08 ± 0.18 | 86.11 ± 0.07 | 80.97 ± 0.15 | 69.55 ± 0.17 | 0.625 ± 0.001 | 13.89 ± 0.07 | 32.95 ± 0.08 | 41.57 ± 0.32 | 1.009 ± 0.000 | 3 |
| Random decision forest | Classical | 77.76 ± 0.04 | 78.07 ± 0.06 | 87.48 ± 0.06 | 81.09 ± 0.04 | 69.66 ± 0.03 | 0.620 ± 0.001 | 12.52 ± 0.06 | 35.18 ± 0.13 | 58.32 ± 1.03 | 52.697 ± 0.335 | 3 |
| Weighted extreme learning machine | Classical | 77.89 ± 0.05 | 79.49 ± 0.08 | 85.38 ± 0.23 | 80.84 ± 0.08 | 69.40 ± 0.10 | 0.624 ± 0.001 | 14.62 ± 0.23 | 32.50 ± 0.09 | 32.28 ± 0.41 | 0.008 ± 0.002 | 3 |
| LeNet | Patch CNN | 68.33 ± 1.00 | 65.53 ± 0.77 | 96.47 ± 0.22 | 76.28 ± 0.51 | 63.17 ± 0.63 | 0.470 ± 0.014 | 3.53 ± 0.22 | 65.82 ± 2.29 | 16.76 ± 3.40 | 0.061 ± 0.000 | 3 |
| AlexNet | Patch CNN | 71.19 ± 0.28 | 68.38 ± 0.16 | 94.55 ± 0.04 | 77.52 ± 0.16 | 64.63 ± 0.19 | 0.520 ± 0.003 | 5.45 ± 0.04 | 54.47 ± 0.39 | 19.22 ± 3.48 | 2.908 ± 0.000 | 3 |
| CNN + stacked AE | Patch CNN | 65.44 ± 1.13 | 63.38 ± 0.80 | 96.68 ± 0.35 | 74.83 ± 0.57 | 61.26 ± 0.68 | 0.424 ± 0.015 | 3.32 ± 0.35 | 73.72 ± 2.62 | 19.44 ± 3.79 | 0.165 ± 0.000 | 3 |
| U-Net | Encoder-decoder | 68.47 ± 2.38 | 70.56 ± 1.36 | 83.44 ± 3.01 | 72.33 ± 2.67 | 58.90 ± 3.02 | 0.506 ± 0.029 | 16.56 ± 3.01 | 43.42 ± 2.42 | 29.57 ± 2.46 | 7.240 ± 0.000 | 3 |
| UNet++ | Encoder-decoder | 66.11 ± 2.69 | 68.64 ± 4.01 | 80.83 ± 1.76 | 69.45 ± 1.58 | 55.93 ± 1.68 | 0.474 ± 0.040 | 19.17 ± 1.76 | 46.34 ± 9.19 | 42.36 ± 12.72 | 9.159 ± 0.000 | 3 |
| DRUnet | Encoder-decoder | 65.44 ± 2.09 | 63.26 ± 2.00 | 96.76 ± 0.99 | 74.74 ± 0.94 | 61.19 ± 1.26 | 0.426 ± 0.042 | 3.24 ± 0.99 | 73.01 ± 8.46 | 33.40 ± 4.96 | 8.383 ± 0.000 | 3 |
| RUnet (no alignment) | Encoder-decoder | 63.69 ± 1.45 | 61.75 ± 1.40 | 97.08 ± 1.40 | 73.73 ± 0.39 | 59.96 ± 0.47 | 0.395 ± 0.025 | 2.92 ± 1.40 | 78.82 ± 5.82 | 34.03 ± 7.51 | 8.383 ± 0.000 | 3 |
| SegFormer-B2 (substitute†) | Transformer | 66.57 ± 6.27 | 64.44 ± 4.73 | 95.95 ± 1.33 | 74.97 ± 3.17 | 61.73 ± 3.90 | 0.440 ± 0.096 | 4.05 ± 1.33 | 70.12 ± 16.85 | 32.69 ± 3.15 | 27.341 ± 0.000 | 3 |
| SUFS (zero-shot) | Pretrained | 76.13 | 77.89 | 85.08 | 78.98 | 67.20 | 0.588 | 14.92 | 38.09 | 21.22 | 7.084 | 1 |
| SUFS (fine-tuned) | Pretrained | 61.53 ± 3.47 | 61.08 ± 2.64 | 93.87 ± 0.93 | 71.56 ± 1.75 | 57.59 ± 1.93 | 0.371 ± 0.044 | 6.13 ± 0.93 | 79.76 ± 7.85 | 26.17 ± 0.14 | 7.084 ± 0.000 | 3 |
| FingerNet (pretrained) | Pretrained | 70.72 | 87.97 | 60.09 | 64.42 | 52.60 | 0.545 | 39.91 | 13.67 | 35.51 | 4.710 | 1 |
| All background | Reference | 42.97 | — | 0.00 | 0.00 | 0.00 | 0.215 | 100.00 | 0.00 | — | 0.000 | 1 |
| All foreground | Reference | 57.03 | 57.03 | 100.00 | 71.30 | 57.03 | 0.285 | 0.00 | 100.00 | — | 0.000 | 1 |

*The 10,024 valid quality maps describe the complete 10,092-impression universe. Table e-2 contains the 2,009 eligible test impressions from 40 test subjects.*

The examiner ROI occupied a macro-average 71.84% of the test image area, as shown by the all-foreground reference, and therefore the all-foreground Dice reached 82.98%. Under usable-ridge ground truth, the corresponding foreground fraction was 57.03% and all-foreground Dice was 71.30%. This comparison demonstrated that the coarse ROI rewarded broad foreground coverage; it did not imply that recall was uniformly inflated, because the direction and magnitude of metric changes depended on each prediction's spatial extent.

The Dice ordering changed materially. AlexNet moved from rank 1 under examiner ROI to rank 7 under usable-ridge ground truth, whereas the random forest moved from rank 11 to rank 1 and the SVM from rank 13 to rank 2. The highest usable-ridge Dice was 81.09% for the random forest (95% CI 80.22%–81.99%), while the highest usable-ridge MIoU was 0.625 for the SVM (95% CI 0.610–0.640).

The preregistered deep-minus-classical Dice contrast was 6.89 percentage points under ROI ground truth (95% CI 6.05 percentage points to 7.80 percentage points; AlexNet minus validation-tuned GMFS) and -3.58 percentage points under usable-ridge ground truth (95% CI -4.93 percentage points to -2.36 percentage points; AlexNet minus random forest). The sign reversal showed that family-level comparisons depended on the operational definition of foreground.

## 5.4 Effect of examiner value

**Table e-1. SD302-Latent results by examiner value.**

| Method | Examiner value | n | Dice (%) | MIoU | MDR (%) | FDR (%) |
|---|---|---|---|---|---|---|
| GMFS (default) | VALUE | 743 | 75.74 | 0.480 | 28.40 | 35.16 |
| GMFS (default) | LIMITED | 336 | 75.59 | 0.456 | 24.80 | 46.22 |
| GMFS (default) | NOVALUE | 922 | 76.10 | 0.453 | 22.64 | 49.49 |
| GMFS (val-tuned) | VALUE | 743 | 78.86 | 0.466 | 19.66 | 51.70 |
| GMFS (val-tuned) | LIMITED | 336 | 79.31 | 0.442 | 15.26 | 63.42 |
| GMFS (val-tuned) | NOVALUE | 922 | 80.21 | 0.442 | 12.47 | 67.26 |
| Random decision forest | VALUE | 743 | 76.61 ± 0.04 | 0.503 ± 0.000 | 28.68 ± 0.05 | 30.82 ± 0.12 |
| Random decision forest | LIMITED | 336 | 77.02 ± 0.14 | 0.486 ± 0.002 | 24.94 ± 0.20 | 41.49 ± 0.17 |
| Random decision forest | NOVALUE | 922 | 76.86 ± 0.06 | 0.486 ± 0.000 | 24.62 ± 0.18 | 41.03 ± 0.39 |
| Support Vector Machine | VALUE | 743 | 75.80 ± 0.05 | 0.498 ± 0.001 | 30.06 ± 0.03 | 29.36 ± 0.07 |
| Support Vector Machine | LIMITED | 336 | 76.16 ± 0.05 | 0.484 ± 0.001 | 26.57 ± 0.18 | 39.63 ± 0.38 |
| Support Vector Machine | NOVALUE | 922 | 75.73 ± 0.09 | 0.482 ± 0.001 | 26.78 ± 0.10 | 38.52 ± 0.13 |
| Weighted extreme learning machine | VALUE | 743 | 75.17 ± 0.17 | 0.494 ± 0.002 | 31.04 ± 0.22 | 28.78 ± 0.09 |
| Weighted extreme learning machine | LIMITED | 336 | 75.65 ± 0.03 | 0.478 ± 0.001 | 27.19 ± 0.09 | 39.79 ± 0.26 |
| Weighted extreme learning machine | NOVALUE | 922 | 75.15 ± 0.28 | 0.476 ± 0.002 | 27.49 ± 0.34 | 38.37 ± 0.12 |
| DRUnet | VALUE | 743 | 85.67 ± 0.59 | 0.506 ± 0.037 | 3.57 ± 1.09 | 68.23 ± 8.47 |
| DRUnet | LIMITED | 336 | 84.77 ± 0.71 | 0.505 ± 0.046 | 5.01 ± 1.55 | 66.10 ± 10.80 |
| DRUnet | NOVALUE | 922 | 84.20 ± 0.51 | 0.510 ± 0.040 | 6.55 ± 1.70 | 61.34 ± 10.80 |
| RUnet (no alignment) | VALUE | 743 | 85.51 ± 0.46 | 0.490 ± 0.025 | 2.84 ± 0.82 | 72.09 ± 5.56 |
| RUnet (no alignment) | LIMITED | 336 | 84.51 ± 0.34 | 0.478 ± 0.026 | 3.55 ± 1.43 | 73.32 ± 6.68 |
| RUnet (no alignment) | NOVALUE | 922 | 83.58 ± 0.54 | 0.486 ± 0.031 | 5.66 ± 2.66 | 67.47 ± 8.82 |
| U-Net | VALUE | 743 | 84.01 ± 0.57 | 0.576 ± 0.016 | 14.28 ± 0.53 | 37.94 ± 2.08 |
| U-Net | LIMITED | 336 | 80.05 ± 1.24 | 0.552 ± 0.008 | 20.70 ± 1.66 | 32.57 ± 1.73 |
| U-Net | NOVALUE | 922 | 75.69 ± 2.37 | 0.527 ± 0.011 | 26.35 ± 3.39 | 28.69 ± 2.37 |
| UNet++ | VALUE | 743 | 84.71 ± 0.36 | 0.587 ± 0.021 | 12.84 ± 2.93 | 39.42 ± 9.74 |
| UNet++ | LIMITED | 336 | 78.88 ± 0.77 | 0.545 ± 0.018 | 21.66 ± 3.37 | 33.71 ± 10.12 |
| UNet++ | NOVALUE | 922 | 72.75 ± 0.75 | 0.514 ± 0.011 | 29.31 ± 3.51 | 27.99 ± 9.05 |
| AlexNet | VALUE | 743 | 87.20 ± 0.04 | 0.600 ± 0.002 | 7.08 ± 0.14 | 44.54 ± 0.53 |
| AlexNet | LIMITED | 336 | 86.49 ± 0.05 | 0.599 ± 0.001 | 8.51 ± 0.06 | 42.84 ± 0.09 |
| AlexNet | NOVALUE | 922 | 85.86 ± 0.06 | 0.595 ± 0.000 | 10.04 ± 0.22 | 39.03 ± 0.44 |
| CNN + stacked AE | VALUE | 743 | 84.78 ± 0.09 | 0.495 ± 0.012 | 6.00 ± 0.80 | 67.94 ± 3.72 |
| CNN + stacked AE | LIMITED | 336 | 83.97 ± 0.06 | 0.477 ± 0.008 | 5.72 ± 0.73 | 71.62 ± 2.62 |
| CNN + stacked AE | NOVALUE | 922 | 83.29 ± 0.22 | 0.476 ± 0.007 | 6.82 ± 1.02 | 69.82 ± 2.71 |
| LeNet | VALUE | 743 | 85.38 ± 0.16 | 0.529 ± 0.008 | 6.49 ± 0.89 | 59.74 ± 2.86 |
| LeNet | LIMITED | 336 | 84.93 ± 0.15 | 0.526 ± 0.010 | 6.42 ± 0.75 | 60.42 ± 2.85 |
| LeNet | NOVALUE | 922 | 84.55 ± 0.23 | 0.532 ± 0.005 | 7.63 ± 0.81 | 55.68 ± 1.95 |
| FingerNet (pretrained) | VALUE | 743 | 78.67 | 0.560 | 28.29 | 17.60 |
| FingerNet (pretrained) | LIMITED | 336 | 65.09 | 0.448 | 45.47 | 13.18 |
| FingerNet (pretrained) | NOVALUE | 922 | 40.55 | 0.318 | 69.35 | 5.74 |
| SUFS (fine-tuned) | VALUE | 743 | 84.47 ± 0.75 | 0.465 ± 0.045 | 4.15 ± 1.58 | 75.91 ± 10.83 |
| SUFS (fine-tuned) | LIMITED | 336 | 82.36 ± 0.51 | 0.444 ± 0.035 | 6.48 ± 1.54 | 76.08 ± 9.13 |
| SUFS (fine-tuned) | NOVALUE | 922 | 79.56 ± 0.79 | 0.434 ± 0.036 | 11.03 ± 1.55 | 72.36 ± 10.40 |
| SUFS (zero-shot) | VALUE | 743 | 74.96 | 0.481 | 29.62 | 33.00 |
| SUFS (zero-shot) | LIMITED | 336 | 75.05 | 0.463 | 25.89 | 42.83 |
| SUFS (zero-shot) | NOVALUE | 922 | 74.87 | 0.467 | 25.38 | 42.60 |
| All background | VALUE | 743 | 0.00 | 0.137 | 100.00 | 0.00 |
| All background | LIMITED | 336 | 0.00 | 0.142 | 100.00 | 0.00 |
| All background | NOVALUE | 922 | 0.00 | 0.143 | 100.00 | 0.00 |
| All foreground | VALUE | 743 | 83.57 | 0.364 | 0.00 | 100.00 |
| All foreground | LIMITED | 336 | 82.95 | 0.361 | 0.00 | 100.00 |
| All foreground | NOVALUE | 922 | 82.52 | 0.361 | 0.00 | 100.00 |
| SegFormer-B2 (substitute†) | VALUE | 743 | 87.19 ± 1.88 | 0.557 ± 0.109 | 4.01 ± 2.97 | 59.86 ± 24.00 |
| SegFormer-B2 (substitute†) | LIMITED | 336 | 85.51 ± 2.09 | 0.531 ± 0.100 | 4.96 ± 2.38 | 62.25 ± 21.46 |
| SegFormer-B2 (substitute†) | NOVALUE | 922 | 84.31 ± 1.97 | 0.522 ± 0.086 | 6.85 ± 2.15 | 61.02 ± 20.11 |

*The stratified table contains VALUE=743, LIMITED=336, and NOVALUE=922 impressions. The nine MIXED impressions remain in overall Table e but are excluded from e-1.*

The VALUE, LIMITED, and NOVALUE strata contained 743, 336, and 922 impressions, respectively; nine MIXED impressions were retained in overall Table e but excluded from e-1. From VALUE to NOVALUE, Dice changed by -1.34 percentage points for AlexNet, -8.32 percentage points for U-Net, -11.96 percentage points for UNet++, and -38.12 percentage points for FingerNet. These differences were method-specific rather than a uniform offset across examiner categories.

## 5.5 Plain and rolled live-scan impressions

The verified SD302g-to-SD302b mapping contained 2,194 baseline records across the fixed split. The test subset contained 439 images: 430 rolled impressions from 40 subjects and nine individual images segmented from slap captures from five subjects. SD302b documents operator-assisted rolls and 4-4-2 slaps; consequently, the latter group is labelled ‘Plain (segmented slap)’ and is not presented as a separately documented plain-impression acquisition. The canonical 500-ppi SD302b inputs were exhaustively pixel-equivalent to the corresponding embedded SD302g inputs. No inference was rerun: authenticated locked B1 predictions were subset and aggregated as behaviour-equivalent evidence. For native 1000-ppi records, equivalence after Pillow LANCZOS resizing was an exhaustive empirical observation rather than an authenticated archive-creation recipe.

**Table b-plain. SD302b Plain (segmented slap), expert-minutiae-hull evaluation.**

| Method | Family | Training | MC (%) [95% CI] | HR (%) [95% CI] | PAR [95% CI] | OHF (%) [95% CI] | Pred. area (%) [95% CI] | ms/img† | Params (M) | Seeds | Status/reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GMFS (default) | Classical | none | 98.26 [94.79, 100.00] | 99.69 [99.07, 100.00] | 1.540 [1.412, 1.708] | 34.00 [29.39, 39.15] | 68.08 [64.32, 74.26] | 15.62 | 0.000 | 1 |  |
| GMFS (val-tuned) | Classical | SD302-Latent train/val split | 100.00 [100.00, 100.00] | 99.99 [99.98, 100.00] | 1.590 [1.490, 1.739] | 35.97 [32.59, 40.32] | 70.29 [66.23, 75.79] | 15.48 | 0.000 | 1 |  |
| SUFS (zero-shot) | Pretrained | external live-scan data | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 1.639 [1.559, 1.761] | 37.97 [35.52, 41.12] | 72.47 [68.75, 77.73] | 21.22 | 7.084 | 1 |  |
| SUFS (fine-tuned) | Pretrained | SD302-Latent train split | 94.91 ± 3.27 [91.10, 99.02] | 95.25 ± 4.33 [91.70, 98.71] | 2.146 ± 0.108 [1.989, 2.334] | 54.59 ± 0.36 [50.67, 56.87] | 94.33 ± 4.81 [90.82, 97.27] | 26.17 ± 0.14 | 7.084 ± 0.000 | 3 |  |
| FingerNet (pretrained) | Pretrained | external live-scan data | 100.00 [100.00, 100.00] | 99.99 [99.97, 100.00] | 1.968 [1.861, 2.105] | 48.26 [45.78, 50.66] | 86.76 [83.14, 90.96] | 35.51 | 4.710 | 1 |  |
| SVM (RBF via Nystrom) | Classical | SD302-Latent train split | 12.94 ± 1.56 [0.13, 37.88] | 12.47 ± 1.14 [0.04, 37.15] | 0.195 ± 0.025 [0.003, 0.567] | 13.49 ± 10.76 [3.21, 26.72] | 8.94 ± 1.17 [0.12, 26.10] | 41.57 ± 0.32 | 1.009 ± 0.000 | 3 |  |
| Random forest | Classical | SD302-Latent train split | 92.22 ± 1.09 [84.97, 96.83] | 97.19 ± 0.67 [93.11, 99.53] | 1.385 ± 0.029 [1.276, 1.487] | 28.81 ± 1.37 [26.23, 31.71] | 61.35 ± 1.25 [59.09, 65.63] | 58.32 ± 1.03 | 52.697 ± 0.335 | 3 |  |
| Weighted ELM | Classical | SD302-Latent train split | 26.18 ± 7.34 [4.27, 56.34] | 27.16 ± 9.76 [4.99, 56.37] | 0.438 ± 0.133 [0.078, 0.916] | 18.33 ± 9.25 [8.07, 29.84] | 20.01 ± 5.92 [3.46, 42.57] | 32.28 ± 0.41 | 0.008 ± 0.002 | 3 |  |
| U-Net | Encoder-decoder | SD302-Latent train split | 91.18 ± 8.24 [88.03, 93.56] | 94.72 ± 6.60 [92.16, 96.35] | 1.751 ± 0.306 [1.601, 1.901] | 43.42 ± 9.09 [40.48, 46.95] | 76.80 ± 13.33 [74.76, 78.72] | 29.57 ± 2.46 | 7.240 ± 0.000 | 3 |  |
| UNet++ | Encoder-decoder | SD302-Latent train split | 94.93 ± 1.84 [93.94, 96.23] | 96.12 ± 1.45 [93.80, 97.91] | 1.891 ± 0.227 [1.762, 2.013] | 47.28 ± 5.79 [43.97, 50.55] | 83.18 ± 9.63 [80.50, 87.28] | 42.36 ± 12.72 | 9.159 ± 0.000 | 3 |  |
| DRUnet | Encoder-decoder | SD302-Latent train split | 96.50 ± 5.42 [93.26, 99.34] | 98.89 ± 1.90 [97.29, 99.90] | 1.842 ± 0.447 [1.680, 2.020] | 42.71 ± 14.29 [38.96, 46.63] | 81.05 ± 19.37 [78.41, 84.92] | 33.40 ± 4.96 | 8.383 ± 0.000 | 3 |  |
| RUnet (no alignment) | Encoder-decoder | SD302-Latent train split | 100.00 ± 0.00 [100.00, 100.00] | 99.82 ± 0.31 [99.68, 99.91] | 2.149 ± 0.085 [1.969, 2.320] | 52.35 ± 1.95 [48.88, 54.90] | 94.32 ± 3.69 [92.51, 96.25] | 34.03 ± 7.51 | 8.383 ± 0.000 | 3 |  |
| LeNet | Patch CNN | SD302-Latent train split | 100.00 ± 0.00 [100.00, 100.00] | 100.00 ± 0.00 [100.00, 100.00] | 1.897 ± 0.018 [1.765, 2.098] | 45.83 ± 0.47 [43.05, 49.34] | 83.25 ± 0.76 [79.46, 87.79] | 16.76 ± 3.40 | 0.061 ± 0.000 | 3 |  |
| AlexNet | Patch CNN | SD302-Latent train split | 99.48 ± 0.22 [98.70, 99.93] | 99.98 ± 0.00 [99.92, 100.00] | 2.180 ± 0.005 [1.987, 2.346] | 52.88 ± 0.09 [49.14, 55.22] | 95.55 ± 0.19 [94.75, 96.74] | 19.22 ± 3.48 | 2.908 ± 0.000 | 3 |  |
| CNN + stacked AE | Patch CNN | SD302-Latent train split | 99.07 ± 1.22 [97.29, 100.00] | 99.07 ± 1.50 [98.14, 100.00] | 2.215 ± 0.016 [2.004, 2.373] | 54.11 ± 0.37 [49.95, 56.18] | 97.16 ± 0.72 [95.83, 98.77] | 19.44 ± 3.79 | 0.165 ± 0.000 | 3 |  |
| SegFormer-B2 (substitute) | Transformer | SD302-Latent train split | 100.00 ± 0.00 [100.00, 100.00] | 100.00 ± 0.00 [100.00, 100.00] | 2.144 ± 0.120 [1.951, 2.322] | 52.05 ± 2.63 [48.19, 54.70] | 94.08 ± 5.21 [92.97, 95.83] | 32.69 ± 3.15 | 27.341 ± 0.000 | 3 |  |
| RA-RUnet (latent-to-live-scan alignment) | Encoder-decoder | labelled SD302-Latent train plus unlabelled live-scan train subjects | pending | pending | pending | pending | pending | pending | pending | 0 | Pending: no authenticated aligned RA-RUnet checkpoint was completed before the benchmark freeze; never substitute the scored no-alignment RUnet. |
| All-background | Reference | deterministic reference | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | — | 0.000 | 1 |  |
| All-foreground | Reference | deterministic reference | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 2.280 [2.056, 2.437] | 55.00 [50.81, 57.25] | 100.00 [100.00, 100.00] | — | 0.000 | 1 |  |
| Hull-as-prediction | Reference | deterministic reference | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 1.000 [1.000, 1.000] | 0.00 [0.00, 0.00] | 45.00 [42.75, 49.19] | — | 0.000 | 1 |  |
| Hull + 10 px dilation (500 ppi) | Reference | deterministic reference | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 1.187 [1.176, 1.196] | 15.74 [14.97, 16.29] | 53.26 [50.74, 57.76] | — | 0.000 | 1 |  |

**Table b-rolled. SD302b rolled impressions, expert-minutiae-hull evaluation.**

| Method | Family | Training | MC (%) [95% CI] | HR (%) [95% CI] | PAR [95% CI] | OHF (%) [95% CI] | Pred. area (%) [95% CI] | ms/img† | Params (M) | Seeds | Status/reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GMFS (default) | Classical | none | 99.69 [99.55, 99.79] | 99.54 [99.42, 99.65] | 1.463 [1.382, 1.561] | 28.17 [25.65, 30.79] | 35.84 [34.08, 37.72] | 15.62 | 0.000 | 1 |  |
| GMFS (val-tuned) | Classical | SD302-Latent train/val split | 99.90 [99.85, 99.94] | 99.76 [99.69, 99.84] | 1.490 [1.407, 1.592] | 29.24 [26.65, 31.92] | 36.50 [34.66, 38.43] | 15.48 | 0.000 | 1 |  |
| SUFS (zero-shot) | Pretrained | external live-scan data | 99.90 [99.77, 99.97] | 99.81 [99.73, 99.88] | 1.518 [1.434, 1.619] | 30.53 [27.88, 33.22] | 37.24 [35.36, 39.23] | 21.22 | 7.084 | 1 |  |
| SUFS (fine-tuned) | Pretrained | SD302-Latent train split | 95.68 ± 1.97 [95.08, 96.24] | 95.96 ± 1.66 [95.32, 96.53] | 3.939 ± 0.420 [3.621, 4.268] | 71.88 ± 3.01 [70.11, 73.61] | 89.25 ± 9.26 [88.76, 89.72] | 26.17 ± 0.14 | 7.084 ± 0.000 | 3 |  |
| FingerNet (pretrained) | Pretrained | external live-scan data | 99.78 [99.63, 99.89] | 99.95 [99.88, 99.99] | 1.803 [1.717, 1.907] | 41.73 [39.76, 43.74] | 44.20 [42.13, 46.41] | 35.51 | 4.710 | 1 |  |
| SVM (RBF via Nystrom) | Classical | SD302-Latent train split | 4.16 ± 3.26 [2.79, 5.78] | 4.38 ± 3.37 [2.92, 6.09] | 0.114 ± 0.071 [0.079, 0.157] | 35.58 ± 5.20 [29.88, 41.53] | 2.29 ± 1.40 [1.73, 2.94] | 41.57 ± 0.32 | 1.009 ± 0.000 | 3 |  |
| Random forest | Classical | SD302-Latent train split | 99.07 ± 0.42 [98.83, 99.30] | 99.15 ± 0.43 [98.93, 99.36] | 1.480 ± 0.019 [1.394, 1.581] | 29.09 ± 0.62 [26.52, 31.61] | 35.88 ± 0.52 [34.40, 37.43] | 58.32 ± 1.03 | 52.697 ± 0.335 | 3 |  |
| Weighted ELM | Classical | SD302-Latent train split | 24.77 ± 7.37 [20.08, 29.71] | 25.89 ± 7.83 [21.23, 30.81] | 0.483 ± 0.092 [0.392, 0.586] | 43.70 ± 3.58 [39.12, 48.09] | 10.27 ± 2.25 [8.83, 11.81] | 32.28 ± 0.41 | 0.008 ± 0.002 | 3 |  |
| U-Net | Encoder-decoder | SD302-Latent train split | 91.56 ± 3.73 [90.38, 92.69] | 93.01 ± 3.36 [92.20, 93.80] | 3.476 ± 0.494 [3.202, 3.761] | 69.19 ± 3.32 [67.28, 71.03] | 79.08 ± 10.52 [78.82, 79.35] | 29.57 ± 2.46 | 7.240 ± 0.000 | 3 |  |
| UNet++ | Encoder-decoder | SD302-Latent train split | 91.53 ± 4.42 [90.63, 92.43] | 92.70 ± 3.85 [92.01, 93.38] | 2.645 ± 0.740 [2.470, 2.839] | 60.21 ± 7.63 [58.41, 62.01] | 61.84 ± 15.38 [60.66, 63.02] | 42.36 ± 12.72 | 9.159 ± 0.000 | 3 |  |
| DRUnet | Encoder-decoder | SD302-Latent train split | 98.56 ± 2.23 [98.39, 98.71] | 99.26 ± 1.19 [99.16, 99.35] | 2.532 ± 1.175 [2.354, 2.722] | 49.98 ± 23.55 [48.02, 51.98] | 58.91 ± 25.48 [57.90, 59.93] | 33.40 ± 4.96 | 8.383 ± 0.000 | 3 |  |
| RUnet (no alignment) | Encoder-decoder | SD302-Latent train split | 98.74 ± 1.75 [98.49, 98.97] | 99.03 ± 1.39 [98.87, 99.18] | 3.206 ± 1.291 [2.965, 3.466] | 60.50 ± 18.58 [58.71, 62.26] | 73.61 ± 27.47 [73.01, 74.30] | 34.03 ± 7.51 | 8.383 ± 0.000 | 3 |  |
| LeNet | Patch CNN | SD302-Latent train split | 99.95 ± 0.03 [99.93, 99.97] | 99.95 ± 0.05 [99.91, 99.98] | 1.689 ± 0.016 [1.584, 1.820] | 36.92 ± 0.56 [34.42, 39.56] | 41.10 ± 0.35 [39.01, 43.28] | 16.76 ± 3.40 | 0.061 ± 0.000 | 3 |  |
| AlexNet | Patch CNN | SD302-Latent train split | 99.30 ± 0.23 [99.06, 99.50] | 99.67 ± 0.14 [99.56, 99.76] | 4.374 ± 0.014 [4.030, 4.737] | 73.83 ± 0.05 [72.17, 75.48] | 99.06 ± 0.31 [98.97, 99.14] | 19.22 ± 3.48 | 2.908 ± 0.000 | 3 |  |
| CNN + stacked AE | Patch CNN | SD302-Latent train split | 99.04 ± 0.23 [98.76, 99.28] | 99.40 ± 0.23 [99.23, 99.55] | 4.368 ± 0.015 [4.028, 4.726] | 73.88 ± 0.03 [72.22, 75.50] | 98.98 ± 0.31 [98.85, 99.11] | 19.44 ± 3.79 | 0.165 ± 0.000 | 3 |  |
| SegFormer-B2 (substitute) | Transformer | SD302-Latent train split | 99.80 ± 0.32 [99.63, 99.92] | 99.83 ± 0.26 [99.68, 99.94] | 3.986 ± 0.226 [3.662, 4.321] | 71.36 ± 1.53 [69.60, 73.08] | 90.60 ± 5.12 [90.10, 91.10] | 32.69 ± 3.15 | 27.341 ± 0.000 | 3 |  |
| RA-RUnet (latent-to-live-scan alignment) | Encoder-decoder | labelled SD302-Latent train plus unlabelled live-scan train subjects | pending | pending | pending | pending | pending | pending | pending | 0 | Pending: no authenticated aligned RA-RUnet checkpoint was completed before the benchmark freeze; never substitute the scored no-alignment RUnet. |
| All-background | Reference | deterministic reference | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | — | 0.000 | 1 |  |
| All-foreground | Reference | deterministic reference | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 4.414 [4.068, 4.781] | 73.99 [72.35, 75.60] | 100.00 [100.00, 100.00] | — | 0.000 | 1 |  |
| Hull-as-prediction | Reference | deterministic reference | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 1.000 [1.000, 1.000] | 0.00 [0.00, 0.00] | 26.01 [24.40, 27.65] | — | 0.000 | 1 |  |
| Hull + 10 px dilation (500 ppi) | Reference | deterministic reference | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 1.103 [1.099, 1.108] | 9.32 [8.98, 9.66] | 28.51 [26.82, 30.23] | — | 0.000 | 1 |  |

*Hull ground truth is a lower bound of the true foreground; pixel Dice is not reported. Values are mean ± seed SD [subject-bootstrap 95% CI]. †Timing was hash-pinned from locked Table e and was not remeasured on live-scan images.*

**Table b-device. Hull metrics by capture and device.**

| Method | Capture | Device | n | MC (%) | HR (%) | PAR | OHF (%) | Seeds |
|---|---|---|---|---|---|---|---|---|
| GMFS (default) | Plain (segmented slap) | R | 6 | 100.00 | 100.00 | 1.618 | 36.93 | 1 |
| GMFS (val-tuned) | Plain (segmented slap) | R | 6 | 100.00 | 100.00 | 1.649 | 38.14 | 1 |
| SUFS (zero-shot) | Plain (segmented slap) | R | 6 | 100.00 | 100.00 | 1.671 | 39.01 | 1 |
| SUFS (fine-tuned) | Plain (segmented slap) | R | 6 | 92.61 ± 4.58 | 93.25 ± 6.03 | 2.160 ± 0.132 | 55.67 ± 0.21 | 3 |
| FingerNet (pretrained) | Plain (segmented slap) | R | 6 | 100.00 | 99.98 | 1.971 | 48.18 | 1 |
| SVM (RBF via Nystrom) | Plain (segmented slap) | R | 6 | 0.17 ± 0.30 | 0.09 ± 0.15 | 0.005 ± 0.009 | 4.64 ± 8.04 | 3 |
| Random forest | Plain (segmented slap) | R | 6 | 96.77 ± 0.75 | 99.19 ± 0.29 | 1.451 ± 0.030 | 30.40 ± 1.45 | 3 |
| Weighted ELM | Plain (segmented slap) | R | 6 | 6.41 ± 7.45 | 8.31 ± 7.82 | 0.133 ± 0.131 | 10.38 ± 12.54 | 3 |
| U-Net | Plain (segmented slap) | R | 6 | 90.98 ± 9.45 | 94.41 ± 7.54 | 1.784 ± 0.315 | 44.35 ± 9.48 | 3 |
| UNet++ | Plain (segmented slap) | R | 6 | 94.14 ± 1.99 | 94.99 ± 3.16 | 1.884 ± 0.279 | 47.14 ± 6.96 | 3 |
| DRUnet | Plain (segmented slap) | R | 6 | 98.40 ± 2.78 | 99.75 ± 0.43 | 1.919 ± 0.422 | 44.72 ± 12.98 | 3 |
| RUnet (no alignment) | Plain (segmented slap) | R | 6 | 100.00 ± 0.00 | 99.78 ± 0.39 | 2.197 ± 0.083 | 53.24 ± 1.85 | 3 |
| LeNet | Plain (segmented slap) | R | 6 | 100.00 ± 0.00 | 100.00 ± 0.00 | 1.956 ± 0.022 | 47.08 ± 0.56 | 3 |
| AlexNet | Plain (segmented slap) | R | 6 | 99.66 ± 0.33 | 100.00 ± 0.00 | 2.224 ± 0.007 | 53.62 ± 0.17 | 3 |
| CNN + stacked AE | Plain (segmented slap) | R | 6 | 98.60 ± 1.82 | 98.61 ± 2.25 | 2.245 ± 0.027 | 54.75 ± 0.52 | 3 |
| SegFormer-B2 (substitute) | Plain (segmented slap) | R | 6 | 100.00 ± 0.00 | 100.00 ± 0.00 | 2.199 ± 0.119 | 53.09 ± 2.48 | 3 |
| RA-RUnet (latent-to-live-scan alignment) | Plain (segmented slap) | R | 6 | pending | pending | pending | pending | 0 |
| All-background | Plain (segmented slap) | R | 6 | 0.00 | 0.00 | 0.000 | 0.00 | 1 |
| All-foreground | Plain (segmented slap) | R | 6 | 100.00 | 100.00 | 2.333 | 55.87 | 1 |
| Hull-as-prediction | Plain (segmented slap) | R | 6 | 100.00 | 100.00 | 1.000 | 0.00 | 1 |
| Hull + 10 px dilation (500 ppi) | Plain (segmented slap) | R | 6 | 100.00 | 100.00 | 1.185 | 15.55 | 1 |
| GMFS (default) | Plain (segmented slap) | S | 3 | 94.79 | 99.07 | 1.384 | 28.14 | 1 |
| GMFS (val-tuned) | Plain (segmented slap) | S | 3 | 100.00 | 99.98 | 1.471 | 31.63 | 1 |
| SUFS (zero-shot) | Plain (segmented slap) | S | 3 | 100.00 | 100.00 | 1.575 | 35.87 | 1 |
| SUFS (fine-tuned) | Plain (segmented slap) | S | 3 | 99.50 ± 0.86 | 99.26 ± 0.98 | 2.120 ± 0.065 | 52.41 ± 0.97 | 3 |
| FingerNet (pretrained) | Plain (segmented slap) | S | 3 | 100.00 | 100.00 | 1.963 | 48.41 | 1 |
| SVM (RBF via Nystrom) | Plain (segmented slap) | S | 3 | 38.47 ± 4.12 | 37.23 ± 3.13 | 0.573 ± 0.059 | 31.18 ± 16.21 | 3 |
| Random forest | Plain (segmented slap) | S | 3 | 83.12 ± 2.35 | 93.20 ± 2.54 | 1.254 ± 0.040 | 25.64 ± 1.23 | 3 |
| Weighted ELM | Plain (segmented slap) | S | 3 | 65.73 ± 17.93 | 64.85 ± 21.62 | 1.047 ± 0.278 | 34.23 ± 7.83 | 3 |
| U-Net | Plain (segmented slap) | S | 3 | 91.57 ± 6.93 | 95.33 ± 4.92 | 1.685 ± 0.287 | 41.58 ± 8.38 | 3 |
| UNet++ | Plain (segmented slap) | S | 3 | 96.50 ± 5.32 | 98.37 ± 2.52 | 1.903 ± 0.171 | 47.57 ± 3.78 | 3 |
| DRUnet | Plain (segmented slap) | S | 3 | 92.70 ± 10.71 | 97.16 ± 4.85 | 1.690 ± 0.499 | 38.70 ± 16.89 | 3 |
| RUnet (no alignment) | Plain (segmented slap) | S | 3 | 100.00 ± 0.00 | 99.91 ± 0.16 | 2.052 ± 0.091 | 50.57 ± 2.16 | 3 |
| LeNet | Plain (segmented slap) | S | 3 | 100.00 ± 0.00 | 100.00 ± 0.00 | 1.780 ± 0.011 | 43.31 ± 0.28 | 3 |
| AlexNet | Plain (segmented slap) | S | 3 | 99.12 ± 0.00 | 99.94 ± 0.00 | 2.092 ± 0.006 | 51.41 ± 0.17 | 3 |
| CNN + stacked AE | Plain (segmented slap) | S | 3 | 100.00 ± 0.00 | 100.00 ± 0.00 | 2.157 ± 0.007 | 52.85 ± 0.12 | 3 |
| SegFormer-B2 (substitute) | Plain (segmented slap) | S | 3 | 100.00 ± 0.00 | 100.00 ± 0.00 | 2.035 ± 0.125 | 49.98 ± 2.95 | 3 |
| RA-RUnet (latent-to-live-scan alignment) | Plain (segmented slap) | S | 3 | pending | pending | pending | pending | 0 |
| All-background | Plain (segmented slap) | S | 3 | 0.00 | 0.00 | 0.000 | 0.00 | 1 |
| All-foreground | Plain (segmented slap) | S | 3 | 100.00 | 100.00 | 2.176 | 53.25 | 1 |
| Hull-as-prediction | Plain (segmented slap) | S | 3 | 100.00 | 100.00 | 1.000 | 0.00 | 1 |
| Hull + 10 px dilation (500 ppi) | Plain (segmented slap) | S | 3 | 100.00 | 100.00 | 1.193 | 16.13 | 1 |
| GMFS (default) | Rolled | U | 30 | 99.48 | 99.33 | 1.562 | 33.75 | 1 |
| GMFS (val-tuned) | Rolled | U | 30 | 99.88 | 99.69 | 1.592 | 34.84 | 1 |
| SUFS (zero-shot) | Rolled | U | 30 | 99.88 | 99.83 | 1.626 | 36.17 | 1 |
| SUFS (fine-tuned) | Rolled | U | 30 | 96.66 ± 1.90 | 96.73 ± 1.59 | 3.845 ± 0.443 | 71.52 ± 2.97 | 3 |
| FingerNet (pretrained) | Rolled | U | 30 | 99.79 | 99.83 | 1.897 | 45.42 | 1 |
| SVM (RBF via Nystrom) | Rolled | U | 30 | 5.97 ± 4.24 | 6.09 ± 4.34 | 0.177 ± 0.105 | 44.27 ± 8.45 | 3 |
| Random forest | Rolled | U | 30 | 98.46 ± 0.73 | 98.37 ± 0.79 | 1.561 ± 0.026 | 34.07 ± 0.78 | 3 |
| Weighted ELM | Rolled | U | 30 | 27.48 ± 7.76 | 28.99 ± 7.45 | 0.592 ± 0.104 | 46.89 ± 1.51 | 3 |
| U-Net | Rolled | U | 30 | 95.52 ± 2.26 | 95.86 ± 2.04 | 3.418 ± 0.469 | 68.39 ± 3.64 | 3 |
| UNet++ | Rolled | U | 30 | 94.79 ± 2.65 | 94.81 ± 2.72 | 2.675 ± 0.640 | 60.77 ± 6.61 | 3 |
| DRUnet | Rolled | U | 30 | 98.62 ± 2.33 | 99.23 ± 1.29 | 2.536 ± 1.092 | 52.02 ± 20.51 | 3 |
| RUnet (no alignment) | Rolled | U | 30 | 99.49 ± 0.77 | 99.44 ± 0.90 | 3.182 ± 1.202 | 61.53 ± 16.43 | 3 |
| LeNet | Rolled | U | 30 | 99.97 ± 0.05 | 99.89 ± 0.17 | 1.790 ± 0.014 | 41.82 ± 0.49 | 3 |
| AlexNet | Rolled | U | 30 | 99.78 ± 0.07 | 99.93 ± 0.01 | 4.286 ± 0.012 | 73.71 ± 0.08 | 3 |
| CNN + stacked AE | Rolled | U | 30 | 99.49 ± 0.41 | 99.68 ± 0.33 | 4.280 ± 0.023 | 73.74 ± 0.07 | 3 |
| SegFormer-B2 (substitute) | Rolled | U | 30 | 99.90 ± 0.17 | 99.91 ± 0.15 | 3.939 ± 0.214 | 71.41 ± 1.51 | 3 |
| RA-RUnet (latent-to-live-scan alignment) | Rolled | U | 30 | pending | pending | pending | pending | 0 |
| All-background | Rolled | U | 30 | 0.00 | 0.00 | 0.000 | 0.00 | 1 |
| All-foreground | Rolled | U | 30 | 100.00 | 100.00 | 4.332 | 73.99 | 1 |
| Hull-as-prediction | Rolled | U | 30 | 100.00 | 100.00 | 1.000 | 0.00 | 1 |
| Hull + 10 px dilation (500 ppi) | Rolled | U | 30 | 100.00 | 100.00 | 1.103 | 9.31 | 1 |
| GMFS (default) | Rolled | V | 400 | 99.70 | 99.55 | 1.455 | 27.75 | 1 |
| GMFS (val-tuned) | Rolled | V | 400 | 99.90 | 99.77 | 1.482 | 28.83 | 1 |
| SUFS (zero-shot) | Rolled | V | 400 | 99.90 | 99.81 | 1.510 | 30.11 | 1 |
| SUFS (fine-tuned) | Rolled | V | 400 | 95.61 ± 2.01 | 95.90 ± 1.70 | 3.946 ± 0.419 | 71.91 ± 3.02 | 3 |
| FingerNet (pretrained) | Rolled | V | 400 | 99.78 | 99.96 | 1.796 | 41.45 | 1 |
| SVM (RBF via Nystrom) | Rolled | V | 400 | 4.02 ± 3.19 | 4.25 ± 3.30 | 0.109 ± 0.068 | 34.93 ± 4.98 | 3 |
| Random forest | Rolled | V | 400 | 99.12 ± 0.40 | 99.21 ± 0.40 | 1.474 ± 0.018 | 28.72 ± 0.61 | 3 |
| Weighted ELM | Rolled | V | 400 | 24.57 ± 7.41 | 25.66 ± 7.91 | 0.475 ± 0.092 | 43.46 ± 3.75 | 3 |
| U-Net | Rolled | V | 400 | 91.26 ± 3.84 | 92.79 ± 3.46 | 3.481 ± 0.496 | 69.25 ± 3.29 | 3 |
| UNet++ | Rolled | V | 400 | 91.28 ± 4.56 | 92.54 ± 3.93 | 2.643 ± 0.748 | 60.17 ± 7.71 | 3 |
| DRUnet | Rolled | V | 400 | 98.55 ± 2.22 | 99.26 ± 1.18 | 2.532 ± 1.181 | 49.83 ± 23.78 | 3 |
| RUnet (no alignment) | Rolled | V | 400 | 98.69 ± 1.82 | 99.00 ± 1.43 | 3.208 ± 1.298 | 60.43 ± 18.74 | 3 |
| LeNet | Rolled | V | 400 | 99.95 ± 0.03 | 99.95 ± 0.04 | 1.682 ± 0.016 | 36.56 ± 0.56 | 3 |
| AlexNet | Rolled | V | 400 | 99.27 ± 0.24 | 99.65 ± 0.15 | 4.380 ± 0.014 | 73.84 ± 0.05 | 3 |
| CNN + stacked AE | Rolled | V | 400 | 99.00 ± 0.24 | 99.38 ± 0.23 | 4.374 ± 0.015 | 73.89 ± 0.02 | 3 |
| SegFormer-B2 (substitute) | Rolled | V | 400 | 99.79 ± 0.33 | 99.82 ± 0.27 | 3.989 ± 0.227 | 71.35 ± 1.54 | 3 |
| RA-RUnet (latent-to-live-scan alignment) | Rolled | V | 400 | pending | pending | pending | pending | 0 |
| All-background | Rolled | V | 400 | 0.00 | 0.00 | 0.000 | 0.00 | 1 |
| All-foreground | Rolled | V | 400 | 100.00 | 100.00 | 4.420 | 74.00 | 1 |
| Hull-as-prediction | Rolled | V | 400 | 100.00 | 100.00 | 1.000 | 0.00 | 1 |
| Hull + 10 px dilation (500 ppi) | Rolled | V | 400 | 100.00 | 100.00 | 1.103 | 9.32 | 1 |

On the nine segmented-slap images, GMFS default, SUFS zero-shot, and FingerNet produced HR values of 99.69%, 100.00%, and 99.99%, with PAR values of 1.540, 1.639, and 1.968. On the 430 rolled images, the corresponding HR values were 99.54%, 99.81%, and 99.95%, with PAR values of 1.463, 1.518, and 1.803. The 10-pixel hull-dilation reference had PAR 1.187 for segmented slaps and 1.103 for rolls. These values showed why hull coverage was interpreted jointly with predicted area rather than as an accuracy measure on its own.

Among two representative latent-trained models, U-Net changed from HR 94.72% and PAR 1.751 on segmented slaps to HR 93.01% and PAR 3.476 on rolls; AlexNet changed from HR 99.98% and PAR 2.180 to HR 99.67% and PAR 4.374. All non-adapted trainable rows had been trained on SD302-Latent, so Table b measured latent-to-live-scan generalisation. RA-RUnet alignment remained pending because no authenticated aligned checkpoint existed; the scored RUnet row was the no-alignment model and was not used as a substitute.

The image-macro inter-method Dice was 0.7263 on segmented slaps (95% CI 0.6880–0.7965) and 0.6018 on rolls (95% CI 0.5928–0.6102). This was an agreement statistic, not an accuracy estimate. The segmented-slap interval was based on only nine images from five subjects. Device and capture type were confounded in the mapped subset: R/S occurred only for segmented slaps, whereas U/V occurred only for rolls, so device differences were not interpreted as isolated sensor effects.

Human pixel masks were not available; the 60-image plus 20-repeat manual-annotation package was ready in the restricted project deliverables. Until those annotations are completed, the tables support hull coverage, area, and agreement claims only, not pixel Dice or pixel false-detection claims.

*[Figure F4 insertion point: plain-versus-rolled HR and PAR, with the 10-pixel hull-dilation reference.]*

## 5.6 Efficiency

Using accuracy as the quality coordinate and measured milliseconds per image as the cost coordinate, the non-dominated frontier comprised GMFS (val-tuned) (71.36%, 15.48 ms/image), LeNet (77.46%, 16.76 ms/image), AlexNet (80.33%, 19.22 ms/image). Parameter count alone did not determine runtime: for example, the weighted ELM used 0.008 M parameters and required 32.28 ms/image, while AlexNet used 2.908 M parameters and required 19.22 ms/image. These timings were implementation- and device-specific and were interpreted as within-benchmark measurements.

*[Figure F5 insertion point: accuracy versus milliseconds per image, with the frontier identified above.]*

## 5.7 Comparison with originally reported results

**Table 6. Originally reported values and current SD302-Latent values.**

| Fig. 2 entry | Original source | Originally reported result | Current SD302-Latent result | Comparability note |
|---|---|---|---|---|
| CNN & autoencoders | Chhabra et al., 2023 | 98.45% | CNN + stacked AE: Acc 75.62%, Dice 83.97%, Jaccard 73.08%, MIoU 0.483, MDR 6.32%, FDR 69.41% | Patch/hybrid result, not full-image SD302 pixel segmentation; do not reinterpret this value as Dice or Jaccard. |
| GMFS & SUFS | Cappelli, 2023 | GMFS default: ER 2.98%, Dice 97.48%, Jaccard 95.19%; SUFS zero-shot: ER 1.51%, Dice 98.69%, Jaccard 97.43%; GMFS val-tuned: not reported; SUFS fine-tuned: not reported | GMFS (default): Acc 69.10%, Dice 75.91%, Jaccard 62.61%, MIoU 0.464, MDR 25.06%, FDR 43.72%; GMFS (val-tuned): Acc 71.36%, Dice 79.58%, Jaccard 67.19%, MIoU 0.451, MDR 15.56%, FDR 60.93%; SUFS (zero-shot): Acc 68.62%, Dice 74.95%, Jaccard 61.70%, MIoU 0.471, MDR 27.00%, FDR 39.13%; SUFS (fine-tuned): Acc 73.46%, Dice 81.85%, Jaccard 70.91%, MIoU 0.447, MDR 7.71%, FDR 74.34% | FVC is a live-scan/synthetic benchmark, not latent SD302. The validation-tuned GMFS and fine-tuned SUFS rows are experiment-specific and have no original-source result. |
| U-Net | Jaafar et al., 2022 | not determined | U-Net: Acc 74.74%, Dice 79.50%, Jaccard 68.21%, MIoU 0.549, MDR 20.95%, FDR 32.78% | Do not substitute a different U-Net implementation or a secondary-source value. |
| Adaptive Gaussian transformer | Liu et al., 2022 | MSE 5.80; average NFIQ 49.50; EER 1.60% | SegFormer-B2 (substitute†): Acc 78.84%, Dice 85.59%, Jaccard 75.93%, MIoU 0.537, MDR 5.48%, FDR 60.77% | These are OCT layer/reconstruction/recognition results, not 2-D foreground ROI metrics. SegFormer-B2 is a declared substitute, not a reproduction. |
| DRUnet | Joshi et al., 2021 | Dice 96.8917%; Jaccard 94.2058% | DRUnet: Acc 76.85%, Dice 84.85%, Jaccard 74.39%, MIoU 0.508, MDR 5.17%, FDR 64.75% | These means were computed from the 12 exact printed cells and were not printed by the authors as averages; FVC is not latent SD302. |
| Recurrent adversarial network | Joshi et al., 2021 | Dice 88.6222%; Jaccard 81.2489% | RUnet (no alignment): Acc 76.17%, Dice 84.46%, Jaccard 73.97%, MIoU 0.486, MDR 4.25%, FDR 70.20% | The original setup is transductive unsupervised domain adaptation because target A images participate unlabeled in training; it is not comparable to a subject-disjoint held-out protocol. |
| Deep Nested UNets | Liu & Qian, 2021 | not determined | UNet++: Acc 74.49%, Dice 78.22%, Jaccard 67.49%, MIoU 0.546, MDR 21.92%, FDR 33.21% | Do not use the unchecked secondary-review PA/MPA/MIoU claim. The benchmark's native UNet++ is an implementation mapping, not a reproduction of the source segmentation-plus-enhancement network. |
| Support Vector Machine | Sankaran et al., 2017; generic classifier citation: Cortes & Vapnik, 1995 | SA 66.24 ± 0.4%; FSA 78.45 ± 0.1%; BSA 63.18 ± 0.5% | Support Vector Machine: Acc 69.12%, Dice 75.85%, Jaccard 62.26%, MIoU 0.488, MDR 27.91%, FDR 35.35% | These are block segmentation accuracy measures, not Dice. The benchmark's Nyström SVM is not an exact reproduction. |
| AlexNet & LeNet | Serafim et al., 2019; Cappelli, 2023 consolidation | AlexNet: Hausdorff 5.92, Dice 97.28%, Jaccard 96.77%, ER 3.22%; LeNet: ER 3.26%, Dice 97.18%, Jaccard 96.60% | AlexNet: Acc 80.33%, Dice 86.46%, Jaccard 76.62%, MIoU 0.598, MDR 8.68%, FDR 41.73%; LeNet: Acc 77.46%, Dice 84.93%, Jaccard 74.46%, MIoU 0.530, MDR 7.00%, FDR 58.00% | The original accessible abstract does not identify which architecture produced its reported architecture-level values. Cite Cappelli explicitly for architecture-specific context and the LeNet row. |
| Deep CNN | Tang et al., 2017 | TPR 0.88; TNR 0.92 | FingerNet (pretrained): Acc 60.00%, Dice 58.93%, Jaccard 46.45%, MIoU 0.430, MDR 49.94%, FDR 11.54% | No Dice or Jaccard was reported. Weak hull labels are not examiner ROI masks, and accuracy cannot be derived without class prevalence. |
| Random decision forest | Sankaran et al., 2017 | SA 73.76 ± 0.2%; FSA 83.41 ± 0.1%; BSA 71.34 ± 0.2% | Random decision forest: Acc 69.82%, Dice 76.82%, Jaccard 63.40%, MIoU 0.492, MDR 26.13%, FDR 37.35% | These are block metrics, not Dice. The benchmark's scikit-learn random forest is a documented reimplementation. |
| Weighted extreme learning machine | Ezeobiejesi & Bhanu, 2016 | Combined accuracy 91.43%; NIST SD27 MDR 9.22%, FDR 18.70%; WVU MDR 15.54%, FDR 9.65%; IIIT-D MDR 6.38%, FDR 10.07% | Weighted extreme learning machine: Acc 68.61%, Dice 75.27%, Jaccard 61.57%, MIoU 0.483, MDR 28.70%, FDR 35.10% | This is a patch-level protocol. The benchmark's 16×16 block implementation is a reimplementation, not an exact reproduction. |

The source-only comparison showed that the original studies used heterogeneous datasets, label definitions, and metrics. Reported outcomes included full-image overlap, block accuracy, patch accuracy, true-positive and true-negative rates, OCT reconstruction criteria, and transductive adaptation results. Consequently, the originally reported values were reproduced verbatim with their native protocols and were not treated as commensurate with the SD302 examiner-ROI results.

# 6. Discussion

## 6.1 Main findings

1. Ground-truth definition altered the observed family contrast. The deep-minus-classical Dice difference was 6.89 percentage points (95% CI 6.05 percentage points to 7.80 percentage points) under examiner ROI and -3.58 percentage points (95% CI -4.93 percentage points to -2.36 percentage points) under usable-ridge ground truth.
2. Patch CNNs occupied high-accuracy operating points under the coarse ROI: AlexNet reached 80.33% accuracy and 0.598 MIoU, while LeNet reached 77.46% and 0.530. Their usable-ridge Dice ranks changed to 7 and 8, respectively.
3. High recall coincided with substantial background inclusion for several dense models. DRUnet, RUnet, CNN + stacked AE, and SegFormer-B2 had recall values of 94.83%, 95.75%, 93.68%, and 94.52%, with FDR values of 64.75%, 70.20%, 69.41%, and 60.77%. U-Net and UNet++ had lower FDR values of 32.78% and 33.21%, together with recall values of 79.05% and 78.08%.
4. Fine-tuning SUFS increased recall by 19.30 percentage points and Dice by 6.90 percentage points relative to its zero-shot row, while precision decreased by 7.57 percentage points. FingerNet instead produced 92.79% precision and 50.06% recall. The three trained traditional classifiers had precision between 83.67% and 84.18%, but their MDR remained between 26.13% and 28.70%.
5. On mapped live-scan images, GMFS default had HR 99.69% with PAR 1.540 on segmented slaps and HR 99.54% with PAR 1.463 on rolls. AlexNet had HR 99.98% with PAR 2.180 and HR 99.67% with PAR 4.374, respectively. Inter-method Dice was 0.7263 for segmented slaps and 0.6018 for rolls; these were agreement values, not accuracy.
6. The source-paper comparison could not support a cross-paper numerical ranking because the reported studies used different datasets, supervision, units of analysis, and endpoints. The common SD302 split and evaluator addressed this comparability problem within the present study.

## 6.2 Implications for downstream forensic processing

MDR and FDR represented different downstream risks. A missed ridge-bearing region could remove minutiae before extraction, whereas a false foreground region could introduce clutter into ridge enhancement and candidate minutiae generation. The observed precision–recall differences therefore supported selecting an operating point according to the downstream matcher and review workflow, rather than relying on a single overlap statistic. No downstream rank-identification or verification claim was made because the present tables did not measure matcher performance.

## 6.3 Limitations

- Several methods were faithful reimplementations rather than executions of original author code, so implementation differences remained a source of variation.
- SegFormer-B2 was an explicitly labelled 2-D substitute for the OCT-specific adaptive Gaussian transformer and was not a reproduction of that method.
- Thresholds for pretrained zero-shot models were chosen on the SD302 validation split; this preserved test isolation but constituted target-domain calibration.
- The examiner ROI was a coarse region definition, while the usable-ridge quality map represented a narrower target. The observed rank changes showed that conclusions depended on this choice.
- The SD302g expert-minutiae hull was a lower bound on live-scan foreground, so hull metrics could not be interpreted as pixel-level Dice or FDR.
- All evaluated impressions came from the SD302 source, limiting evidence for cross-dataset generalisation.
- Human pixel masks for SD302b segmented-slap and rolled impressions were not available; the live-scan results therefore used a lower-bound hull and could not report pixel overlap.
- The segmented-slap subset contained only nine images from five subjects, and capture type was confounded with device (R/S for segmented slaps and U/V for rolls), limiting plain-versus-rolled and per-device inference.

## 6.4 Future work

Future work will add independently produced human masks for SD302b plain and rolled impressions, evaluate downstream rank-1 and verification performance with one fixed matcher, and test external live-scan data such as FVC with the Thai–Huckemann–Gottschlich ground truth. The same subject-isolation, validation-only selection, and locked-scoring rules should be retained.

---

Generation note: this draft was generated from authenticated locked-result aggregates and saved per-image-count statistics. It did not train a model, select a threshold, run inference, or score a mask. Every quantitative claim is mapped to its source in traceability.json.
