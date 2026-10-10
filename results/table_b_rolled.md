# SD302b rolled — hull evaluation

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

Hull ground truth is a lower bound of the true foreground; pixel Dice is not reported. Values are mean ± seed SD [subject-bootstrap 95% CI]. Timing was carried from locked Table e.
