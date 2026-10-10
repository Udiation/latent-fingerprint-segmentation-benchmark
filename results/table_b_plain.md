# SD302b Plain (segmented slap) — hull evaluation

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

Hull ground truth is a lower bound of the true foreground; pixel Dice is not reported. Values are mean ± seed SD [subject-bootstrap 95% CI]. Timing was carried from locked Table e.
