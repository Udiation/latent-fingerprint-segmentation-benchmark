## Table e-1 — SD302-Latent by examiner value

| Method | Examiner value | n | Dice (%) | MIoU | MDR (%) | FDR (%) | Seeds |
|---|---|---:|---:|---:|---:|---:|---:|
| GMFS (default) | VALUE | 743 | 75.74 | 0.480 | 28.40 | 35.16 | 1 |
| GMFS (default) | LIMITED | 336 | 75.59 | 0.456 | 24.80 | 46.22 | 1 |
| GMFS (default) | NOVALUE | 922 | 76.10 | 0.453 | 22.64 | 49.49 | 1 |
| GMFS (val-tuned) | VALUE | 743 | 78.86 | 0.466 | 19.66 | 51.70 | 1 |
| GMFS (val-tuned) | LIMITED | 336 | 79.31 | 0.442 | 15.26 | 63.42 | 1 |
| GMFS (val-tuned) | NOVALUE | 922 | 80.21 | 0.442 | 12.47 | 67.26 | 1 |
| Support Vector Machine | VALUE | 743 | 75.80 ± 0.05 | 0.498 ± 0.001 | 30.06 ± 0.03 | 29.36 ± 0.07 | 3 |
| Support Vector Machine | LIMITED | 336 | 76.16 ± 0.05 | 0.484 ± 0.001 | 26.57 ± 0.18 | 39.63 ± 0.38 | 3 |
| Support Vector Machine | NOVALUE | 922 | 75.73 ± 0.09 | 0.482 ± 0.001 | 26.78 ± 0.10 | 38.52 ± 0.13 | 3 |
| Random decision forest | VALUE | 743 | 76.61 ± 0.04 | 0.503 ± 0.000 | 28.68 ± 0.05 | 30.82 ± 0.12 | 3 |
| Random decision forest | LIMITED | 336 | 77.02 ± 0.14 | 0.486 ± 0.002 | 24.94 ± 0.20 | 41.49 ± 0.17 | 3 |
| Random decision forest | NOVALUE | 922 | 76.86 ± 0.06 | 0.486 ± 0.000 | 24.62 ± 0.18 | 41.03 ± 0.39 | 3 |
| Weighted extreme learning machine | VALUE | 743 | 75.17 ± 0.17 | 0.494 ± 0.002 | 31.04 ± 0.22 | 28.78 ± 0.09 | 3 |
| Weighted extreme learning machine | LIMITED | 336 | 75.65 ± 0.03 | 0.478 ± 0.001 | 27.19 ± 0.09 | 39.79 ± 0.26 | 3 |
| Weighted extreme learning machine | NOVALUE | 922 | 75.15 ± 0.28 | 0.476 ± 0.002 | 27.49 ± 0.34 | 38.37 ± 0.12 | 3 |
| LeNet | VALUE | 743 | 85.38 ± 0.16 | 0.529 ± 0.008 | 6.49 ± 0.89 | 59.74 ± 2.86 | 3 |
| LeNet | LIMITED | 336 | 84.93 ± 0.15 | 0.526 ± 0.010 | 6.42 ± 0.75 | 60.42 ± 2.85 | 3 |
| LeNet | NOVALUE | 922 | 84.55 ± 0.23 | 0.532 ± 0.005 | 7.63 ± 0.81 | 55.68 ± 1.95 | 3 |
| AlexNet | VALUE | 743 | 87.20 ± 0.04 | 0.600 ± 0.002 | 7.08 ± 0.14 | 44.54 ± 0.53 | 3 |
| AlexNet | LIMITED | 336 | 86.49 ± 0.05 | 0.599 ± 0.001 | 8.51 ± 0.06 | 42.84 ± 0.09 | 3 |
| AlexNet | NOVALUE | 922 | 85.86 ± 0.06 | 0.595 ± 0.000 | 10.04 ± 0.22 | 39.03 ± 0.44 | 3 |
| CNN + stacked AE | VALUE | 743 | 84.78 ± 0.09 | 0.495 ± 0.012 | 6.00 ± 0.80 | 67.94 ± 3.72 | 3 |
| CNN + stacked AE | LIMITED | 336 | 83.97 ± 0.06 | 0.477 ± 0.008 | 5.72 ± 0.73 | 71.62 ± 2.62 | 3 |
| CNN + stacked AE | NOVALUE | 922 | 83.29 ± 0.22 | 0.476 ± 0.007 | 6.82 ± 1.02 | 69.82 ± 2.71 | 3 |
| U-Net | VALUE | 743 | 84.01 ± 0.57 | 0.576 ± 0.016 | 14.28 ± 0.53 | 37.94 ± 2.08 | 3 |
| U-Net | LIMITED | 336 | 80.05 ± 1.24 | 0.552 ± 0.008 | 20.70 ± 1.66 | 32.57 ± 1.73 | 3 |
| U-Net | NOVALUE | 922 | 75.69 ± 2.37 | 0.527 ± 0.011 | 26.35 ± 3.39 | 28.69 ± 2.37 | 3 |
| UNet++ | VALUE | 743 | 84.71 ± 0.36 | 0.587 ± 0.021 | 12.84 ± 2.93 | 39.42 ± 9.74 | 3 |
| UNet++ | LIMITED | 336 | 78.88 ± 0.77 | 0.545 ± 0.018 | 21.66 ± 3.37 | 33.71 ± 10.12 | 3 |
| UNet++ | NOVALUE | 922 | 72.75 ± 0.75 | 0.514 ± 0.011 | 29.31 ± 3.51 | 27.99 ± 9.05 | 3 |
| DRUnet | VALUE | 743 | 85.67 ± 0.59 | 0.506 ± 0.037 | 3.57 ± 1.09 | 68.23 ± 8.47 | 3 |
| DRUnet | LIMITED | 336 | 84.77 ± 0.71 | 0.505 ± 0.046 | 5.01 ± 1.55 | 66.10 ± 10.80 | 3 |
| DRUnet | NOVALUE | 922 | 84.20 ± 0.51 | 0.510 ± 0.040 | 6.55 ± 1.70 | 61.34 ± 10.80 | 3 |
| RUnet (no alignment) | VALUE | 743 | 85.51 ± 0.46 | 0.490 ± 0.025 | 2.84 ± 0.82 | 72.09 ± 5.56 | 3 |
| RUnet (no alignment) | LIMITED | 336 | 84.51 ± 0.34 | 0.478 ± 0.026 | 3.55 ± 1.43 | 73.32 ± 6.68 | 3 |
| RUnet (no alignment) | NOVALUE | 922 | 83.58 ± 0.54 | 0.486 ± 0.031 | 5.66 ± 2.66 | 67.47 ± 8.82 | 3 |
| SegFormer-B2 (substitute†) | VALUE | 743 | 87.19 ± 1.88 | 0.557 ± 0.109 | 4.01 ± 2.97 | 59.86 ± 24.00 | 3 |
| SegFormer-B2 (substitute†) | LIMITED | 336 | 85.51 ± 2.09 | 0.531 ± 0.100 | 4.96 ± 2.38 | 62.25 ± 21.46 | 3 |
| SegFormer-B2 (substitute†) | NOVALUE | 922 | 84.31 ± 1.97 | 0.522 ± 0.086 | 6.85 ± 2.15 | 61.02 ± 20.11 | 3 |
| SUFS (zero-shot) | VALUE | 743 | 74.96 | 0.481 | 29.62 | 33.00 | 1 |
| SUFS (zero-shot) | LIMITED | 336 | 75.05 | 0.463 | 25.89 | 42.83 | 1 |
| SUFS (zero-shot) | NOVALUE | 922 | 74.87 | 0.467 | 25.38 | 42.60 | 1 |
| SUFS (fine-tuned) | VALUE | 743 | 84.47 ± 0.75 | 0.465 ± 0.045 | 4.15 ± 1.58 | 75.91 ± 10.83 | 3 |
| SUFS (fine-tuned) | LIMITED | 336 | 82.36 ± 0.51 | 0.444 ± 0.035 | 6.48 ± 1.54 | 76.08 ± 9.13 | 3 |
| SUFS (fine-tuned) | NOVALUE | 922 | 79.56 ± 0.79 | 0.434 ± 0.036 | 11.03 ± 1.55 | 72.36 ± 10.40 | 3 |
| FingerNet (pretrained) | VALUE | 743 | 78.67 | 0.560 | 28.29 | 17.60 | 1 |
| FingerNet (pretrained) | LIMITED | 336 | 65.09 | 0.448 | 45.47 | 13.18 | 1 |
| FingerNet (pretrained) | NOVALUE | 922 | 40.55 | 0.318 | 69.35 | 5.74 | 1 |
| All background | VALUE | 743 | 0.00 | 0.137 | 100.00 | 0.00 | 1 |
| All background | LIMITED | 336 | 0.00 | 0.142 | 100.00 | 0.00 | 1 |
| All background | NOVALUE | 922 | 0.00 | 0.143 | 100.00 | 0.00 | 1 |
| All foreground | VALUE | 743 | 83.57 | 0.364 | 0.00 | 100.00 | 1 |
| All foreground | LIMITED | 336 | 82.95 | 0.361 | 0.00 | 100.00 | 1 |
| All foreground | NOVALUE | 922 | 82.52 | 0.361 | 0.00 | 100.00 | 1 |

Examiner-value breakdown of the locked Table-e test results. MIXED (n=9) is retained in overall Table e but excluded from the requested VALUE/LIMITED/NOVALUE companion table.
