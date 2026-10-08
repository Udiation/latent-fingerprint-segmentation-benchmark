"""Block-wise handcrafted features for the classical segmenters (SVM, RF, WELM).

Each image (already at the target ppi) is cut into BLOCK x BLOCK blocks. Every block
is described by statistics of a WINDOW x WINDOW neighbourhood centred on it, so the
classifier sees a little context around each block. All window sums use integral
images, so feature extraction is vectorised (no Python loop over blocks).

Features (in order, see FEATURE_NAMES):
  mean, std                     - intensity level and contrast in the window
  mean_rel, std_rel             - the same relative to the whole image (helps across devices)
  grad_mag                      - mean gradient magnitude
  grad_energy                   - mean (Gx^2 + Gy^2)
  coherence                     - ridge-orientation consistency (Bazen & Gerez style)
  fractal_dim                   - differential box-counting fractal dimension
                                  (the feature family used by Ezeobiejesi & Bhanu 2016)
"""
import numpy as np
import cv2

BLOCK = 16
WINDOW = 32
HALO = (WINDOW - BLOCK) // 2
FD_SCALES = (2, 4, 8, 16)
GRAY_LEVELS = 256

FEATURE_NAMES = ["mean", "std", "mean_rel", "std_rel", "grad_mag",
                 "grad_energy", "coherence", "fractal_dim"]


def grid_shape(h, w):
    return -(-h // BLOCK), -(-w // BLOCK)


def _pad(img):
    h, w = img.shape
    hb, wb = grid_shape(h, w)
    pad_b = hb * BLOCK - h + HALO
    pad_r = wb * BLOCK - w + HALO
    return np.pad(img, ((HALO, pad_b), (HALO, pad_r)), mode="reflect"), hb, wb


def _window_sum(cells, cell, hb, wb):
    """Sum of `cells` (one value per cell x cell pixel tile of the padded image)
    over each block's WINDOW-pixel neighbourhood."""
    length = WINDOW // cell
    stride = BLOCK // cell
    s = np.zeros((cells.shape[0] + 1, cells.shape[1] + 1), dtype=np.float64)
    s[1:, 1:] = cells.cumsum(0).cumsum(1)
    r = (np.arange(hb) * stride)[:, None]
    c = (np.arange(wb) * stride)[None, :]
    return s[r + length, c + length] - s[r, c + length] - s[r + length, c] + s[r, c]


def _fractal_dimension(p8, hb, wb):
    """Differential box counting (Sarkar & Chaudhuri) per block window."""
    log_n = []
    for s in FD_SCALES:
        h_box = s * GRAY_LEVELS / WINDOW
        hc, wc = p8.shape[0] // s, p8.shape[1] // s
        tiles = p8[:hc * s, :wc * s].reshape(hc, s, wc, s)
        mx = tiles.max(axis=(1, 3))
        mn = tiles.min(axis=(1, 3))
        n_r = np.floor(mx / h_box) - np.floor(mn / h_box) + 1.0
        log_n.append(np.log(_window_sum(n_r, s, hb, wb)))
    x = np.log(WINDOW / np.asarray(FD_SCALES, dtype=np.float64))
    xc = x - x.mean()
    y = np.stack(log_n, axis=-1)
    return ((y - y.mean(axis=-1, keepdims=True)) @ xc) / (xc @ xc)


def extract_block_features(img_u8):
    """img_u8: 2-D uint8 image at the target ppi -> (hb, wb, n_features) float32."""
    img = img_u8.astype(np.float64) / 255.0
    p, hb, wb = _pad(img)
    n = float(WINDOW * WINDOW)

    mean = _window_sum(p, 1, hb, wb) / n
    mean_sq = _window_sum(p * p, 1, hb, wb) / n
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))

    gx = cv2.Sobel(p, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(p, cv2.CV_64F, 0, 1, ksize=3)
    gxx = _window_sum(gx * gx, 1, hb, wb) / n
    gyy = _window_sum(gy * gy, 1, hb, wb) / n
    gxy = _window_sum(gx * gy, 1, hb, wb) / n
    grad_mag = _window_sum(np.sqrt(gx * gx + gy * gy), 1, hb, wb) / n
    energy = gxx + gyy
    coherence = np.sqrt((gxx - gyy) ** 2 + 4.0 * gxy ** 2) / (energy + 1e-12)

    img_mean, img_std = img.mean(), img.std() + 1e-6
    fd = _fractal_dimension(p * 255.0, hb, wb)

    feats = np.stack([mean, std, mean - img_mean, std / img_std, grad_mag,
                      energy, coherence, fd], axis=-1)
    return feats.astype(np.float32)


def block_labels(mask, hb, wb, threshold=0.5):
    """Block is foreground if at least `threshold` of its pixels are foreground."""
    h, w = mask.shape
    m = np.zeros((hb * BLOCK, wb * BLOCK), dtype=np.float32)
    m[:h, :w] = mask
    return m.reshape(hb, BLOCK, wb, BLOCK).mean(axis=(1, 3)) >= threshold


def blocks_to_mask(block_mask, shape):
    up = np.repeat(np.repeat(block_mask, BLOCK, axis=0), BLOCK, axis=1)
    return up[:shape[0], :shape[1]]
