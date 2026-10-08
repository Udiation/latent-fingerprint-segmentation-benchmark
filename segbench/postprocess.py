"""One shared post-processing step for all block-based classical methods, so
differences in the results table come from the classifier, not the clean-up."""
import numpy as np
from scipy import ndimage

_STRUCT = np.ones((3, 3), dtype=bool)


def postprocess_blocks(block_mask, min_component_blocks=4):
    b = np.pad(block_mask.astype(bool), 1, mode="edge")
    b = ndimage.binary_closing(b, structure=_STRUCT)
    b = ndimage.binary_opening(b, structure=_STRUCT)
    b = ndimage.binary_fill_holes(b)
    b = b[1:-1, 1:-1]
    if min_component_blocks > 1 and b.any():
        lab, n = ndimage.label(b, structure=_STRUCT)
        sizes = ndimage.sum(b, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = sizes >= min_component_blocks
        b = keep[lab]
    return b
