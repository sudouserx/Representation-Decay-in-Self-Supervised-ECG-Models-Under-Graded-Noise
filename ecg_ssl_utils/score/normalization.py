"""Reference-anchored min-max normalization."""
import numpy as np


def reference_anchored_normalize(values, eps=1e-8):
    """Normalize array to [0,1] using min-max. Returns normalized array."""
    vmin, vmax = np.min(values), np.max(values)
    return (values - vmin) / (vmax - vmin + eps)
