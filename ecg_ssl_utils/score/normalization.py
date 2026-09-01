"""
Reference-anchored min-max normalization.

METHODOLOGY NOTE:
Global min-max normalization is anchored to the observed worst/best case in the 
current experimental grid. If a future run contains different corruption extremes, 
all normalized scores will shift. This transformation is valid for within-experiment 
ranking, but limits absolute comparability across independent studies.
"""
import numpy as np


def reference_anchored_normalize(values, eps=1e-8):
    """Normalize array to [0,1] using min-max. Returns normalized array."""
    vmin, vmax = np.min(values), np.max(values)
    return (values - vmin) / (vmax - vmin + eps)
