"""
SCP-ECG label encoding for PTB-XL multi-label classification.
Supports both 71 fine-grained SCP codes and 5 diagnostic superclasses.
"""

import os
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional


# All 71 SCP-ECG codes used in PTB-XL
# Grouped by superclass for reference
SCP_SUPERCLASSES = {
    'NORM': 'Normal ECG',
    'MI': 'Myocardial Infarction',
    'STTC': 'ST/T Change',
    'CD': 'Conduction Disturbance',
    'HYP': 'Hypertrophy',
}

# Ordered superclass names for consistent indexing
SUPERCLASS_NAMES: List[str] = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def get_scp_code_list(metadata: pd.DataFrame) -> List[str]:
    """
    Extract all unique SCP codes from the dataset.

    Parameters
    ----------
    metadata : pd.DataFrame
        Must have 'scp_codes' column with dict values.

    Returns
    -------
    sorted list of unique SCP code strings.
    """
    all_codes = set()
    for codes_dict in metadata['scp_codes']:
        if isinstance(codes_dict, dict):
            all_codes.update(codes_dict.keys())
    return sorted(all_codes)


def get_label_map(metadata: pd.DataFrame) -> Dict[str, int]:
    """
    Create a mapping from SCP code to index.

    Returns
    -------
    dict: {scp_code_str: index}
    """
    codes = get_scp_code_list(metadata)
    return {code: idx for idx, code in enumerate(codes)}


def encode_scp_labels(
    metadata: pd.DataFrame,
    label_map: Optional[Dict[str, int]] = None,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Encode SCP codes as multi-hot vectors.

    Parameters
    ----------
    metadata : pd.DataFrame
        Must have 'scp_codes' column with dict values {code: likelihood}.
    label_map : dict, optional
        Mapping from SCP code to index. If None, auto-generated.
    threshold : float
        Minimum likelihood to include a label (PTB-XL uses 0-100 scale).
        Default 0.0 includes all annotated codes.

    Returns
    -------
    labels : np.ndarray
        Shape (N, n_classes) multi-hot float32.
    """
    if label_map is None:
        label_map = get_label_map(metadata)

    n_classes = len(label_map)
    n_samples = len(metadata)
    labels = np.zeros((n_samples, n_classes), dtype=np.float32)

    for i, codes_dict in enumerate(metadata['scp_codes']):
        if isinstance(codes_dict, dict):
            for code, likelihood in codes_dict.items():
                if code in label_map and likelihood > threshold:
                    labels[i, label_map[code]] = 1.0

    return labels


def save_label_map(label_map: Dict[str, int], path: str):
    """Save label map to JSON."""
    with open(path, 'w') as f:
        json.dump(label_map, f, indent=2)


def load_label_map(path: str) -> Dict[str, int]:
    """Load label map from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def get_superclass_mapping(scp_statements_path: str) -> Dict[str, str]:
    """
    Load the SCP statements CSV and create code → superclass mapping.

    Parameters
    ----------
    scp_statements_path : str
        Path to scp_statements.csv from PTB-XL.

    Returns
    -------
    dict: {scp_code: superclass_name}
    """
    df = pd.read_csv(scp_statements_path, index_col=0)
    mapping = {}
    for code in df.index:
        # diagnostic_class column contains the superclass
        if 'diagnostic_class' in df.columns:
            superclass = df.loc[code, 'diagnostic_class']
            if pd.notna(superclass):
                mapping[code] = superclass
    return mapping


def encode_superclass_labels(
    metadata: pd.DataFrame,
    scp_statements_path: str,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Encode SCP codes as 5-class superclass multi-hot vectors.

    Uses scp_statements.csv to map individual SCP codes to their
    diagnostic superclass (NORM, MI, STTC, CD, HYP), then creates
    a (N, 5) binary array.

    Parameters
    ----------
    metadata : pd.DataFrame
        Must have 'scp_codes' column with dict values {code: likelihood}.
    scp_statements_path : str
        Path to scp_statements.csv from PTB-XL dataset.
    threshold : float
        Minimum likelihood to include a label (PTB-XL uses 0-100 scale).
        Default 0.0 includes all annotated codes.

    Returns
    -------
    labels : np.ndarray
        Shape (N, 5) multi-hot float32, columns ordered as SUPERCLASS_NAMES.
    """
    # Build code → superclass mapping
    code_to_super = get_superclass_mapping(scp_statements_path)
    super_to_idx = {name: i for i, name in enumerate(SUPERCLASS_NAMES)}

    n_samples = len(metadata)
    labels = np.zeros((n_samples, len(SUPERCLASS_NAMES)), dtype=np.float32)

    for i, codes_dict in enumerate(metadata['scp_codes']):
        if isinstance(codes_dict, dict):
            for code, likelihood in codes_dict.items():
                if code in code_to_super and likelihood > threshold:
                    superclass = code_to_super[code]
                    if superclass in super_to_idx:
                        labels[i, super_to_idx[superclass]] = 1.0

    return labels
