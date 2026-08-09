"""
PTB-XL dataset loader.
Loads waveforms and metadata from the PTB-XL dataset using wfdb.
"""

import os
import ast
import numpy as np
import pandas as pd
import wfdb
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm


def load_ptbxl(
    data_dir: str,
    sampling_rate: int = 500,
    signal_length: int = 5000,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load PTB-XL dataset waveforms and metadata.

    Parameters
    ----------
    data_dir : str
        Root directory of the PTB-XL dataset (containing ptbxl_database.csv
        and records500/ or records100/ subdirectories).
    sampling_rate : int
        Target sampling rate (500 or 100).
    signal_length : int
        Expected signal length (samples). 500 Hz × 10 s = 5000.

    Returns
    -------
    signals : np.ndarray
        Shape (N, 12, signal_length) — all ECG waveforms.
    metadata : pd.DataFrame
        Metadata with columns: ecg_id, patient_id, age, sex, scp_codes,
        strat_fold, filename.
    """
    # Load the metadata CSV
    meta_path = os.path.join(data_dir, 'ptbxl_database.csv')
    df = pd.read_csv(meta_path, index_col='ecg_id')

    # Parse scp_codes from string to dict
    df['scp_codes'] = df['scp_codes'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Determine which filename column to use
    if sampling_rate == 500:
        filename_col = 'filename_hr'
    else:
        filename_col = 'filename_lr'

    # Load all signals
    signals = []
    valid_indices = []

    print(f"Loading PTB-XL signals at {sampling_rate} Hz...")
    for idx in tqdm(df.index, desc="Loading ECGs"):
        fpath = os.path.join(data_dir, df.loc[idx, filename_col])
        try:
            record = wfdb.rdsamp(fpath)
            sig = record[0]  # (signal_length, n_leads)

            # Transpose to (n_leads, signal_length)
            sig = sig.T.astype(np.float32)

            # Handle length mismatch
            if sig.shape[1] < signal_length:
                # Pad with zeros
                pad_width = signal_length - sig.shape[1]
                sig = np.pad(sig, ((0, 0), (0, pad_width)), mode='constant')
            elif sig.shape[1] > signal_length:
                sig = sig[:, :signal_length]

            signals.append(sig)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Warning: Could not load record {idx}: {e}")
            continue

    signals = np.stack(signals, axis=0)  # (N, 12, 5000)

    # Filter metadata to only valid records
    df = df.loc[valid_indices].copy()
    df = df.reset_index()

    # Standardize column names
    df = df.rename(columns={
        'patient_id': 'patient_id',
        'age': 'age',
        'sex': 'sex',
        'strat_fold': 'strat_fold',
    })

    # Convert sex: 0=male, 1=female
    df['sex_numeric'] = df['sex'].map({0: 0, 1: 1}).fillna(-1).astype(int)

    print(f"Loaded {len(signals)} records, shape: {signals.shape}")
    return signals, df


def get_patient_split(
    metadata: pd.DataFrame,
    train_folds: List[int],
    val_folds: List[int],
    test_folds: List[int],
) -> Dict[str, np.ndarray]:
    """
    Create patient-level train/val/test split using PTB-XL's strat_fold.

    Returns dict with keys 'train', 'val', 'test' → arrays of integer indices
    into the metadata DataFrame.
    """
    train_idx = metadata[metadata['strat_fold'].isin(train_folds)].index.values
    val_idx = metadata[metadata['strat_fold'].isin(val_folds)].index.values
    test_idx = metadata[metadata['strat_fold'].isin(test_folds)].index.values

    print(f"Split sizes — Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Verify patient-level separation
    train_patients = set(metadata.loc[train_idx, 'patient_id'].unique())
    val_patients = set(metadata.loc[val_idx, 'patient_id'].unique())
    test_patients = set(metadata.loc[test_idx, 'patient_id'].unique())

    assert len(train_patients & val_patients) == 0, "Patient leak: train ∩ val"
    assert len(train_patients & test_patients) == 0, "Patient leak: train ∩ test"
    assert len(val_patients & test_patients) == 0, "Patient leak: val ∩ test"
    print("✓ No patient leakage across splits.")

    return {'train': train_idx, 'val': val_idx, 'test': test_idx}
