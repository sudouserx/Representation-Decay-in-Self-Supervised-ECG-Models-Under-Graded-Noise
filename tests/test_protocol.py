"""Protocol regression tests for labels, accumulation, and patient batching."""

import numpy as np
import pandas as pd

from ecg_ssl_utils.data.label_encoder import encode_superclass_labels
from ecg_ssl_utils.repro import PatientPairBatchSampler, accumulation_state


def test_official_labels_include_zero_likelihood_and_only_diagnostic(tmp_path):
    statements = pd.DataFrame(
        {
            "diagnostic": [1.0, 1.0, 0.0],
            "diagnostic_class": ["MI", "NORM", np.nan],
        },
        index=["AMI", "NORM", "AFIB"],
    )
    path = tmp_path / "scp_statements.csv"
    statements.to_csv(path)
    metadata = pd.DataFrame(
        {"scp_codes": [{"AMI": 0.0, "AFIB": 100.0}, {"NORM": 100.0}]}
    )

    official = encode_superclass_labels(metadata, str(path), threshold=None)
    strict_gt50 = encode_superclass_labels(metadata, str(path), threshold=50.0)

    assert official[0, 1] == 1.0  # MI is included even with unknown likelihood.
    assert strict_gt50[0].sum() == 0.0
    assert official[:, 0].tolist() == [0.0, 1.0]


def test_accumulation_steps_final_partial_window():
    states = [accumulation_state(i, total_steps=59, grad_accum_steps=4) for i in range(59)]
    step_indices = [i for i, (_, should_step) in enumerate(states) if should_step]

    assert step_indices[-1] == 58
    assert len(step_indices) == 15
    assert states[56][0] == states[58][0] == 3


def test_patient_pair_sampler_guarantees_two_distinct_patient_pairs():
    patient_ids = np.repeat(np.arange(20), 2)
    sampler = PatientPairBatchSampler(
        patient_ids, batch_size=8, seed=7, pairs_per_batch=2, drop_last=True,
    )

    for batch in sampler:
        batch_pids = patient_ids[batch]
        _, counts = np.unique(batch_pids, return_counts=True)
        assert (counts >= 2).sum() >= 2
        assert len(batch) == len(set(batch))
