"""
DeLong test for comparing two paired AUROCs.
Z = (AUC_A - AUC_B) / sqrt(Var(A) + Var(B) - 2*Cov(A,B))
Reference: DeLong et al., Biometrics 1988.
Uses scipy for computation.
"""
import numpy as np
from scipy import stats


def _compute_midrank(x):
    """Compute midranks for the DeLong test."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        for k in range(i, j):
            T[k] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N)
    T2[J] = T + 1
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    aucs = np.zeros(k)
    for j in range(k):
        all_pred = np.concatenate([positive_examples[j], negative_examples[j]])
        ranks = _compute_midrank(all_pred)
        aucs[j] = (np.sum(ranks[:m]) - m*(m+1)/2) / (m*n)
    return aucs


def delong_test(y_true, y_score_a, y_score_b):
    """
    Two-sided DeLong test comparing AUC of model A vs B.
    y_true: (N,) binary, y_score_a/b: (N,) scores.
    Returns (z_stat, p_value, auc_a, auc_b).
    """
    order = np.argsort(-y_true)  # positives first
    label_ordered = y_true[order]
    m = int(label_ordered.sum())

    preds = np.vstack([y_score_a[order], y_score_b[order]])
    aucs = _fast_delong(preds, m)

    n = len(y_true) - m
    pos_a, neg_a = y_score_a[order][:m], y_score_a[order][m:]
    pos_b, neg_b = y_score_b[order][:m], y_score_b[order][m:]

    # Structural components for variance
    # Placement values
    v10_a = np.array([np.mean(neg_a < p) + 0.5*np.mean(neg_a == p) for p in pos_a])
    v01_a = np.array([np.mean(pos_a > n_) + 0.5*np.mean(pos_a == n_) for n_ in neg_a])
    v10_b = np.array([np.mean(neg_b < p) + 0.5*np.mean(neg_b == p) for p in pos_b])
    v01_b = np.array([np.mean(pos_b > n_) + 0.5*np.mean(pos_b == n_) for n_ in neg_b])

    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    S = s10 / m + s01 / n

    diff = aucs[0] - aucs[1]
    var_diff = S[0,0] + S[1,1] - 2*S[0,1]
    if var_diff < 1e-12:
        return 0.0, 1.0, aucs[0], aucs[1]
    z = diff / np.sqrt(var_diff)
    p = 2 * stats.norm.sf(abs(z))
    return float(z), float(p), float(aucs[0]), float(aucs[1])
