"""Kendall-τ rank stability across Dirichlet weight draws.

Replaces the mis-specified Dirichlet-dispersion 'Sobol' proxy.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .robustness_score import compute_robustness_score


def weight_rank_stability(
    dss_df: pd.DataFrame,
    base_weights: Dict[str, float],
    n_samples: int = 1024,
    seed: int = 42,
    weight_keys: Tuple[str, ...] = ("cka", "erank", "ece", "auroc_decay"),
) -> Dict:
    """Compare rankings under Dirichlet weight perturbation vs pre-registered weights.

    Returns mean Kendall τ, fraction of top-1 agreement, and per-model
    fraction of draws that preserve the model's base rank.
    """
    model_ids = list(dss_df["model_id"].unique())
    if len(model_ids) < 2:
        return {
            "mean_kendall_tau": 1.0,
            "frac_top1_agreement": 1.0,
            "per_model_stability": {m: 1.0 for m in model_ids},
        }

    def scores_for(weights):
        out = {}
        for mid, g in dss_df.groupby("model_id"):
            vals = [
                compute_robustness_score(
                    r["delta_cka_norm"], r["delta_erank_norm"],
                    r["delta_ece_norm"], r["delta_auroc_norm"],
                    weights=weights,
                )[0]
                for _, r in g.iterrows()
            ]
            out[mid] = float(np.mean(vals))
        return pd.Series(out)

    base_scores = scores_for(base_weights)
    base_ranking = base_scores.rank(ascending=False, method="average")
    base_top = base_scores.idxmax()

    rng = np.random.RandomState(seed)
    taus, top1 = [], []
    keep_rank = {m: 0 for m in model_ids}
    for _ in range(n_samples):
        w = dict(zip(weight_keys, rng.dirichlet(np.ones(len(weight_keys)))))
        scores = scores_for(w)
        ranking = scores.rank(ascending=False, method="average")
        tau = kendalltau(ranking.reindex(base_ranking.index), base_ranking).statistic
        taus.append(float(tau) if tau == tau else 0.0)
        top1.append(scores.idxmax() == base_top)
        for m in model_ids:
            if ranking[m] == base_ranking[m]:
                keep_rank[m] += 1

    return {
        "mean_kendall_tau": float(np.mean(taus)),
        "frac_top1_agreement": float(np.mean(top1)),
        "per_model_stability": {m: keep_rank[m] / n_samples for m in model_ids},
    }
