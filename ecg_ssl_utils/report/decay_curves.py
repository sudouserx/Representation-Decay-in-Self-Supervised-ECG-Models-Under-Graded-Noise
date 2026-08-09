"""Decay curve plotting: metric vs SNR with bootstrap CI."""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_decay_curves(metric_data, metric_name, noise_type,
                      snr_grid, encoder_names, save_path=None):
    """
    Plot a single metric's decay curve across SNR for multiple encoders.
    metric_data: {encoder_name: {'mean': [vals], 'ci_lo': [...], 'ci_hi': [...]}}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(encoder_names)))

    for i, enc in enumerate(encoder_names):
        d = metric_data[enc]
        ax.plot(snr_grid, d['mean'], 'o-', color=colors[i], label=enc, lw=2)
        ax.fill_between(snr_grid, d['ci_lo'], d['ci_hi'],
                        alpha=0.15, color=colors[i])

    ax.set_xlabel('SNR (dB)', fontsize=13)
    ax.set_ylabel(metric_name, fontsize=13)
    ax.set_title(f'{metric_name} vs SNR — {noise_type}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # higher SNR = less noise = left side
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig
