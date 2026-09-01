"""Markdown model card generator."""


def generate_model_card(model_info, save_path):
    """Generate a markdown model card for an SSL encoder."""
    card = f"""# Model Card: {model_info['model_id']}

## Overview
- **SSL Paradigm**: {model_info['paradigm']}
- **Backbone**: {model_info['backbone']}
- **Parameters**: {model_info.get('n_params', 'N/A')}
- **Training Epochs**: {model_info.get('epochs', 200)}
- **Training Data**: PTB-XL (folds 1-8)

## Clean Performance
- **Macro AUROC**: {model_info.get('clean_auroc', 'N/A'):.4f}
- **ECE**: {model_info.get('clean_ece', 'N/A'):.4f}

## Noise Robustness
- **CKA @ 0dB (worst single noise)**: {model_info.get('cka_0db', 'N/A'):.4f}
- **AUROC @ -6dB (worst case)**: {model_info.get('auroc_neg6db', 'N/A'):.4f}

## Deployment Cost
- **FP32 Latency (p50)**: {model_info.get('latency_fp32', 'N/A'):.4f} s
- **INT8 Latency (p50)**: {model_info.get('latency_int8', 'N/A'):.4f} s
- **Model Size (FP32)**: {model_info.get('size_fp32', 'N/A'):.1f} MB
- **Estimated Energy**: {model_info.get('estimated_energy_j', 'N/A'):.4f} J

## Robustness Score
- **Robustness Score (equal weights)**: {model_info.get('robustness_score', 'N/A'):.4f}
- **95% CI**: [{model_info.get('robustness_score_ci_lo', 'N/A'):.4f}, {model_info.get('robustness_score_ci_hi', 'N/A'):.4f}]

## Limitations
- Evaluated on PTB-XL only (single-center German cohort)
- Edge latency is simulated, not measured on physical device
- Inverter switching noise is synthetically approximated
"""
    with open(save_path, 'w') as f:
        f.write(card)
    return save_path
