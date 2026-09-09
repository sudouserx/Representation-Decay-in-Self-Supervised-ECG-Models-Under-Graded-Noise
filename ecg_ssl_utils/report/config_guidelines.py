import os
from typing import Dict, List

from ecg_ssl_utils.score.decision_support import ConfigurationResult


def generate_config_guidelines(
    results: List[ConfigurationResult],
    noise_condition: str,
    constraints: Dict[str, float],
    output_dir: str,
) -> str:
    """Deterministic configuration guidelines (R2) using real profile fields."""
    os.makedirs(output_dir, exist_ok=True)

    recommended = [r for r in results if r.status in ["RECOMMEND", "RECOMMEND_QUALIFIED"]]

    if not recommended:
        guideline_md = f"""# Configuration Guidelines
## Condition: {noise_condition}

**Status:** REJECT ALL
No configuration satisfied all constraints and robustness gates.

### Constraints:
- Min Robustness index: {constraints.get('min_robustness', 'N/A')}
- Max Latency (ms): {constraints.get('max_latency_ms', 'N/A')}
- Max Memory (MB): {constraints.get('max_memory_mb', 'N/A')}

### Top Rejection Reasons:
"""
        for r in results[:5]:
            guideline_md += f"- **{r.model_id} ({r.quantization} on {r.hardware})**: {', '.join(r.rejection_reasons)}\n"
    else:
        top = recommended[0]
        guideline_md = f"""# Configuration Guidelines
## Condition: {noise_condition}

**Status:** {top.status}

### Recommended Configuration:
- **Model**: {top.model_id}
- **Quantization**: {top.quantization}
- **Hardware**: {top.hardware} (ONNX execution provider — server-proxy, not edge hardware identity)

### Evidence:
- **Robustness index**: {top.robustness_score:.4f}
- **Weight-stability (Kendall τ rank retention)**: {top.weight_stability:.3f}
- **Latency p95 (ms)**: {top.latency_p95_ms:.2f}
- **Memory (process RSS delta, MB)**: {top.memory_mb:.2f}
- **Quantization parity cosine**: {top.parity_cosine if top.parity_cosine is not None else 'n/a'}

### Applied Constraints:
- Min Robustness: {constraints.get('min_robustness', 'N/A')}
- Max Latency (ms): {constraints.get('max_latency_ms', 'N/A')}
- Max Memory (MB): {constraints.get('max_memory_mb', 'N/A')}

### Alternative Options:
"""
        for r in recommended[1:4]:
            guideline_md += (
                f"- **{r.model_id} ({r.quantization} on {r.hardware})**: "
                f"index {r.robustness_score:.4f}, latency {r.latency_p95_ms:.1f} ms, "
                f"status {r.status}\n"
            )

    out_path = os.path.join(output_dir, "config_guidelines.md")
    with open(out_path, "w") as f:
        f.write(guideline_md)
    return out_path
