import os
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List
from ecg_ssl_utils.score.decision_support import ConfigurationResult

@dataclass
class ConfigGuideline:
    noise_condition: str
    constraints: Dict[str, float]
    recommended_model: str
    recommended_quantization: str
    recommended_hardware: str
    evidence: Dict[str, float]  # robustness, latency, memory, sobol_stability

def generate_config_guidelines(
    results: List[ConfigurationResult],
    noise_condition: str,
    constraints: Dict[str, float],
    output_dir: str
) -> str:
    """
    Generate deterministic configuration guidelines based on Decision Support evaluation.
    This fulfills the R2 methodology artifact.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter to recommended ones
    recommended = [r for r in results if r.status in ['RECOMMEND', 'RECOMMEND_QUALIFIED']]
    
    if not recommended:
        guideline_md = f"""# Configuration Guidelines
## Condition: {noise_condition}

**Status:** REJECT ALL
No configuration satisfied all constraints and robustness gates.

### Constraints:
- Min Robustness: {constraints.get('min_robustness', 'N/A')}
- Max Latency (ms): {constraints.get('max_latency_ms', 'N/A')}
- Max Memory (MB): {constraints.get('max_memory_mb', 'N/A')}

### Top Rejection Reasons:
"""
        for r in results[:5]:
            guideline_md += f"- **{r.model_id} ({r.quantization} on {r.hardware})**: {', '.join(r.rejection_reasons)}\n"
            
    else:
        top = recommended[0]
        guideline = ConfigGuideline(
            noise_condition=noise_condition,
            constraints=constraints,
            recommended_model=top.model_id,
            recommended_quantization=top.quantization,
            recommended_hardware=top.hardware,
            evidence={
                'robustness_score': top.robustness_score,
                'latency_ms': [r for r in results if r == top][0].robustness_score, # Mock, need actual latency passing
                'memory_mb': 0.0, 
                'sobol_stability': top.sobol_stability
            }
        )
        
        guideline_md = f"""# Configuration Guidelines
## Condition: {noise_condition}

**Status:** {top.status}

### Recommended Configuration:
- **Model**: {top.model_id}
- **Quantization**: {top.quantization}
- **Hardware**: {top.hardware}

### Evidence:
- **Robustness Score**: {top.robustness_score:.4f}
- **Sobol Stability**: {top.sobol_stability*100:.1f}%

### Applied Constraints:
- Min Robustness: {constraints.get('min_robustness', 'N/A')}
- Max Latency (ms): {constraints.get('max_latency_ms', 'N/A')}
- Max Memory (MB): {constraints.get('max_memory_mb', 'N/A')}

### Alternative Options:
"""
        for r in recommended[1:4]:
            guideline_md += f"- **{r.model_id} ({r.quantization} on {r.hardware})**: Score {r.robustness_score:.4f}, Status {r.status}\n"

    out_path = os.path.join(output_dir, 'config_guidelines.md')
    with open(out_path, 'w') as f:
        f.write(guideline_md)
        
    return out_path
