from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
from .gates import check_robustness_gates, GateResult

@dataclass
class ConfigurationResult:
    model_id: str
    quantization: str
    hardware: str
    robustness_score: float
    robustness_gate: GateResult
    deployment_feasible: bool
    sobol_stability: float
    status: str  # 'REJECT' | 'RECOMMEND' | 'RECOMMEND_QUALIFIED'
    rejection_reasons: List[str]
    rank: Optional[int] = None

class DecisionSupportEngine:
    """
    Evaluates robustness scores and deployment profiles to produce
    actionable model configuration recommendations based on hardware constraints.
    """
    def __init__(self, 
                 min_robustness: float = 0.50, 
                 max_latency_ms: float = 100.0, 
                 max_memory_mb: float = 512.0,
                 auroc_gate: float = 0.70,
                 ece_gate: float = 0.15):
        self.min_robustness = min_robustness
        self.max_latency_s = max_latency_ms / 1000.0
        self.max_memory_mb = max_memory_mb
        self.auroc_gate = auroc_gate
        self.ece_gate = ece_gate

    def evaluate(self, 
                 robustness_df: pd.DataFrame, 
                 deployment_df: pd.DataFrame, 
                 sobol_results: dict) -> List[ConfigurationResult]:
        """
        Evaluate all configurations across all noise conditions.
        Since deployment profiles are independent of noise, we rank models
        based on their worst-case or average robustness, or per-condition.
        For simplicity, this assumes robustness_df contains aggregated 
        robustness scores (e.g., worst-case across conditions).
        """
        results = []
        
        # Merge robustness and deployment profiles
        # robustness_df is expected to have 'model_id', 'robustness_score', 'auroc', 'ece'
        # deployment_df has 'model_id', 'precision', 'provider', 'latency_p95', 'memory_mb'
        
        # Cross join models with their deployment profiles
        for _, r_row in robustness_df.iterrows():
            model_id = r_row['model_id']
            rob_score = r_row['robustness_score']
            auroc = r_row.get('auroc', None)
            ece = r_row.get('ece', None)
            
            # Check gates
            gate_res = check_robustness_gates(auroc, ece, self.auroc_gate, self.ece_gate)
            
            # Get Sobol stability for this model if available (mocked as 1.0 if missing)
            # Sobol results might provide a confidence score per model
            stability = sobol_results.get(model_id, 1.0)
            
            d_profiles = deployment_df[deployment_df['model_id'] == model_id]
            for _, d_row in d_profiles.iterrows():
                quant = d_row['precision']
                hw = d_row['provider']
                lat = d_row['latency_p95']
                mem = d_row['memory_mb']
                
                reasons = list(gate_res.rejection_reasons)
                
                # Check constraints
                deployment_feasible = True
                if rob_score < self.min_robustness:
                    deployment_feasible = False
                    reasons.append(f"Robustness {rob_score:.4f} < {self.min_robustness}")
                if lat > self.max_latency_s:
                    deployment_feasible = False
                    reasons.append(f"Latency {lat*1000:.1f}ms > {self.max_latency_s*1000}ms")
                if mem > self.max_memory_mb:
                    deployment_feasible = False
                    reasons.append(f"Memory {mem:.1f}MB > {self.max_memory_mb}MB")
                    
                status = 'REJECT'
                if gate_res.passed and deployment_feasible:
                    if stability >= 0.90:
                        status = 'RECOMMEND'
                    else:
                        status = 'RECOMMEND_QUALIFIED'
                        
                results.append(ConfigurationResult(
                    model_id=model_id,
                    quantization=quant,
                    hardware=hw,
                    robustness_score=rob_score,
                    robustness_gate=gate_res,
                    deployment_feasible=deployment_feasible,
                    sobol_stability=stability,
                    status=status,
                    rejection_reasons=reasons
                ))
                
        # Rank the recommendations
        # Sort by status (RECOMMEND first), then robustness_score
        results.sort(key=lambda x: (
            0 if x.status == 'RECOMMEND' else (1 if x.status == 'RECOMMEND_QUALIFIED' else 2),
            -x.robustness_score
        ))
        
        # Assign rank to feasible ones
        rank = 1
        for res in results:
            if res.status in ['RECOMMEND', 'RECOMMEND_QUALIFIED']:
                res.rank = rank
                rank += 1
                
        return results
