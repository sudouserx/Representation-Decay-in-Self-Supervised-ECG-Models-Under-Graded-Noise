from dataclasses import dataclass
from typing import List, Optional

@dataclass
class GateResult:
    passed: bool
    auroc_passed: bool
    ece_passed: bool
    rejection_reasons: List[str]

def check_robustness_gates(
    auroc: Optional[float],
    ece: Optional[float],
    auroc_gate: float = 0.70,
    ece_gate: float = 0.15
) -> GateResult:
    """
    Check non-compensatory robustness gates.
    
    Parameters
    ----------
    auroc : float, optional
        Raw AUROC value.
    ece : float, optional
        Raw ECE value.
    auroc_gate : float
        Minimum acceptable AUROC.
    ece_gate : float
        Maximum acceptable ECE.
        
    Returns
    -------
    GateResult
        Contains overall passed status, individual gate statuses, and reasons.
    """
    auroc_passed = True
    ece_passed = True
    reasons = []
    
    if auroc is not None and auroc < auroc_gate:
        auroc_passed = False
        reasons.append(f"AUROC {auroc:.4f} < {auroc_gate}")
        
    if ece is not None and ece > ece_gate:
        ece_passed = False
        reasons.append(f"ECE {ece:.4f} > {ece_gate}")
        
    passed = auroc_passed and ece_passed
    
    return GateResult(
        passed=passed,
        auroc_passed=auroc_passed,
        ece_passed=ece_passed,
        rejection_reasons=reasons
    )
