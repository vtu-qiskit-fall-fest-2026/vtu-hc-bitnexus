"""
classical_optimizer.py - Exact Classical Optimization Baseline.
Backwards-compatible wrapper around optimizers.solve_classical_allocation.
"""

from typing import List, Optional, Tuple, Any
from optimizers import solve_classical_allocation

def solve_classically(
    scores: List[float],
    packages: int,
    accessible_flags: Optional[List[bool]] = None,
    previous_plan: Optional[List[int]] = None,
    churn_penalty: float = 0.0,
) -> Tuple[List[int], float]:
    """
    Exact baseline for disaster relief allocation.
    Returns (selected_indices, total_score).
    """
    selected, total_score, _ = solve_classical_allocation(
        scores=scores,
        packages=packages,
        accessible_flags=accessible_flags,
        previous_plan=previous_plan,
        churn_penalty=churn_penalty,
    )
    return selected, total_score
