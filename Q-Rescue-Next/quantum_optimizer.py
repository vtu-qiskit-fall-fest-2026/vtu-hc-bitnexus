"""
quantum_optimizer.py - Qiskit QAOA Quantum Optimization Solver.
Backwards-compatible wrapper around optimizers.solve_quantum_qaoa_allocation.
"""

from typing import Dict, List, Optional, Tuple, Any
from optimizers import solve_quantum_qaoa_allocation

def solve_with_qaoa(
    scores: List[float],
    packages: int,
    seed: int = 1234,
    accessible_flags: Optional[List[bool]] = None,
    previous_plan: Optional[List[int]] = None,
    churn_penalty: float = 0.0,
) -> Tuple[List[int], float, Dict[str, Any]]:
    """
    Qiskit QAOA solver for relief allocation with road constraints
    and adaptive recovery planning.
    Returns (selected_indices, total_score, detail_dict).
    """
    return solve_quantum_qaoa_allocation(
        scores=scores,
        packages=packages,
        accessible_flags=accessible_flags,
        previous_plan=previous_plan,
        churn_penalty=churn_penalty,
        seed=seed,
    )
