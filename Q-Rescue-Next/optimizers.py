"""
optimizers.py - Optimization Solvers for Q-Rescue Next.
Includes exact classical baseline and Qiskit QAOA quantum optimizer,
with full support for road accessibility constraints and minimum-change recovery planning.
"""

from itertools import combinations
from typing import Dict, List, Optional, Tuple, Any
import warnings
import numpy as np
import pandas as pd

try:
    from scipy.sparse import SparseEfficiencyWarning
    warnings.simplefilter("ignore", SparseEfficiencyWarning)
except ImportError:
    pass

def solve_classical_allocation(
    scores: List[float],
    packages: int,
    accessible_flags: Optional[List[bool]] = None,
    previous_plan: Optional[List[int]] = None,
    churn_penalty: float = 0.0,
) -> Tuple[List[int], float, Dict[str, Any]]:
    """
    Exact classical optimization for resource allocation.
    Solves:
      max sum_{i in selected} scores[i] - churn_penalty * sum_i |x_i - x_i^{prev}|
      s.t. sum(x_i) == min(packages, count(accessible))
           x_i == 0 if not accessible_flags[i]
    """
    n = len(scores)
    if accessible_flags is None:
        accessible_flags = [True] * n
    else:
        accessible_flags = [bool(a) for a in accessible_flags]

    accessible_indices = [i for i, acc in enumerate(accessible_flags) if acc]
    k_eff = min(max(int(packages), 0), len(accessible_indices))

    if k_eff == 0:
        return [], 0.0, {
            "method": "classical_exact",
            "effective_packages": 0,
            "captured_risk": 0.0,
            "churn_count": sum(previous_plan) if previous_plan else 0,
            "churn_penalty_cost": (churn_penalty * sum(previous_plan)) if previous_plan else 0.0,
            "net_objective": 0.0,
            "evaluated_combinations": 1,
        }

    best_selection: Optional[Tuple[int, ...]] = None
    best_net_score = float("-inf")
    best_risk_sum = 0.0
    best_churn = 0

    evaluated_count = 0
    for combo in combinations(accessible_indices, k_eff):
        evaluated_count += 1
        current_x = [1 if i in combo else 0 for i in range(n)]
        risk_sum = sum(float(scores[i]) for i in combo)
        
        churn_count = 0
        if previous_plan is not None:
            churn_count = sum(abs(current_x[i] - int(previous_plan[i])) for i in range(n))
        
        net_score = risk_sum - (churn_penalty * churn_count)

        if net_score > best_net_score:
            best_net_score = net_score
            best_selection = combo
            best_risk_sum = risk_sum
            best_churn = churn_count

    selected_list = sorted(list(best_selection)) if best_selection else []
    details = {
        "method": "classical_exact",
        "effective_packages": k_eff,
        "captured_risk": round(best_risk_sum, 2),
        "churn_count": best_churn,
        "churn_penalty_cost": round(churn_penalty * best_churn, 2),
        "net_objective": round(best_net_score, 2),
        "evaluated_combinations": evaluated_count,
    }
    return selected_list, round(best_risk_sum, 2), details


def solve_quantum_qaoa_allocation(
    scores: List[float],
    packages: int,
    accessible_flags: Optional[List[bool]] = None,
    previous_plan: Optional[List[int]] = None,
    churn_penalty: float = 0.0,
    seed: int = 1234,
    reps: int = 1,
    maxiter: int = 40,
) -> Tuple[List[int], float, Dict[str, Any]]:
    """
    Qiskit QAOA solver for disaster relief allocation with road constraints
    and adaptive plan-stability penalty.
    
    Converts the problem into a QuadraticProgram on accessible variables:
      max sum_{j in Acc} (score_j - churn_penalty * (1 - 2 * prev_j)) * x_j
      s.t. sum(x_j) == k_eff
    """
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms.minimum_eigensolvers import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    n = len(scores)
    if accessible_flags is None:
        accessible_flags = [True] * n
    else:
        accessible_flags = [bool(a) for a in accessible_flags]

    accessible_indices = [i for i, acc in enumerate(accessible_flags) if acc]
    k_eff = min(max(int(packages), 0), len(accessible_indices))

    if k_eff == 0:
        return [], 0.0, {
            "status": "SUCCESS",
            "effective_packages": 0,
            "captured_risk": 0.0,
            "churn_count": sum(previous_plan) if previous_plan else 0,
            "qubits_used": 0,
            "objective_value": 0.0,
        }

    # Map accessible indices to compact qubit decision variables y0, y1, ...
    num_qubits = len(accessible_indices)
    problem = QuadraticProgram("q_rescue_adaptive_qaoa")
    for j in range(num_qubits):
        problem.binary_var(name=f"y{j}")

    linear_obj = {}
    constant_offset = 0.0

    for j, orig_idx in enumerate(accessible_indices):
        risk = float(scores[orig_idx])
        if previous_plan is not None and churn_penalty > 0.0:
            prev_val = int(previous_plan[orig_idx])
            # |y - prev| = (1 - 2*prev)*y + prev
            # penalty = churn_penalty * |y - prev|
            # objective = risk * y - churn_penalty * ((1 - 2*prev)*y + prev)
            #           = (risk - churn_penalty*(1 - 2*prev)) * y - churn_penalty*prev
            coeff = risk - churn_penalty * (1.0 - 2.0 * prev_val)
            constant_offset -= churn_penalty * prev_val
        else:
            coeff = risk
        linear_obj[f"y{j}"] = coeff

    # Also account for churn in inaccessible areas that previously had a package
    if previous_plan is not None and churn_penalty > 0.0:
        for idx in range(n):
            if not accessible_flags[idx] and previous_plan[idx] == 1:
                # previous was 1, current is forced 0 -> churn of 1
                constant_offset -= churn_penalty * 1.0

    problem.maximize(linear=linear_obj, constant=constant_offset)
    problem.linear_constraint(
        linear={f"y{j}": 1 for j in range(num_qubits)},
        sense="==",
        rhs=int(k_eff),
        name="exact_package_constraint",
    )

    sampler = StatevectorSampler(seed=seed)
    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=maxiter),
        reps=reps,
    )
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(problem)

    selected_compact = [j for j, val in enumerate(result.x) if val > 0.5]
    selected_indices = [accessible_indices[j] for j in selected_compact]
    total_risk = sum(float(scores[i]) for i in selected_indices)

    # Compute actual churn
    churn_count = 0
    if previous_plan is not None:
        full_x = [1 if i in selected_indices else 0 for i in range(n)]
        churn_count = sum(abs(full_x[i] - int(previous_plan[i])) for i in range(n))

    detail = {
        "status": str(result.status),
        "objective_value": float(result.fval),
        "captured_risk": round(total_risk, 2),
        "churn_count": churn_count,
        "selected_indices": selected_indices,
        "qubits_used": num_qubits,
        "effective_packages": k_eff,
    }

    if len(selected_indices) != k_eff:
        # Feasibility check warning
        detail["warning"] = f"QAOA selected {len(selected_indices)} packages, expected {k_eff}."

    return selected_indices, round(total_risk, 2), detail
