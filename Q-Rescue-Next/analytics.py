"""
analytics.py - Unmet-Need, Fairness, and Resilience Metrics for Q-Rescue Next.
Computes relief coverage ratios, underserved populations, Jain's fairness index,
and plan stability metrics across disaster steps.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

def jains_fairness_index(values: List[float]) -> float:
    """
    Computes Jain's Fairness Index:
      J = (sum(x_i))^2 / (n * sum(x_i^2))
    Returns float between 0.0 and 1.0 (1.0 = perfectly fair).
    """
    if not values:
        return 1.0
    arr = np.array(values, dtype=float)
    if np.all(arr == 0):
        return 1.0
    n = len(arr)
    numerator = float(np.sum(arr) ** 2)
    denominator = float(n * np.sum(arr ** 2))
    if denominator == 0:
        return 1.0
    return round(float(numerator / denominator), 3)


def calculate_unmet_needs(
    df: pd.DataFrame,
    allocated_indices: List[int]
) -> Dict[str, Any]:
    """
    Detailed unmet need analysis:
    - Analyzes risk captured vs risk left unaddressed
    - Segregates unmet need into accessible shortfall vs cut-off road isolation
    - Computes unserved population and vulnerable individuals at risk
    - Computes fairness across all candidate zones
    """
    n = len(df)
    allocated_set = set(allocated_indices)
    
    total_risk = float(df["Risk score (0-100)"].sum())
    captured_risk = float(df.iloc[list(allocated_set)]["Risk score (0-100)"].sum()) if allocated_set else 0.0
    unmet_risk = max(0.0, total_risk - captured_risk)
    
    # Accessibility breakdown
    accessible_mask = df["Road Accessible"].astype(bool) if "Road Accessible" in df.columns else pd.Series([True]*n)
    
    accessible_df = df[accessible_mask]
    inaccessible_df = df[~accessible_mask]
    
    accessible_risk = float(accessible_df["Risk score (0-100)"].sum())
    inaccessible_risk = float(inaccessible_df["Risk score (0-100)"].sum())
    
    unmet_accessible_risk = max(0.0, accessible_risk - captured_risk)
    
    # Population metrics
    total_pop = int(df["Population"].sum())
    served_pop = int(df.iloc[list(allocated_set)]["Population"].sum()) if allocated_set else 0
    unserved_pop = max(0, total_pop - served_pop)
    
    # Vulnerable population at risk: weighted by vulnerability (1-5)/5
    vulnerability_weights = df["Vulnerability (1-5)"].astype(float) / 5.0
    vulnerable_pop_per_area = df["Population"].astype(float) * vulnerability_weights
    
    unserved_vulnerable_pop = 0.0
    for i in range(n):
        if i not in allocated_set:
            unserved_vulnerable_pop += float(vulnerable_pop_per_area.iloc[i])
            
    # Demand satisfaction ratio
    coverage_pct = round((captured_risk / total_risk * 100.0), 1) if total_risk > 0 else 0.0
    accessible_coverage_pct = round((captured_risk / accessible_risk * 100.0), 1) if accessible_risk > 0 else 0.0
    
    # Fairness index on risk satisfaction:
    # A community with risk R receives 1 package (satisfied = 1.0) or 0 (satisfied = 1.0 - R/100)
    satisfaction_values = []
    for i in range(n):
        r = float(df["Risk score (0-100)"].iloc[i])
        if i in allocated_set:
            satisfaction_values.append(1.0)
        else:
            satisfaction_values.append(max(0.05, 1.0 - (r / 100.0)))
            
    fairness = jains_fairness_index(satisfaction_values)

    return {
        "total_active_areas": n,
        "accessible_areas_count": int(accessible_mask.sum()),
        "inaccessible_areas_count": int((~accessible_mask).sum()),
        "served_areas_indices": sorted(list(allocated_set)),
        "served_areas_names": df.iloc[list(allocated_set)]["Area"].tolist() if allocated_set else [],
        "unserved_accessible_names": [
            df["Area"].iloc[i] for i in range(n) if i not in allocated_set and accessible_mask.iloc[i]
        ],
        "inaccessible_names": [
            df["Area"].iloc[i] for i in range(n) if not accessible_mask.iloc[i]
        ],
        "total_risk_demanded": round(total_risk, 1),
        "accessible_risk_demanded": round(accessible_risk, 1),
        "captured_risk": round(captured_risk, 1),
        "unmet_risk": round(unmet_risk, 1),
        "unmet_accessible_risk": round(unmet_accessible_risk, 1),
        "unmet_inaccessible_risk": round(inaccessible_risk, 1),
        "risk_coverage_pct": coverage_pct,
        "accessible_coverage_pct": accessible_coverage_pct,
        "total_population": total_pop,
        "served_population": served_pop,
        "unserved_population": unserved_pop,
        "unserved_vulnerable_pop": int(round(unserved_vulnerable_pop)),
        "jains_fairness_index": fairness,
    }


def compute_plan_stability(
    plan_prev: Optional[List[int]],
    plan_curr: List[int],
    area_names: List[str],
    max_k: int = 1
) -> Dict[str, Any]:
    """
    Evaluates stability and logistical churn between two consecutive relief plans.
    """
    if plan_prev is None:
        # Initial step has 100% stability
        return {
            "churn_count": 0,
            "stability_pct": 100.0,
            "retained_areas": [area_names[i] for i, val in enumerate(plan_curr) if val == 1],
            "dropped_areas": [],
            "added_areas": [area_names[i] for i, val in enumerate(plan_curr) if val == 1],
            "explanation": "Baseline deployment plan initialized.",
        }

    n = max(len(plan_prev), len(plan_curr))
    p_prev = plan_prev + [0] * (n - len(plan_prev))
    p_curr = plan_curr + [0] * (n - len(plan_curr))

    churn = sum(abs(p_curr[i] - p_prev[i]) for i in range(n))
    retained = [area_names[i] for i in range(n) if p_prev[i] == 1 and p_curr[i] == 1]
    dropped = [area_names[i] for i in range(n) if p_prev[i] == 1 and p_curr[i] == 0]
    added = [area_names[i] for i in range(n) if p_prev[i] == 0 and p_curr[i] == 1]

    denom = 2.0 * max(max_k, 1)
    stability = max(0.0, (1.0 - (churn / denom))) * 100.0

    if churn == 0:
        msg = "Plan remained 100% stable: no relief packages rerouted."
    else:
        msg = (
            f"Plan adapted with {churn} assignment changes ({len(dropped)} package(s) rerouted). "
            f"Maintained {stability:.1f}% logistical continuity."
        )

    return {
        "churn_count": churn,
        "stability_pct": round(stability, 1),
        "retained_areas": retained,
        "dropped_areas": dropped,
        "added_areas": added,
        "explanation": msg,
    }
