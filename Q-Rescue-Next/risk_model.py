"""
risk_model.py - Explainable Disaster Risk Scoring Module for Q-Rescue Next.
Computes multi-factor, transparent disaster risk scores and provides mathematical explanations.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    "severity": 0.30,
    "medical_urgency": 0.30,
    "vulnerability": 0.15,
    "population": 0.15,
    "access_difficulty": 0.10,
}

def calculate_factor_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes normalized 0-100 sub-scores for each risk factor.
    """
    comp = pd.DataFrame(index=df.index)
    
    # Population factor: normalized relative to max or minimum floor
    max_pop = max(float(df["Population"].max()), 1.0)
    comp["pop_factor"] = (df["Population"].astype(float) / max_pop) * 100.0
    
    # 1-5 scale factors mapped linearly to 0-100
    comp["severity_factor"] = (df["Severity (1-5)"].astype(float) - 1.0) / 4.0 * 100.0
    comp["medical_factor"] = (df["Medical urgency (1-5)"].astype(float) - 1.0) / 4.0 * 100.0
    
    if "Access difficulty (1-5)" in df.columns:
        comp["access_factor"] = (df["Access difficulty (1-5)"].astype(float) - 1.0) / 4.0 * 100.0
    elif "Distance (km)" in df.columns:
        max_dist = max(float(df["Distance (km)"].max()), 1.0)
        comp["access_factor"] = (df["Distance (km)"].astype(float) / max_dist) * 100.0
    else:
        comp["access_factor"] = 50.0

    if "Vulnerability (1-5)" in df.columns:
        comp["vulnerability_factor"] = (df["Vulnerability (1-5)"].astype(float) - 1.0) / 4.0 * 100.0
    else:
        comp["vulnerability_factor"] = 50.0
    
    return comp

def calculate_risk_scores(
    data: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Calculate explainable risk score (0-100) using weighted linear combination.
    Maintains full backward compatibility while adding explainability breakdown.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    w_sum = sum(weights.values())
    w = {k: v / w_sum for k, v in weights.items()}  # Ensure normalized weights

    res = data.copy()
    
    # Ensure Road Accessible column exists
    if "Road Accessible" not in res.columns:
        res["Road Accessible"] = True
    else:
        res["Road Accessible"] = res["Road Accessible"].astype(bool)

    comp = calculate_factor_components(res)

    raw_score = (
        w["severity"] * comp["severity_factor"]
        + w["medical_urgency"] * comp["medical_factor"]
        + w["vulnerability"] * comp["vulnerability_factor"]
        + w["population"] * comp["pop_factor"]
        + w["access_difficulty"] * comp["access_factor"]
    )
    
    res["Risk score (0-100)"] = raw_score.round(1)
    
    # Categorize Priority Tier
    def assign_tier(score: float, accessible: bool) -> str:
        if not accessible:
            return "Cut-Off (Inaccessible)"
        if score >= 75.0:
            return "Critical"
        if score >= 50.0:
            return "High"
        if score >= 30.0:
            return "Moderate"
        return "Low"

    res["Priority Tier"] = [
        assign_tier(s, acc) for s, acc in zip(res["Risk score (0-100)"], res["Road Accessible"])
    ]
    
    # Add factor contribution columns for transparency and visualization
    res["Severity Contrib"] = (w["severity"] * comp["severity_factor"]).round(1)
    res["Medical Contrib"] = (w["medical_urgency"] * comp["medical_factor"]).round(1)
    res["Vulnerability Contrib"] = (w["vulnerability"] * comp["vulnerability_factor"]).round(1)
    res["Population Contrib"] = (w["population"] * comp["pop_factor"]).round(1)
    res["Access Contrib"] = (w["access_difficulty"] * comp["access_factor"]).round(1)

    return res

def add_risk_scores(data: pd.DataFrame) -> pd.DataFrame:
    """
    Direct backward-compatible wrapper for existing code.
    """
    return calculate_risk_scores(data, DEFAULT_WEIGHTS)

def explain_area_risk(row: pd.Series, max_pop: float = 10000.0) -> str:
    """
    Generates a natural-language explainability summary for a specific area.
    """
    area = row.get("Area", "Unknown Area")
    score = row.get("Risk score (0-100)", 0.0)
    sev = row.get("Severity (1-5)", 1)
    med = row.get("Medical urgency (1-5)", 1)
    pop = row.get("Population", 0)
    vuln = row.get("Vulnerability (1-5)", 1)
    acc = row.get("Access difficulty (1-5)", 1)
    accessible = row.get("Road Accessible", True)

    status_str = "Road connection intact" if accessible else "⚠️ ROAD ACCESS SEVERED (Inaccessible)"

    return (
        f"**{area}** receives a Disaster Risk Score of **{score}/100** ({row.get('Priority Tier', 'Standard')}).\n"
        f"- **Primary Drivers:** Disaster Severity = {sev}/5 (30% weight) and Medical Urgency = {med}/5 (30% weight).\n"
        f"- **Demographics:** Population = {pop:,} (relative scale) with Vulnerability = {vuln}/5.\n"
        f"- **Logistics Status:** {status_str} with terrain difficulty = {acc}/5."
    )
