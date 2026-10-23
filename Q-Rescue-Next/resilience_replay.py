"""
resilience_replay.py - Resilience Replay Engine for Q-Rescue Next.
Applies sequential disruptions, validates plan feasibility, re-optimizes with
plan stability and accessibility constraints, and tracks the full recovery timeline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from risk_model import calculate_risk_scores
from optimizers import solve_classical_allocation, solve_quantum_qaoa_allocation
from disruptions import DisruptionEvent
from analytics import calculate_unmet_needs, compute_plan_stability

@dataclass
class ReplayStepState:
    step_index: int
    step_label: str
    event: Optional[DisruptionEvent]
    data_df: pd.DataFrame
    available_packages: int
    prior_plan_vector: Optional[List[int]]
    
    # Feasibility of the prior plan under new conditions
    prior_plan_feasible: bool
    feasibility_violations: List[str]
    
    # New Adaptive Plan
    allocated_indices: List[int]
    allocated_names: List[str]
    plan_vector: List[int]
    solver_details: Dict[str, Any]
    
    # Quantum QAOA (optional / comparison)
    qaoa_allocated_indices: Optional[List[int]] = None
    qaoa_allocated_names: Optional[List[str]] = None
    qaoa_plan_vector: Optional[List[int]] = None
    qaoa_details: Optional[Dict[str, Any]] = None
    
    # Analytics
    stability_metrics: Dict[str, Any] = field(default_factory=dict)
    unmet_needs: Dict[str, Any] = field(default_factory=dict)
    operational_briefing: str = ""


class ResilienceReplayEngine:
    """
    Simulates disaster timeline evolution under sequential disruptions.
    """

    def __init__(
        self,
        initial_df: pd.DataFrame,
        initial_packages: int,
        disruptions: List[DisruptionEvent],
        churn_penalty: float = 15.0,
        run_qaoa: bool = False,
        qaoa_seed: int = 1234,
    ):
        self.initial_df = initial_df.copy()
        self.initial_packages = int(initial_packages)
        self.disruptions = disruptions
        self.churn_penalty = float(churn_penalty)
        self.run_qaoa = run_qaoa
        self.qaoa_seed = qaoa_seed
        self.timeline: List[ReplayStepState] = []

    def check_plan_feasibility(
        self,
        prior_plan: Optional[List[int]],
        current_df: pd.DataFrame,
        current_packages: int,
    ) -> Tuple[bool, List[str]]:
        """
        Tests whether the prior allocation plan satisfies current operational realities:
        1. Does it assign a package to a newly cut-off / inaccessible area?
        2. Does it require more packages than are currently available in the warehouse?
        """
        if prior_plan is None:
            return True, []

        violations = []
        n = len(current_df)
        p_len = len(prior_plan)

        # Pad or match length
        active_prior = [prior_plan[i] if i < p_len else 0 for i in range(n)]

        # Check inaccessibility
        for i, row in current_df.iterrows():
            area_name = row["Area"]
            is_accessible = bool(row.get("Road Accessible", True))
            if not is_accessible and i < len(active_prior) and active_prior[i] == 1:
                violations.append(
                    f"CRITICAL CONSTRAINT VIOLATION: Road to '{area_name}' is cut off, "
                    f"making previously committed package delivery physically impossible."
                )

        # Check supply capacity
        allocated_count = sum(active_prior)
        if allocated_count > current_packages:
            violations.append(
                f"RESOURCE DEFICIT VIOLATION: Prior plan committed {allocated_count} packages, "
                f"but available warehouse stock has dropped to {current_packages}."
            )

        is_feasible = len(violations) == 0
        return is_feasible, violations

    def run_replay(self) -> List[ReplayStepState]:
        """
        Executes the entire replay timeline from step 0 to the final disruption.
        """
        self.timeline = []
        current_df = self.initial_df.copy()
        current_packages = self.initial_packages
        prior_plan: Optional[List[int]] = None

        # --- STEP 0: Initial Baseline ---
        scored_df_0 = calculate_risk_scores(current_df)
        scores_0 = scored_df_0["Risk score (0-100)"].tolist()
        acc_flags_0 = scored_df_0["Road Accessible"].tolist()

        c_sel_0, c_risk_0, c_det_0 = solve_classical_allocation(
            scores=scores_0,
            packages=current_packages,
            accessible_flags=acc_flags_0,
            previous_plan=None,
            churn_penalty=0.0,
        )
        plan_vector_0 = [1 if i in c_sel_0 else 0 for i in range(len(scored_df_0))]
        sel_names_0 = scored_df_0.iloc[c_sel_0]["Area"].tolist() if c_sel_0 else []

        q_sel_0, q_names_0, q_vec_0, q_det_0 = None, None, None, None
        if self.run_qaoa:
            try:
                q_s0, _, q_det0 = solve_quantum_qaoa_allocation(
                    scores=scores_0,
                    packages=current_packages,
                    accessible_flags=acc_flags_0,
                    previous_plan=None,
                    churn_penalty=0.0,
                    seed=self.qaoa_seed,
                )
                q_sel_0 = q_s0
                q_names_0 = scored_df_0.iloc[q_s0]["Area"].tolist() if q_s0 else []
                q_vec_0 = [1 if i in q_s0 else 0 for i in range(len(scored_df_0))]
                q_det_0 = q_det0
            except Exception as e:
                q_det_0 = {"error": str(e)}

        stab_0 = compute_plan_stability(None, plan_vector_0, scored_df_0["Area"].tolist(), current_packages)
        unmet_0 = calculate_unmet_needs(scored_df_0, c_sel_0)
        briefing_0 = (
            f"**Baseline Relief Plan Activated.** Allocated {len(c_sel_0)} package(s) across "
            f"highest-risk accessible zones: **{', '.join(sel_names_0)}**. Total captured risk: {c_risk_0:.1f} pts. "
            f"All transport routes currently reported intact."
        )

        step_0 = ReplayStepState(
            step_index=0,
            step_label="T0: Initial Assessment",
            event=None,
            data_df=scored_df_0,
            available_packages=current_packages,
            prior_plan_vector=None,
            prior_plan_feasible=True,
            feasibility_violations=[],
            allocated_indices=c_sel_0,
            allocated_names=sel_names_0,
            plan_vector=plan_vector_0,
            solver_details=c_det_0,
            qaoa_allocated_indices=q_sel_0,
            qaoa_allocated_names=q_names_0,
            qaoa_plan_vector=q_vec_0,
            qaoa_details=q_det_0,
            stability_metrics=stab_0,
            unmet_needs=unmet_0,
            operational_briefing=briefing_0,
        )
        self.timeline.append(step_0)
        prior_plan = plan_vector_0

        # --- SEQUENTIAL DISRUPTIONS: STEP 1 TO M ---
        for step_idx, event in enumerate(self.disruptions, start=1):
            # 1. Apply Disruption to world state
            current_df = current_df.copy()
            ev_type = event.event_type
            params = event.parameters
            target = event.target_area

            if ev_type == "road_block" and target:
                mask = current_df["Area"] == target
                current_df.loc[mask, "Road Accessible"] = False
                current_df.loc[mask, "Access difficulty (1-5)"] = 5

            elif ev_type == "road_cleared" and target:
                mask = current_df["Area"] == target
                current_df.loc[mask, "Road Accessible"] = True

            elif ev_type == "medical_surge" and target:
                mask = current_df["Area"] == target
                if "new_medical_level" in params:
                    current_df.loc[mask, "Medical urgency (1-5)"] = params["new_medical_level"]
                else:
                    delta = params.get("delta_medical", 1)
                    current_df.loc[mask, "Medical urgency (1-5)"] = (
                        current_df.loc[mask, "Medical urgency (1-5)"] + delta
                    ).clip(1, 5)

            elif ev_type == "severity_spike" and target:
                mask = current_df["Area"] == target
                delta_s = params.get("delta_severity", 1)
                current_df.loc[mask, "Severity (1-5)"] = (
                    current_df.loc[mask, "Severity (1-5)"] + delta_s
                ).clip(1, 5)
                if "delta_medical" in params:
                    current_df.loc[mask, "Medical urgency (1-5)"] = (
                        current_df.loc[mask, "Medical urgency (1-5)"] + params["delta_medical"]
                    ).clip(1, 5)

            elif ev_type == "supply_loss":
                lost = params.get("lost_packages", 1)
                current_packages = max(0, current_packages - lost)

            elif ev_type == "supply_gain":
                gained = params.get("gained_packages", 1)
                current_packages = current_packages + gained

            elif ev_type == "emergent_area":
                new_data = params.get("new_area_data")
                if new_data and new_data["Area"] not in current_df["Area"].values:
                    current_df = pd.concat([current_df, pd.DataFrame([new_data])], ignore_index=True)
                    # Expand prior plan by 0 for new area
                    if prior_plan is not None:
                        prior_plan = prior_plan + [0]

            # 2. Recalculate Risk Scores
            scored_df = calculate_risk_scores(current_df)
            scores = scored_df["Risk score (0-100)"].tolist()
            acc_flags = scored_df["Road Accessible"].tolist()

            # 3. Check Feasibility of Prior Plan under new reality
            is_feasible, violations = self.check_plan_feasibility(
                prior_plan=prior_plan,
                current_df=scored_df,
                current_packages=current_packages,
            )

            # 4. Adaptive Recovery Optimization (with Churn Penalty)
            c_sel, c_risk, c_det = solve_classical_allocation(
                scores=scores,
                packages=current_packages,
                accessible_flags=acc_flags,
                previous_plan=prior_plan,
                churn_penalty=self.churn_penalty,
            )
            curr_plan_vector = [1 if i in c_sel else 0 for i in range(len(scored_df))]
            sel_names = scored_df.iloc[c_sel]["Area"].tolist() if c_sel else []

            # Optional QAOA solve
            q_sel, q_names, q_vec, q_det = None, None, None, None
            if self.run_qaoa:
                try:
                    q_s, _, q_d = solve_quantum_qaoa_allocation(
                        scores=scores,
                        packages=current_packages,
                        accessible_flags=acc_flags,
                        previous_plan=prior_plan,
                        churn_penalty=self.churn_penalty,
                        seed=self.qaoa_seed,
                    )
                    q_sel = q_s
                    q_names = scored_df.iloc[q_s]["Area"].tolist() if q_s else []
                    q_vec = [1 if i in q_s else 0 for i in range(len(scored_df))]
                    q_det = q_d
                except Exception as e:
                    q_det = {"error": str(e)}

            # 5. Stability & Unmet Needs Analytics
            stability = compute_plan_stability(
                plan_prev=prior_plan,
                plan_curr=curr_plan_vector,
                area_names=scored_df["Area"].tolist(),
                max_k=max(self.initial_packages, current_packages, 1),
            )
            unmet = calculate_unmet_needs(scored_df, c_sel)

            # 6. Operational Incident Commander Briefing
            briefing = self._generate_step_briefing(
                event=event,
                is_feasible=is_feasible,
                violations=violations,
                stability=stability,
                sel_names=sel_names,
                c_risk=c_risk,
                unmet=unmet,
            )

            step_state = ReplayStepState(
                step_index=step_idx,
                step_label=f"T{step_idx}: {event.icon} {event.title}",
                event=event,
                data_df=scored_df,
                available_packages=current_packages,
                prior_plan_vector=prior_plan,
                prior_plan_feasible=is_feasible,
                feasibility_violations=violations,
                allocated_indices=c_sel,
                allocated_names=sel_names,
                plan_vector=curr_plan_vector,
                solver_details=c_det,
                qaoa_allocated_indices=q_sel,
                qaoa_allocated_names=q_names,
                qaoa_plan_vector=q_vec,
                qaoa_details=q_det,
                stability_metrics=stability,
                unmet_needs=unmet,
                operational_briefing=briefing,
            )
            self.timeline.append(step_state)
            prior_plan = curr_plan_vector

        return self.timeline

    def _generate_step_briefing(
        self,
        event: DisruptionEvent,
        is_feasible: bool,
        violations: List[str],
        stability: Dict[str, Any],
        sel_names: List[str],
        c_risk: float,
        unmet: Dict[str, Any],
    ) -> str:
        """
        Synthesizes human-readable operational reasoning for disaster commanders.
        """
        lines = []
        lines.append(f"**Disruption Event ({event.timestamp_label}):** {event.title}")
        lines.append(f"*{event.description}*")
        lines.append("")

        if not is_feasible:
            lines.append("🚨 **PREVIOUS PLAN INFEASIBLE:**")
            for v in violations:
                lines.append(f"- {v}")
            lines.append("⚡ **Action Taken:** Adaptive Recovery Engine automatically reconfigured allocation.")
        else:
            lines.append("✅ **Plan Feasibility Intact:** Previous plan remained physically viable under new conditions.")

        lines.append("")
        if stability["churn_count"] > 0:
            lines.append(
                f"🔄 **Logistical Reallocation:** "
                f"Packages rerouted from **{', '.join(stability['dropped_areas']) or 'None'}** "
                f"to **{', '.join(stability['added_areas']) or 'None'}**."
            )
            lines.append(f"- Operational Plan Continuity: **{stability['stability_pct']}%**.")
        else:
            lines.append("🔒 **Plan Continuity Preserved:** 0 truck reroutes necessary.")

        lines.append("")
        lines.append(
            f"📋 **Active Relief Distribution:** **{', '.join(sel_names) if sel_names else 'No packages allocated'}** "
            f"capturing **{c_risk:.1f}** risk points ({unmet['accessible_coverage_pct']}% of accessible demand)."
        )

        if unmet["inaccessible_areas_count"] > 0:
            lines.append(
                f"⚠️ **Tactical Alert:** {unmet['inaccessible_areas_count']} area(s) "
                f"({', '.join(unmet['inaccessible_names'])}) are cut off from ground routes. "
                f"Recommend immediate rotary-wing (airdrop) or amphibious transport."
            )

        return "\n".join(lines)
