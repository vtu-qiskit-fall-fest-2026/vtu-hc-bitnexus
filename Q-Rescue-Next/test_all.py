"""
test_all.py - Complete Unit & Integration Test Suite for Q-Rescue Next.
"""

import unittest
import pandas as pd
import numpy as np

from risk_model import calculate_risk_scores, explain_area_risk, DEFAULT_WEIGHTS
from optimizers import solve_classical_allocation, solve_quantum_qaoa_allocation
from disruptions import (
    DisruptionEvent,
    get_preset_canonical_scenario,
    get_preset_communities_scenario,
    get_preset_coastal_cyclone_scenario,
)
from resilience_replay import ResilienceReplayEngine
from analytics import calculate_unmet_needs, compute_plan_stability, jains_fairness_index
from charts import (
    plot_risk_factor_breakdown,
    plot_allocation_matrix,
    plot_unmet_need_progression,
    plot_stability_chart,
    plot_radar_profile,
)

class TestQRescueNext(unittest.TestCase):

    def setUp(self):
        self.df_canonical, self.pkgs_canonical, self.disruptions_canonical = get_preset_canonical_scenario()

    def test_risk_scoring(self):
        scored = calculate_risk_scores(self.df_canonical)
        self.assertEqual(len(scored), 3)
        self.assertIn("Risk score (0-100)", scored.columns)
        self.assertIn("Priority Tier", scored.columns)
        # All scores should be bounded within [0, 100]
        for score in scored["Risk score (0-100)"]:
            self.assertTrue(0.0 <= score <= 100.0)
        # Area A has Sev 5, Med 5, Pop 8000 -> Should have highest risk
        self.assertTrue(scored.iloc[0]["Risk score (0-100)"] > scored.iloc[1]["Risk score (0-100)"])

    def test_classical_optimizer_initial(self):
        scores = [85.0, 70.0, 60.0]
        selected, total_risk, details = solve_classical_allocation(scores, packages=2)
        self.assertEqual(selected, [0, 1])
        self.assertAlmostEqual(total_risk, 155.0)
        self.assertEqual(details["effective_packages"], 2)

    def test_classical_optimizer_road_blocked(self):
        # If Area 0 is cut off, even with highest score, Area 1 and 2 must be picked
        scores = [95.0, 70.0, 60.0]
        acc_flags = [False, True, True]
        selected, total_risk, details = solve_classical_allocation(
            scores, packages=2, accessible_flags=acc_flags
        )
        self.assertEqual(selected, [1, 2])
        self.assertAlmostEqual(total_risk, 130.0)

    def test_classical_optimizer_adaptive_stability(self):
        # Test churn penalty:
        # Previous plan had [1, 1, 0] (Areas 0 and 1)
        # Scores: Area 0: 70, Area 1: 70, Area 2: 72
        # Without churn penalty, Area 2 (72) beats Area 0 or 1 (70).
        # With churn penalty = 10, switching costs 20 pts (churn=2), so planner maintains [0, 1]!
        scores = [70.0, 70.0, 72.0]
        prev_plan = [1, 1, 0]
        # Churn penalty = 0 -> switches to [1, 2] or [0, 2]
        sel_reactive, risk_reactive, _ = solve_classical_allocation(
            scores, packages=2, previous_plan=prev_plan, churn_penalty=0.0
        )
        self.assertIn(2, sel_reactive)

        # Churn penalty = 10 -> maintains [0, 1]
        sel_stable, risk_stable, _ = solve_classical_allocation(
            scores, packages=2, previous_plan=prev_plan, churn_penalty=10.0
        )
        self.assertEqual(sel_stable, [0, 1])

    def test_qaoa_optimizer(self):
        scores = [80.0, 70.0, 60.0]
        selected, total_risk, details = solve_quantum_qaoa_allocation(
            scores, packages=2, seed=42
        )
        self.assertEqual(selected, [0, 1])
        self.assertAlmostEqual(total_risk, 150.0)
        self.assertEqual(details["status"], "OptimizationResultStatus.SUCCESS")

    def test_analytics_and_fairness(self):
        scored = calculate_risk_scores(self.df_canonical)
        unmet = calculate_unmet_needs(scored, allocated_indices=[0, 1])
        self.assertEqual(len(unmet["served_areas_indices"]), 2)
        self.assertIn("Area C", unmet["unserved_accessible_names"])
        self.assertTrue(unmet["captured_risk"] > 0)
        self.assertTrue(0.0 <= unmet["jains_fairness_index"] <= 1.0)

    def test_resilience_replay_canonical(self):
        engine = ResilienceReplayEngine(
            self.df_canonical,
            self.pkgs_canonical,
            self.disruptions_canonical,
            churn_penalty=10.0,
            run_qaoa=False,
        )
        timeline = engine.run_replay()
        # Initial + 3 disruptions = 4 steps
        self.assertEqual(len(timeline), 4)

        # Step 0: Initial
        self.assertTrue(timeline[0].prior_plan_feasible)
        self.assertEqual(len(timeline[0].allocated_names), 3)

        # Step 1: Road block on Area B -> Infeasible prior plan
        self.assertFalse(timeline[1].prior_plan_feasible)
        self.assertNotIn("Area B", timeline[1].allocated_names)
        self.assertIn("Area A", timeline[1].allocated_names)
        self.assertIn("Area C", timeline[1].allocated_names)

        # Step 3: Warehouse supply loss (down to 2 packages)
        self.assertEqual(timeline[3].available_packages, 2)
        self.assertEqual(len(timeline[3].allocated_names), 2)

    def test_communities_preset_scenario(self):
        df_comm, pkgs_comm, dis_comm = get_preset_communities_scenario()
        self.assertEqual(len(df_comm), 5)
        self.assertEqual(len(dis_comm), 3)
        scored = calculate_risk_scores(df_comm)
        self.assertEqual(len(scored), 5)
        engine = ResilienceReplayEngine(scored, pkgs_comm, dis_comm, churn_penalty=15.0, run_qaoa=False)
        timeline = engine.run_replay()
        self.assertEqual(len(timeline), 4)
        # Disruption 1 cuts off Eastbank -> previous plan violated
        self.assertFalse(timeline[1].prior_plan_feasible)

    def test_charts_generation(self):
        scored = calculate_risk_scores(self.df_canonical)
        engine = ResilienceReplayEngine(
            self.df_canonical,
            self.pkgs_canonical,
            self.disruptions_canonical,
            churn_penalty=10.0,
            run_qaoa=False,
        )
        timeline = engine.run_replay()

        fig1 = plot_risk_factor_breakdown(scored)
        self.assertIsNotNone(fig1)

        fig2 = plot_allocation_matrix(timeline)
        self.assertIsNotNone(fig2)

        fig3 = plot_unmet_need_progression(timeline)
        self.assertIsNotNone(fig3)

        fig4 = plot_stability_chart(timeline)
        self.assertIsNotNone(fig4)

        fig5 = plot_radar_profile(scored.iloc[0])
        self.assertIsNotNone(fig5)

if __name__ == "__main__":
    unittest.main()
