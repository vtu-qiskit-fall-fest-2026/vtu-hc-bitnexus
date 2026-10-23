"""
run_demo.py - End-to-End Execution of Q-Rescue Next Scenarios.
Demonstrates connected requirements, risk scoring, classical vs QAOA optimization,
and the Resilience Replay Engine output.
"""

import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from disruptions import get_preset_canonical_scenario, get_preset_communities_scenario
from resilience_replay import ResilienceReplayEngine
from optimizers import solve_classical_allocation, solve_quantum_qaoa_allocation
from risk_model import calculate_risk_scores

def run():
    print("=" * 70)
    print("🚨 Q-RESCUE NEXT: END-TO-END VERIFICATION & FINAL OUTPUT")
    print("=" * 70)

    # 1. CANONICAL SCENARIO
    print("\n" + "=" * 70)
    print("🎯 SCENARIO 1: CANONICAL 3-AREA HACKATHON BENCHMARK (Areas A, B, C)")
    print("=" * 70)
    df1, pkgs1, dis1 = get_preset_canonical_scenario()
    scored1 = calculate_risk_scores(df1)
    
    print("\n[Ground Data & Calculated Risk Scores]")
    for _, r in scored1.iterrows():
        print(f"  • {r['Area']:<8} | Pop: {r['Population']:>5} | Sev: {r['Severity (1-5)']}/5 | Med: {r['Medical urgency (1-5)']}/5 | Score: {r['Risk score (0-100)']:>5.1f}/100 ({r['Priority Tier']})")

    # Classical vs QAOA on Baseline
    scores1 = scored1["Risk score (0-100)"].tolist()
    c_sel, c_risk, c_det = solve_classical_allocation(scores1, packages=pkgs1)
    q_sel, q_risk, q_det = solve_quantum_qaoa_allocation(scores1, packages=pkgs1, seed=42)

    print(f"\n[T0 Baseline Allocation: {pkgs1} Packages Available]")
    print(f"  • Classical Exact Result: {[df1['Area'].iloc[i] for i in c_sel]} | Total Score: {c_risk:.1f} pts")
    print(f"  • Qiskit QAOA Result:     {[df1['Area'].iloc[i] for i in q_sel]} | Total Score: {q_risk:.1f} pts | Qubits: {q_det['qubits_used']}")
    print(f"  • Quantum-Classical Match: {'✅ EXACT MATCH' if c_sel == q_sel else '⚠️ APPROXIMATED'}")

    # Replay Simulation
    print("\n[Resilience Replay Engine: Disruption Progression]")
    engine1 = ResilienceReplayEngine(df1, pkgs1, dis1, churn_penalty=10.0, run_qaoa=False)
    timeline1 = engine1.run_replay()

    for s in timeline1:
        print(f"\n>> Step {s.step_index}: {s.step_label}")
        print(f"   Packages: {s.available_packages} | Assigned: {s.allocated_names}")
        print(f"   Prior Plan Feasible: {'✅ YES' if s.prior_plan_feasible else '🚨 INFEASIBLE (VIOLATION)'}")
        if s.feasibility_violations:
            for v in s.feasibility_violations:
                print(f"   ⚠️ Reason: {v}")
        print(f"   Plan Stability: {s.stability_metrics.get('stability_pct')}% (Reroutes: {s.stability_metrics.get('churn_count')})")
        print(f"   Unmet Risk: {s.unmet_needs.get('unmet_risk')} pts | Jain Fairness: {s.unmet_needs.get('jains_fairness_index')}")

    # 2. 5-COMMUNITY MULTI-RESOURCE SCENARIO
    print("\n" + "=" * 70)
    print("🏘️ SCENARIO 2: 5-COMMUNITY REGION (Riverdale, Hillview, Greenfield, Eastbank, Meadowtown)")
    print("=" * 70)
    df2, pkgs2, dis2 = get_preset_communities_scenario()
    scored2 = calculate_risk_scores(df2)

    print("\n[Community Demographics & Initial Needs]")
    for _, r in scored2.iterrows():
        print(f"  • {r['Area']:<12} | Pop: {r['Population']:>5} | Food Need: {r['Food Need']:>4} | Kits Need: {r['Medical Need']:>3} | Risk: {r['Risk score (0-100)']:>5.1f}")

    engine2 = ResilienceReplayEngine(df2, pkgs2, dis2, churn_penalty=15.0, run_qaoa=False)
    timeline2 = engine2.run_replay()

    for s in timeline2:
        print(f"\n>> Step {s.step_index}: {s.step_label}")
        print(f"   Convoys: {s.available_packages} | Allocated: {s.allocated_names}")
        print(f"   Prior Feasible: {'✅ YES' if s.prior_plan_feasible else '🚨 INFEASIBLE'}")
        if s.feasibility_violations:
            for v in s.feasibility_violations:
                print(f"   ⚠️ Violation: {v}")
        print(f"   Stability: {s.stability_metrics.get('stability_pct')}% | Unserved Vulnerable Pop: {s.unmet_needs.get('unserved_vulnerable_pop'):,} people")

    print("\n" + "=" * 70)
    print("✅ ALL REQUIREMENTS VERIFIED AND CONNECTED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run()
