"""
app.py - Q-Rescue Next: A Self-Adapting Quantum-Assisted Disaster Relief Planner.
Interactive Streamlit Dashboard combining Explainable Risk Assessment,
Classical & Qiskit QAOA Quantum Optimization, Multi-Resource Allocation (Food & Kits),
and the Resilience Replay Engine. Fully mobile-responsive and non-redundant.
"""

import os
import sys

# Force UTF-8 environment and console streams before importing Streamlit
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

if sys.platform == "win32":
    try:
        if sys.stdout is not None:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if sys.stderr is not None:
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import copy
import csv
import io
import socket
import streamlit as st
import pandas as pd
import numpy as np

def get_local_ip() -> str:
    """Detects local LAN IPv4 address for mobile access."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

from risk_model import calculate_risk_scores, explain_area_risk, DEFAULT_WEIGHTS
from optimizers import solve_classical_allocation, solve_quantum_qaoa_allocation
from disruptions import (
    DisruptionEvent,
    get_preset_canonical_scenario,
    get_preset_communities_scenario,
    get_preset_coastal_cyclone_scenario,
    EVENT_TYPE_ICONS,
)
from resilience_replay import ResilienceReplayEngine, ReplayStepState
from analytics import calculate_unmet_needs, compute_plan_stability, jains_fairness_index
from charts import (
    plot_risk_factor_breakdown,
    plot_allocation_matrix,
    plot_unmet_need_progression,
    plot_stability_chart,
    plot_radar_profile,
)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Q-Rescue Next | Adaptive Disaster Planner",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Mobile-First Responsive CSS
st.markdown("""
<style>
    /* Desktop default header */
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 1.2rem 1.6rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
        border-left: 6px solid #EF4444;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
        color: #94A3B8;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #3B82F6;
        color: white;
        margin-right: 0.4rem;
        margin-bottom: 0.2rem;
    }
    .badge-quantum { background-color: #8B5CF6; }
    .badge-resilience { background-color: #10B981; }
    .badge-alert { background-color: #EF4444; }
    
    .disruption-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 0.8rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
    }
    .violation-box {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 0.8rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
    }
    .feasible-box {
        background-color: #ECFDF5;
        border-left: 4px solid #10B981;
        padding: 0.8rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
    }

    /* Mobile screens (smartphones < 768px) */
    @media (max-width: 768px) {
        .main-header {
            padding: 0.8rem 1rem !important;
            border-radius: 8px !important;
        }
        .main-header h1 {
            font-size: 1.35rem !important;
        }
        .main-header p {
            font-size: 0.85rem !important;
        }
        .stButton button {
            width: 100% !important;
            min-height: 44px !important;
            font-size: 0.95rem !important;
        }
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        .stPlotlyChart {
            width: 100% !important;
            overflow-x: auto !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown("""
<div class="main-header">
    <div style="display: flex; flex-wrap: wrap; gap: 0.3rem; margin-bottom: 0.4rem;">
        <span class="badge-pill badge-quantum">⚛️ Qiskit QAOA</span>
        <span class="badge-pill badge-resilience">🌟 Resilience Replay</span>
        <span class="badge-pill badge-alert">📱 Mobile Optimized</span>
    </div>
    <h1>🚨 Q-Rescue Next</h1>
    <p>Adaptive Disaster Relief Planner: Risk Assessment, Quantum Optimization & Resilience Replay</p>
</div>
""", unsafe_allow_html=True)

# Helper for multi-resource allocation
def allocate_units(total_supply: int, needs: list, weights: list, accessible_flags: list) -> list:
    allocations = [0] * len(needs)
    remaining = int(total_supply)

    while remaining > 0:
        eligible = [
            i for i in range(len(needs))
            if accessible_flags[i] and allocations[i] < needs[i]
        ]
        if not eligible:
            break

        chosen = max(
            eligible,
            key=lambda i: (
                weights[i] * (needs[i] - allocations[i]),
                needs[i] - allocations[i]
            )
        )
        allocations[chosen] += 1
        remaining -= 1

    return allocations

# --- SIDEBAR CONFIGURATION ---
local_ip = get_local_ip()

with st.sidebar:
    st.header("⚙️ Scenario & Setup")

    scenario_choice = st.selectbox(
        "Disaster Scenario:",
        [
            "🏘️ Riverdale & Surrounding Towns (5 Communities)",
            "🎯 Canonical Hackathon Scenario (Areas A, B, C)",
            "🌊 Multi-Front Coastal Cyclone (4 Communities + Island)",
            "🛠️ Custom Disaster Setup",
        ],
        index=0,
    )

    # Mobile Connectivity Box
    with st.expander("📱 View on Phone / Mobile", expanded=False):
        st.write("Open this URL in your mobile browser:")
        st.code(f"http://{local_ip}:8501", language="text")
        st.caption("• Phone must be on the same Wi-Fi / hotspot as this PC.")
        st.caption("• Start with: `streamlit run app.py --server.address 0.0.0.0`")

    st.markdown("---")
    st.subheader("📦 Supplies Available")

    if "Riverdale" in scenario_choice:
        default_food, default_kits, default_pkgs = 1800, 200, 3
    elif "Canonical" in scenario_choice:
        default_food, default_kits, default_pkgs = 1650, 200, 3
    elif "Cyclone" in scenario_choice:
        default_food, default_kits, default_pkgs = 2650, 295, 3
    else:
        default_food, default_kits, default_pkgs = 1500, 150, 2

    food_supply = st.number_input("Food Supplies (units)", min_value=0, value=default_food, step=100)
    medical_supply = st.number_input("Medical Kits (units)", min_value=0, value=default_kits, step=10)
    available_packages = st.number_input("Priority Convoys ($K$)", min_value=1, max_value=10, value=default_pkgs, step=1)

    st.markdown("---")
    st.subheader("⚖️ Recovery Tuning")
    churn_penalty = st.slider(
        r"Plan Stability Weight ($\lambda_{\text{churn}}$)",
        min_value=0.0,
        max_value=40.0,
        value=15.0,
        step=2.5,
        help="Higher values prevent unnecessary truck reroutes."
    )

    st.markdown("---")
    st.subheader("⚛️ Quantum QAOA")
    run_qaoa_toggle = st.checkbox("Run Qiskit QAOA Solver", value=True)
    qaoa_seed = st.number_input("Quantum Seed", min_value=1, max_value=9999, value=1234, step=1)

    st.caption("Q-Rescue Next | Educational Prototype")

# --- INITIALIZE PRESET DATA ---
if "current_preset" not in st.session_state or st.session_state["current_preset"] != scenario_choice:
    st.session_state["current_preset"] = scenario_choice
    
    if "Riverdale" in scenario_choice:
        base_df, base_k, base_disruptions = get_preset_communities_scenario()
    elif "Canonical" in scenario_choice:
        base_df, base_k, base_disruptions = get_preset_canonical_scenario()
    elif "Cyclone" in scenario_choice:
        base_df, base_k, base_disruptions = get_preset_coastal_cyclone_scenario()
    else:
        base_df, base_k, base_disruptions = get_preset_canonical_scenario()

    st.session_state["communities_df"] = base_df.copy()
    st.session_state["disruptions_list"] = copy.deepcopy(base_disruptions)
    st.session_state["replay_timeline"] = None
    st.session_state["qrescue_results"] = None

active_df = st.session_state["communities_df"]

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 1. Communities & Risk",
    "⚡ 2. Relief Allocation",
    "🌪️ 3. Resilience Replay",
    "📊 4. Timeline & Analytics",
    "🧠 5. Architecture & Tech",
])

# ==============================================================================
# TAB 1: COMMUNITY INFORMATION & RISK ASSESSMENT (NO DUPLICATE CHARTS)
# ==============================================================================
with tab1:
    st.header("1. Community Information & Risk Assessment")
    st.caption("Touch-friendly inputs: update community demands, hazard levels, or toggle road accessibility.")

    updated_communities = []
    
    for i, row in active_df.iterrows():
        name = str(row["Area"])
        expanded = (i == 0)  # Expand only the first card to save mobile scroll space
        
        with st.expander(f"📍 {name} — Sev: {int(row.get('Severity (1-5)', 3))}/5 | Med: {int(row.get('Medical urgency (1-5)', 3))}/5", expanded=expanded):
            c_name = st.text_input("Name", value=name, key=f"comm_name_{i}")
            
            c1, c2 = st.columns(2)
            with c1:
                pop = st.number_input("Population", min_value=1, value=int(row["Population"]), step=100, key=f"comm_pop_{i}")
                sev = st.slider("Severity (1–5)", 1, 5, int(row.get("Severity (1-5)", 3)), key=f"comm_sev_{i}")
                food = st.number_input("Food Needed", min_value=0, value=int(row.get("Food Need", 500)), step=50, key=f"comm_food_{i}")

            with c2:
                med = st.slider("Medical Urgency (1–5)", 1, 5, int(row.get("Medical urgency (1-5)", 3)), key=f"comm_med_{i}")
                dist = st.number_input("Distance (km)", min_value=1, value=int(row.get("Distance (km)", 5)), step=1, key=f"comm_dist_{i}")
                kits = st.number_input("Medical Kits Needed", min_value=0, value=int(row.get("Medical Need", 50)), step=5, key=f"comm_kits_{i}")

            # Road and Vulnerability
            acc_col, vuln_col = st.columns([1, 1])
            with acc_col:
                acc = st.toggle("Road Open?", value=bool(row.get("Road Accessible", True)), key=f"comm_acc_{i}")
            with vuln_col:
                vuln = st.slider("Vulnerability (1–5)", 1, 5, int(row.get("Vulnerability (1-5)", 4)), key=f"comm_vuln_{i}")

            updated_communities.append({
                "Area": c_name.strip() or name,
                "Population": int(pop),
                "Severity (1-5)": int(sev),
                "Medical urgency (1-5)": int(med),
                "Access difficulty (1-5)": min(5, max(1, int(round(dist / 2)))),
                "Distance (km)": int(dist),
                "Vulnerability (1-5)": int(vuln),
                "Food Need": int(food),
                "Medical Need": int(kits),
                "Road Accessible": bool(acc),
            })

    current_df = pd.DataFrame(updated_communities)
    st.session_state["communities_df"] = current_df
    scored_df = calculate_risk_scores(current_df)

    st.markdown("---")
    st.subheader("📈 Risk Scores & Need Profiles")

    # Single view selector to prevent duplicate charts!
    chart_view = st.radio(
        "Choose Visualization View:",
        ["📊 Factor Breakdown (All Communities)", "🕸️ Radar Profile (Selected Community)"],
        horizontal=True,
    )

    r_col1, r_col2 = st.columns([1.1, 1.0])
    with r_col1:
        st.markdown("**Calculated Priority Ranks:**")
        for _, r in scored_df.sort_values("Risk score (0-100)", ascending=False).iterrows():
            tier_icon = "🔴" if r["Risk score (0-100)"] >= 75 else ("🟠" if r["Risk score (0-100)"] >= 50 else "🟡")
            road_tag = "✅ Road Open" if r["Road Accessible"] else "🚧 Cut-Off"
            st.write(f"**{r['Area']}** — **{r['Risk score (0-100)']}/100** ({r['Priority Tier']} {tier_icon}) | *{road_tag}*")
            st.progress(min(100, max(0, int(r["Risk score (0-100)"]))))

    with r_col2:
        if "Factor Breakdown" in chart_view:
            fig_breakdown = plot_risk_factor_breakdown(scored_df)
            st.plotly_chart(fig_breakdown, use_container_width=True, config={"responsive": True, "displayModeBar": False})
        else:
            sel_comm = st.selectbox("Select Community for Radar:", scored_df["Area"].tolist())
            target_comm_row = scored_df[scored_df["Area"] == sel_comm].iloc[0]
            radar_fig = plot_radar_profile(target_comm_row)
            st.plotly_chart(radar_fig, use_container_width=True, config={"responsive": True, "displayModeBar": False})

    # Deep dive text & mathematical explainability (NO duplicate radar chart)
    with st.expander("🔍 Deep Dive: Factor Formula & Explainability Text", expanded=False):
        insp_name = st.selectbox("Explain Community:", scored_df["Area"].tolist(), key="explain_select")
        target_row = scored_df[scored_df["Area"] == insp_name].iloc[0]
        
        st.markdown(explain_area_risk(target_row))
        st.latex(
            r"\text{Score} = 0.30 \cdot S + 0.30 \cdot M + 0.15 \cdot V + 0.15 \cdot P + 0.10 \cdot A"
        )


# ==============================================================================
# TAB 2: RELIEF ALLOCATION (CLASSICAL & QAOA)
# ==============================================================================
with tab2:
    st.header("2. Run Relief Allocation & Optimization")
    st.write("Allocates priority packages, food, and medical kits using Classical Exact and Qiskit QAOA solvers.")

    run_alloc_btn = st.button("🚑 Run Relief Allocation", type="primary", use_container_width=True)

    if run_alloc_btn:
        with st.spinner("Solving optimization (Classical Exact + Qiskit QAOA)..."):
            scores_list = scored_df["Risk score (0-100)"].tolist()
            acc_list = scored_df["Road Accessible"].tolist()
            names_list = scored_df["Area"].tolist()
            food_needs = scored_df["Food Need"].tolist()
            kit_needs = scored_df["Medical Need"].tolist()

            c_sel, c_risk, c_details = solve_classical_allocation(
                scores=scores_list,
                packages=available_packages,
                accessible_flags=acc_list,
                previous_plan=None,
                churn_penalty=0.0,
            )

            q_sel, q_risk, q_details = [], 0.0, {}
            qaoa_ok = False
            if run_qaoa_toggle:
                try:
                    q_sel, q_risk, q_details = solve_quantum_qaoa_allocation(
                        scores=scores_list,
                        packages=available_packages,
                        accessible_flags=acc_list,
                        previous_plan=None,
                        churn_penalty=0.0,
                        seed=qaoa_seed,
                    )
                    qaoa_ok = True
                except Exception as ex:
                    q_details = {"error": str(ex)}
                    qaoa_ok = False

            weights_for_supplies = [scores_list[i] if (i in c_sel and acc_list[i]) else 0.0 for i in range(len(scores_list))]
            
            food_allocated = allocate_units(food_supply, food_needs, weights_for_supplies, acc_list)
            kit_allocated = allocate_units(medical_supply, kit_needs, weights_for_supplies, acc_list)

            results = []
            for i in range(len(names_list)):
                results.append({
                    "Community": names_list[i],
                    "Population": int(scored_df["Population"].iloc[i]),
                    "Risk Score": float(scores_list[i]),
                    "Priority Tier": scored_df["Priority Tier"].iloc[i],
                    "Road Status": "Open ✅" if acc_list[i] else "Cut-Off 🚧",
                    "Priority Package": "📦 Yes" if i in c_sel else "—",
                    "QAOA Selected": ("📦 Yes" if i in q_sel else "—") if qaoa_ok else "Disabled",
                    "Food Needed": int(food_needs[i]),
                    "Food Allocated": int(food_allocated[i]),
                    "Kits Needed": int(kit_needs[i]),
                    "Kits Allocated": int(kit_allocated[i]),
                    "Severity": int(scored_df["Severity (1-5)"].iloc[i]),
                    "Medical Urgency": int(scored_df["Medical urgency (1-5)"].iloc[i]),
                })

            st.session_state["qrescue_results"] = results
            st.session_state["qrescue_c_details"] = c_details
            st.session_state["qrescue_q_details"] = q_details
            st.session_state["qrescue_qaoa_ok"] = qaoa_ok
            st.session_state["qrescue_c_sel"] = c_sel
            st.session_state["qrescue_q_sel"] = q_sel

    if st.session_state.get("qrescue_results"):
        res = st.session_state["qrescue_results"]
        c_det = st.session_state.get("qrescue_c_details", {})
        q_det = st.session_state.get("qrescue_q_details", {})
        qaoa_ok = st.session_state.get("qrescue_qaoa_ok", False)

        st.markdown("---")
        st.subheader("3. Allocation Summary")

        food_used = sum(r["Food Allocated"] for r in res)
        kits_used = sum(r["Kits Allocated"] for r in res)
        pkgs_used = sum(1 for r in res if "Yes" in r["Priority Package"])

        # 2x2 grid for mobile responsiveness
        m_r1_1, m_r1_2 = st.columns(2)
        m_r1_1.metric("Communities Served", f"{pkgs_used} / {len(res)}")
        m_r1_2.metric("Priority Convoys", f"{pkgs_used} / {available_packages}")

        m_r2_1, m_r2_2 = st.columns(2)
        m_r2_1.metric("Food Distributed", f"{food_used:,} / {food_supply:,}", f"Rem: {food_supply - food_used:,}")
        m_r2_2.metric("Medical Kits", f"{kits_used:,} / {medical_supply:,}", f"Rem: {medical_supply - kits_used:,}")

        s_c1, s_c2 = st.columns(2)
        with s_c1:
            st.success(f"**Classical Exact Baseline:** {pkgs_used} packages | Score: **{c_det.get('captured_risk', 0):.1f}** pts")
        with s_c2:
            if qaoa_ok:
                match_txt = "Exact Match ✅" if set(st.session_state.get("qrescue_q_sel", [])) == set(st.session_state.get("qrescue_c_sel", [])) else "Approximated ⚠️"
                st.info(f"**Qiskit QAOA:** Qubits: {q_det.get('qubits_used', 0)} | Score: **{q_det.get('captured_risk', 0):.1f}** pts ({match_txt})")
            else:
                st.warning("QAOA solver bypassed or skipped.")

        st.markdown("---")
        st.subheader("4. Recommended Distribution per Community")

        for r in res:
            with st.container(border=True):
                head_col1, head_col2 = st.columns([1.5, 1])
                with head_col1:
                    st.markdown(f"**{r['Community']}** `{r['Road Status']}`")
                    st.caption(f"Risk: **{r['Risk Score']}** ({r['Priority Tier']}) | Sev: **{r['Severity']}/5** | Med: **{r['Medical Urgency']}/5**")
                with head_col2:
                    if "Yes" in r["Priority Package"]:
                        st.success("📦 **CONVOY ASSIGNED**")
                    elif "Cut-Off" in r["Road Status"]:
                        st.error("🚧 **AIR-DROP REQUIRED**")
                    else:
                        st.warning("⏳ **UNSERVED**")

                a, b = st.columns(2)
                food_pct = (r["Food Allocated"] / max(1, r["Food Needed"])) * 100
                kit_pct = (r["Kits Allocated"] / max(1, r["Kits Needed"])) * 100
                a.metric("Food", f"{r['Food Allocated']:,} / {r['Food Needed']:,}", f"{food_pct:.0f}%")
                b.metric("Medical Kits", f"{r['Kits Allocated']:,} / {r['Kits Needed']:,}", f"{kit_pct:.0f}%")

        st.markdown("---")
        st.subheader("5. Full Allocation Table")
        st.dataframe(pd.DataFrame(res), use_container_width=True, hide_index=True)
    else:
        st.info("👆 Click **'🚑 Run Relief Allocation'** above to compute the plan.")


# ==============================================================================
# TAB 3: RESILIENCE REPLAY ENGINE (LIVE DISRUPTIONS & REPLAY)
# ==============================================================================
with tab3:
    st.header("3. Resilience Replay Engine – Dynamic Disruption Simulator")
    st.write("Inject disruptions live, test prior plan feasibility, and observe adaptive recovery with minimal churn.")

    with st.expander("⚡ Interactive Disruption Injector (Trigger Live)", expanded=True):
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            target_comm = st.selectbox("Target Community:", current_df["Area"].tolist(), key="inj_target")
            disruption_action = st.selectbox(
                "Disruption Type:",
                [
                    "🚧 Sever Road Access (Landslide / Bridge Cut)",
                    "🛣️ Restore Road Route (Crews Clear Road)",
                    "🩺 Medical Trauma Surge (+1 Urgency)",
                    "🌊 Hazard Severity Jump (+1 Severity)",
                    "📉 Central Depot Supply Loss (-1 Package)",
                    "📍 Discover Emergent Cut-Off Outpost",
                ],
                key="inj_action"
            )

        with d_col2:
            custom_note = st.text_input(
                "Incident Description:",
                value=f"Emergency shock recorded in {target_comm}.",
                key="inj_note"
            )
            apply_disruption_btn = st.button("🔥 Inject Disruption Now", type="primary", use_container_width=True)

        if apply_disruption_btn:
            ev_id = f"E{len(st.session_state['disruptions_list']) + 1}"
            if "Sever Road" in disruption_action:
                new_ev = DisruptionEvent(ev_id, f"Road Severed: {target_comm}", "road_block", target_comm, {"road_accessible": False}, custom_note, f"T+{len(st.session_state['disruptions_list']) + 1}h")
            elif "Restore Road" in disruption_action:
                new_ev = DisruptionEvent(ev_id, f"Route Reopened: {target_comm}", "road_cleared", target_comm, {"road_accessible": True}, custom_note, f"T+{len(st.session_state['disruptions_list']) + 1}h")
            elif "Medical Trauma" in disruption_action:
                new_ev = DisruptionEvent(ev_id, f"Medical Surge: {target_comm}", "medical_surge", target_comm, {"delta_medical": 1}, custom_note, f"T+{len(st.session_state['disruptions_list']) + 1}h")
            elif "Hazard Severity" in disruption_action:
                new_ev = DisruptionEvent(ev_id, f"Hazard Escalation: {target_comm}", "severity_spike", target_comm, {"delta_severity": 1}, custom_note, f"T+{len(st.session_state['disruptions_list']) + 1}h")
            elif "Supply Loss" in disruption_action:
                new_ev = DisruptionEvent(ev_id, "Depot Warehouse Flooded", "supply_loss", None, {"lost_packages": 1}, custom_note, f"T+{len(st.session_state['disruptions_list']) + 1}h")
            else:
                new_ev = DisruptionEvent(ev_id, "Emergent Outpost Discovered", "emergent_area", "Isolated Outpost", {
                    "new_area_data": {
                        "Area": "Isolated Outpost", "Population": 1100, "Severity (1-5)": 5, "Medical urgency (1-5)": 5,
                        "Access difficulty (1-5)": 4, "Distance (km)": 9, "Vulnerability (1-5)": 5, "Food Need": 450,
                        "Medical Need": 50, "Road Accessible": True,
                    }
                }, custom_note, f"T+{len(st.session_state['disruptions_list']) + 1}h")

            st.session_state["disruptions_list"].append(new_ev)
            st.session_state["replay_timeline"] = None
            st.success(f"Injected: **{new_ev.title}**")

    if st.button("🔄 Reset Disruption Queue"):
        if "Riverdale" in scenario_choice:
            _, _, reset_disruptions = get_preset_communities_scenario()
        elif "Canonical" in scenario_choice:
            _, _, reset_disruptions = get_preset_canonical_scenario()
        else:
            _, _, reset_disruptions = get_preset_coastal_cyclone_scenario()
        st.session_state["disruptions_list"] = copy.deepcopy(reset_disruptions)
        st.session_state["replay_timeline"] = None
        st.rerun()

    scored_input_df = calculate_risk_scores(current_df)
    active_disruptions = st.session_state["disruptions_list"]

    engine = ResilienceReplayEngine(
        initial_df=scored_input_df,
        initial_packages=available_packages,
        disruptions=active_disruptions,
        churn_penalty=churn_penalty,
        run_qaoa=run_qaoa_toggle,
        qaoa_seed=qaoa_seed,
    )
    timeline_steps = engine.run_replay()
    st.session_state["replay_timeline"] = timeline_steps

    st.markdown("---")
    st.subheader("⏱️ Disaster Timeline Scrubber")

    step_labels = [s.step_label for s in timeline_steps]
    sel_idx = st.select_slider(
        "Scrub Timeline Step:",
        options=list(range(len(timeline_steps))),
        format_func=lambda i: step_labels[i],
        value=0,
    )

    curr_step = timeline_steps[sel_idx]

    # Tactical info layout
    st.markdown(f"### {curr_step.step_label}")
    if curr_step.event is not None:
        st.markdown(f"""
        <div class="disruption-box">
            <b>Incident:</b> {curr_step.event.description}<br>
            <b>Type:</b> <code>{curr_step.event.event_type}</code> | <b>Target:</b> {curr_step.event.target_area or 'Depot'}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📌 **T0 Baseline Operations:** Pre-disruption deployment plan.")

    # Feasibility Check Box
    if curr_step.step_index > 0:
        if not curr_step.prior_plan_feasible:
            st.markdown("""
            <div class="violation-box">
                <b style="color: #991B1B;">🚨 CONSTRAINT VIOLATION: PRIOR PLAN INFEASIBLE</b><br>
                The prior plan cannot be executed under new disaster conditions:
            """, unsafe_allow_html=True)
            for viol in curr_step.feasibility_violations:
                st.markdown(f"- ⚠️ **{viol}**")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="feasible-box">
                <b style="color: #065F46;">✅ PLAN FEASIBILITY INTACT:</b> Prior allocation remained physically viable.
            </div>
            """, unsafe_allow_html=True)

    # Reallocation summary cards
    st.markdown("#### 🔄 Adaptive Reallocation Plan")
    p1, p2, p3 = st.columns(3)
    p1.success(f"**Retained:**\n\n{', '.join(curr_step.stability_metrics.get('retained_areas', [])) or 'None'}")
    p2.error(f"**Dropped:**\n\n{', '.join(curr_step.stability_metrics.get('dropped_areas', [])) or 'None'}")
    p3.info(f"**Added:**\n\n{', '.join(curr_step.stability_metrics.get('added_areas', [])) or 'None'}")

    with st.expander("📢 Incident Commander Executive Briefing", expanded=True):
        st.markdown(curr_step.operational_briefing)


# ==============================================================================
# TAB 4: RECOVERY TIMELINE & ANALYTICS
# ==============================================================================
with tab4:
    st.header("4. Recovery Timeline & Analytics")
    st.write("Tracks relief progression, unmet humanitarian need, and Jain's fairness index.")

    if st.session_state.get("replay_timeline"):
        timeline = st.session_state["replay_timeline"]

        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            st.plotly_chart(plot_allocation_matrix(timeline), use_container_width=True, config={"responsive": True, "displayModeBar": False})
        with r1_c2:
            st.plotly_chart(plot_stability_chart(timeline), use_container_width=True, config={"responsive": True, "displayModeBar": False})

        r2_c1, r2_c2 = st.columns([1.2, 1.0])
        with r2_c1:
            st.plotly_chart(plot_unmet_need_progression(timeline), use_container_width=True, config={"responsive": True, "displayModeBar": False})
        with r2_c2:
            st.markdown("#### ⚖️ Protection Indicators")
            final_step = timeline[-1]
            final_unmet = final_step.unmet_needs

            st.metric("Jain's Fairness Index", f"{final_unmet.get('jains_fairness_index', 1.0):.3f}")
            st.metric("Accessible Demand Coverage", f"{final_unmet.get('accessible_coverage_pct', 0.0):.1f}%")
            st.metric("Unserved Vulnerable Population", f"{final_unmet.get('unserved_vulnerable_pop', 0):,} people")

        st.markdown("### 📋 Multi-Step Timeline Summary")
        summary_rows = []
        for s in timeline:
            u = s.unmet_needs
            stab = s.stability_metrics
            summary_rows.append({
                "Step": s.step_label,
                "Prior Feasible?": "Yes ✅" if s.prior_plan_feasible else "VIOLATION 🚨",
                "Allocated": ", ".join(s.allocated_names),
                "Captured Risk": f"{u.get('captured_risk', 0):.1f} pts",
                "Unmet Risk": f"{u.get('unmet_risk', 0):.1f} pts",
                "Stability": f"{stab.get('stability_pct', 100):.1f}%",
                "Fairness": f"{u.get('jains_fairness_index', 1.0):.3f}",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📥 Export Center")
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            if st.session_state.get("qrescue_results"):
                res_export = st.session_state["qrescue_results"]
                out_csv = io.StringIO()
                csv_writer = csv.DictWriter(out_csv, fieldnames=list(res_export[0].keys()))
                csv_writer.writeheader()
                csv_writer.writerows(res_export)

                st.download_button(
                    "📥 Download Allocation CSV",
                    data=out_csv.getvalue(),
                    file_name="q_rescue_allocation.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        with d_c2:
            report_lines = [
                "# Q-Rescue Next: Incident Commander Report",
                f"**Scenario:** {scenario_choice}",
                "\n## Timeline Briefings",
            ]
            for s in timeline:
                report_lines.append(f"\n### {s.step_label}")
                report_lines.append(s.operational_briefing)
            
            st.download_button(
                "📄 Download Incident Briefing (MD)",
                data="\n".join(report_lines),
                file_name="q_rescue_briefing.md",
                mime="text/markdown",
                use_container_width=True,
            )
    else:
        st.info("Run the Resilience Replay in Tab 3 to view analytics.")


# ==============================================================================
# TAB 5: ARCHITECTURE & TECHNICAL FORMULATION
# ==============================================================================
with tab5:
    st.header("5. Architecture & Quantum QAOA Formulation")
    st.markdown("""
    ### 📐 Explainable Disaster Risk Utility Model
    """)
    st.latex(
        r"R_i = w_{\text{sev}} S_i + w_{\text{med}} M_i + w_{\text{vuln}} V_i + w_{\text{pop}} P_i + w_{\text{acc}} A_i"
    )

    st.markdown("""
    ### 🎯 Classical Adaptive Recovery (Minimum-Change Objective)
    """)
    st.latex(
        r"\max_{x} \quad \sum_{i \in \text{Accessible}} R_i x_i - \lambda_{\text{churn}} \sum_{i=1}^N |x_i - x_i^{\text{prev}}|"
    )

    st.markdown(r"""
    ### ⚛️ Mapping Churn Penalty into Linear QUBO for Qiskit QAOA
    Since $x_i, x_i^{\text{prev}} \in \{0, 1\}$:
    """)
    st.latex(
        r"|x_i - x_i^{\text{prev}}| = (1 - 2x_i^{\text{prev}}) x_i + x_i^{\text{prev}}"
    )
    st.markdown("""
    Thus, the linear coefficient in the QAOA Hamiltonian becomes:
    """)
    st.latex(
        r"\text{Coeff}(x_i) = R_i - \lambda_{\text{churn}} (1 - 2x_i^{\text{prev}})"
    )

st.markdown("---")
st.caption("Q-Rescue Next | Mobile-Optimized Self-Adapting Disaster Relief Planner")
