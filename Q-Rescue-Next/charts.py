"""
charts.py - Plotly Visualizations for Q-Rescue Next.
Optimized for both Desktop and Mobile / Touch screens.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_risk_factor_breakdown(scored_df: pd.DataFrame) -> go.Figure:
    """
    Creates a stacked bar chart illustrating how each component contributes
    to the final explainable risk score for each disaster area.
    Mobile-friendly responsive layout.
    """
    df = scored_df.copy()
    
    contrib_cols = [
        ("Severity Contrib", "Severity (30%)", "#EF553B"),
        ("Medical Contrib", "Medical (30%)", "#FF7F0E"),
        ("Vulnerability Contrib", "Vuln (15%)", "#AB63FA"),
        ("Population Contrib", "Pop (15%)", "#19D3F3"),
        ("Access Contrib", "Access (10%)", "#FFA15A"),
    ]

    fig = go.Figure()
    for col, label, color in contrib_cols:
        if col in df.columns:
            fig.add_trace(go.Bar(
                name=label,
                x=df["Area"],
                y=df[col],
                marker_color=color,
                hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.1f}} pts<extra></extra>",
            ))

    fig.update_layout(
        barmode="stack",
        title="<b>Explainable Risk Factor Breakdown</b>",
        title_font=dict(size=14),
        xaxis_title="",
        yaxis_title="Risk Score (0-100)",
        yaxis=dict(range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
        template="plotly_white",
        margin=dict(l=15, r=15, t=50, b=25),
        height=300,
        autosize=True,
    )
    return fig


def plot_allocation_matrix(timeline: List[Any]) -> go.Figure:
    """
    Visualizes the allocation status of all areas across the entire timeline
    (T0, T1, T2...) showing maintained, rerouted, dropped, or cut-off states.
    Mobile-friendly responsive heatmap.
    """
    all_areas = []
    for step in timeline:
        for area in step.data_df["Area"].tolist():
            if area not in all_areas:
                all_areas.append(area)

    time_labels = [f"T{s.step_index}" for s in timeline]
    
    z_matrix = []
    text_matrix = []

    for area in all_areas:
        row_z = []
        row_text = []
        for s in timeline:
            df = s.data_df
            match = df[df["Area"] == area]
            if match.empty:
                row_z.append(-1)
                row_text.append("New")
            else:
                is_acc = bool(match["Road Accessible"].iloc[0])
                is_alloc = area in s.allocated_names
                if not is_acc:
                    row_z.append(2)
                    row_text.append("🚧 Cut-Off")
                elif is_alloc:
                    row_z.append(1)
                    row_text.append("📦 Package")
                else:
                    row_z.append(0)
                    row_text.append("⏳ Unserved")
        z_matrix.append(row_z)
        text_matrix.append(row_text)

    colorscale = [
        [0.0, "#E2E8F0"],     # -1: Not discovered
        [0.25, "#CBD5E1"],    # 0: Unserved
        [0.5, "#10B981"],     # 1: Allocated
        [1.0, "#EF4444"],     # 2: Inaccessible
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=time_labels,
        y=all_areas,
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 10, "color": "black"},
        colorscale=colorscale,
        showscale=False,
        xgap=3,
        ygap=3,
    ))

    fig.update_layout(
        title="<b>Relief Allocation Matrix Over Time</b>",
        title_font=dict(size=14),
        xaxis_title="Time Step",
        yaxis_title="",
        template="plotly_white",
        height=280,
        autosize=True,
        margin=dict(l=15, r=15, t=40, b=25),
    )
    return fig


def plot_unmet_need_progression(timeline: List[Any]) -> go.Figure:
    """
    Dual-axis chart tracking Captured vs Unmet Risk and Jain's Fairness Index
    over sequential disruption steps. Mobile-friendly layout.
    """
    steps = [f"T{s.step_index}" for s in timeline]
    captured = [s.unmet_needs.get("captured_risk", 0.0) for s in timeline]
    unmet = [s.unmet_needs.get("unmet_risk", 0.0) for s in timeline]
    fairness = [s.unmet_needs.get("jains_fairness_index", 1.0) * 100 for s in timeline]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Captured Risk",
        x=steps,
        y=captured,
        marker_color="#10B981",
        hovertemplate="Captured: %{y:.1f} pts<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        name="Unmet Risk",
        x=steps,
        y=unmet,
        marker_color="#F87171",
        hovertemplate="Unmet: %{y:.1f} pts<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        name="Fairness Index (%)",
        x=steps,
        y=fairness,
        mode="lines+markers",
        marker=dict(size=8, color="#6366F1"),
        line=dict(color="#6366F1", width=2, dash="dot"),
        yaxis="y2",
        hovertemplate="Fairness: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title="<b>Risk Protection & Fairness Index</b>",
        title_font=dict(size=14),
        barmode="stack",
        xaxis_title="Scenario Step",
        yaxis=dict(title="Risk Pts"),
        yaxis2=dict(
            title="Fairness (%)",
            overlaying="y",
            side="right",
            range=[0, 115],
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
        template="plotly_white",
        height=300,
        autosize=True,
        margin=dict(l=15, r=15, t=50, b=25),
    )
    return fig


def plot_stability_chart(timeline: List[Any]) -> go.Figure:
    """
    Chart showing Plan Stability (%) and Logistical Churn Count over each transition.
    Mobile-friendly layout.
    """
    steps = [f"T{s.step_index}" for s in timeline]
    stability = [s.stability_metrics.get("stability_pct", 100.0) for s in timeline]
    churn = [s.stability_metrics.get("churn_count", 0) for s in timeline]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        name="Stability (%)",
        x=steps,
        y=stability,
        mode="lines+markers",
        marker=dict(size=8, color="#3B82F6"),
        line=dict(width=2, color="#3B82F6"),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.15)",
        hovertemplate="Stability: %{y:.1f}%<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        name="Reroutes",
        x=steps,
        y=churn,
        marker_color="#F59E0B",
        yaxis="y2",
        hovertemplate="Reroutes: %{y}<extra></extra>",
    ))

    fig.update_layout(
        title="<b>Plan Stability & Truck Reroutes</b>",
        title_font=dict(size=14),
        xaxis_title="Step",
        yaxis=dict(title="Stability (%)", range=[0, 110]),
        yaxis2=dict(
            title="Reroutes",
            overlaying="y",
            side="right",
            range=[0, max(max(churn, default=1) + 2, 4)],
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
        template="plotly_white",
        height=280,
        autosize=True,
        margin=dict(l=15, r=15, t=50, b=25),
    )
    return fig


def plot_radar_profile(row: pd.Series) -> go.Figure:
    """
    Generates a compact, mobile-friendly radar chart of an individual community's profile.
    """
    categories = ["Severity", "Medical", "Access", "Vulnerability"]
    values = [
        float(row.get("Severity (1-5)", 1)),
        float(row.get("Medical urgency (1-5)", 1)),
        float(row.get("Access difficulty (1-5)", 1)),
        float(row.get("Vulnerability (1-5)", 1)),
    ]
    # Close the radar loop
    categories += [categories[0]]
    values += [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(239, 68, 68, 0.25)",
        line=dict(color="#EF4444", width=2),
        name=str(row.get("Area", "Community")),
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], tickvals=[1, 2, 3, 4, 5])
        ),
        showlegend=False,
        title=dict(text=f"<b>Profile: {row.get('Area', 'Community')}</b>", font=dict(size=13)),
        height=260,
        autosize=True,
        margin=dict(l=25, r=25, t=35, b=20),
    )
    return fig
