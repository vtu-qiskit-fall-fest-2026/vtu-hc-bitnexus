# 🚑 Q-Rescue Next – A Self-Adapting Quantum-Assisted Disaster Relief Planner

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.6.0-6929C4.svg)](https://qiskit.org/)
[![Plotly](https://img.shields.io/badge/Plotly-7.1-3F4F75.svg)](https://plotly.com/)

An intelligent disaster-response planning prototype designed to help emergency management teams allocate limited relief resources under cascading disruptions. It integrates **multi-factor explainable risk assessment**, **classical exact optimization**, **Qiskit-based QAOA quantum optimization**, and an innovative **Resilience Replay Engine** that simulates and adapts relief plans as disaster conditions unfold in real time.

---

## 🌟 Main Innovation: Resilience Replay Engine

Conventional disaster relief optimizers solve a static "snapshot" problem, producing a single allocation plan. However, real disasters are dynamic: bridges collapse, floodwaters breach warehouses, disease outbreaks surge, and newly cut-off populations emerge.

The **Resilience Replay Engine**:
1. **Applies Sequential Disruptions:** Simulates road closures, medical surges, warehouse supply loss, and emergent isolated zones across discrete time steps ($T_0 \to T_1 \to T_2 \dots$).
2. **Performs Plan Feasibility Auditing:** Evaluates whether previously committed relief packages can still reach their destinations or if warehouse stock deficits invalidate the prior plan.
3. **Adaptive Recovery Planning (Minimum-Change Objective):** Re-optimizes resource distribution while penalizing logistical churn ($\lambda_{\text{churn}}$), ensuring emergency fleets are not unnecessarily redirected unless justified by critical urgency or severed access.
4. **Unmet-Need & Fairness Analysis:** Continuously evaluates underserved populations, unmet risk burden, and computes **Jain's Fairness Index** to prevent vulnerable communities from being abandoned.
5. **Interactive Timeline & Allocation Flow:** Lets decision-makers scrub through disaster time, visualize reallocation flow heatmaps, and inspect automated Incident Commander executive briefings.

---

## 📊 Canonical Demonstration Scenario

Consider three disaster sectors requiring urgent relief with 3 initial relief packages available:
- **Area A:** High population (8,000), Severity 5/5, Medical urgency 5/5
- **Area B:** Moderate population (5,000), Severity 4/5, Medical urgency 3/5
- **Area C:** High terrain difficulty (5/5), Severity 3/5, Medical urgency 4/5

### Sequential Disruption Timeline:
- **$T_0$ (Baseline Plan):** All 3 areas receive 1 relief package. Both Classical and Qiskit QAOA identify the optimal allocation.
- **$T_1$ (Landslide on Route 4):** Access road to Area B is severed. The Resilience Replay Engine detects an **Infeasibility Constraint Violation** ($x_{\text{Area B}}^{\text{prev}} = 1$ is no longer reachable). The plan adapts by reassigning or holding the package.
- **$T_2$ (Trauma Surge at Area C Clinic):** Water contamination elevates Area C medical urgency to 5/5, boosting its priority score.
- **$T_3$ (Central Depot Inundation):** Flood destroys warehouse stock, reducing available packages from 3 to 2. The adaptive solver stabilizes the fleet, maintaining delivery to Area A and Area C without unnecessary rerouting.

---

## 📐 Mathematical Formulation

### 1. Explainable Disaster Risk Model
$$R_i = w_{\text{sev}} S_i + w_{\text{med}} M_i + w_{\text{vuln}} V_i + w_{\text{pop}} P_i + w_{\text{acc}} A_i$$
- Default weights: $30\%$ Severity, $30\%$ Medical Urgency, $15\%$ Vulnerability, $15\%$ Population scale, $10\%$ Access Difficulty.
- Normalized onto a transparent $0 - 100$ scale.

### 2. Adaptive Recovery Objective (Plan Stability Formulation)
$$\max_{x} \quad \left( \sum_{i \in \text{Accessible}} R_i x_i \right) - \lambda_{\text{churn}} \sum_{i=1}^N |x_i - x_i^{\text{prev}}|$$
$$\text{subject to} \quad \sum_{i \in \text{Accessible}} x_i = \min(K_t, |\text{Accessible}|), \quad x_i \in \{0, 1\}$$
$$x_i = 0 \quad \forall i \notin \text{Accessible}$$

### 3. Quantum QUBO Mapping for Qiskit QAOA
For binary variables $x_i, x_i^{\text{prev}} \in \{0, 1\}$, the non-linear absolute difference $|x_i - x_i^{\text{prev}}|$ is mapped linearly:
$$|x_i - x_i^{\text{prev}}| = (1 - 2x_i^{\text{prev}}) x_i + x_i^{\text{prev}}$$
Thus, the linear coefficient in the QAOA QuadraticProgram Hamiltonian is:
$$\text{Coeff}(x_i) = R_i - \lambda_{\text{churn}} (1 - 2x_i^{\text{prev}})$$
The equality constraint $\sum x_i = K$ is enforced via an Ising penalty Hamiltonian in Qiskit `MinimumEigenOptimizer`.

---

## 🚀 Quickstart Guide

### 1. Activate Environment & Install Dependencies

```powershell
# Clone or navigate to the repository directory
cd d:\PROJECTS\Q-Rescue-Next

# Activate the existing virtual environment
.\.venv\Scripts\Activate.ps1

# Verify requirements
pip install -r requirements.txt
```

### 2. Run the Streamlit Interactive Dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### 📱 Open on Mobile (Phone on Same Wi-Fi)

To interact with the dashboard from your smartphone (iPhone / Android):

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 0.0.0.0
```

Then open `http://<YOUR_LAN_IP>:8501` on your phone's browser (e.g. `http://172.20.10.3:8501`). The exact URL is also displayed inside the sidebar's **📱 View on Phone / Mobile** panel.

### 3. Run the Verification Test Suite

```powershell
.\.venv\Scripts\python.exe -m unittest test_all.py
```

---

## 📂 Project Architecture

```
d:\PROJECTS\Q-Rescue-Next\
├── app.py                   # Streamlit dashboard & interactive scenario scrubber
├── risk_model.py            # Multi-factor explainable risk scoring model
├── optimizers.py            # Exact classical solver & Qiskit QAOA quantum optimizer
├── disruptions.py           # Disruption event models & scenario presets
├── resilience_replay.py     # Resilience Replay Engine & feasibility auditing
├── analytics.py             # Unmet-need, Jain's fairness index, & plan stability
├── charts.py                # Plotly data visualizations & allocation heatmaps
├── classical_optimizer.py   # Backwards-compatible classical wrapper
├── quantum_optimizer.py     # Backwards-compatible quantum QAOA wrapper
├── test_all.py              # Comprehensive unit and integration test suite
└── requirements.txt         # Dependencies (Streamlit, Qiskit, Plotly, Pandas, Scipy)
```

---

## ⚠️ Important Limitations & Ethical Disclaimer

- **Decision-Support Prototype:** This system is an educational and research prototype built with synthetic illustrative data. It does not replace trained emergency management personnel, civil defense directors, or medical triage officers.
- **Quantum Advantage Scope:** QAOA is run locally via statevector simulation to explore hybrid quantum-classical algorithmic properties. Classical exact solvers remain faster and optimal for small problem instances on classical hardware.
