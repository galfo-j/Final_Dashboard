"""
ASENXO — MSME Project Completion Dashboard (Model-Driven)
============================================================
Uses the trained Logistic Regression pipeline from MSME_CompletionModel.pkl
to estimate completion probability for MSME projects.
Dummy data is used for prototyping – replace with real database queries.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import plotly.express as px

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="ASENXO | Completion Dashboard",
    page_icon="📊",
    layout="wide",
)
# ═══════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-hero">
    <h1>📊 MSME Project Completion Dashboard</h1>
    <p>DOST SETUP 4.0 iFund Program — Western Visayas | Model-driven risk assessment (dummy data)</p>
</div>
""", unsafe_allow_html=True)

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ── Global CSS / Theme ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=JetBrains+Mono:wght@300;400;500&family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #dde3ee;
    font-size: 15px;
    line-height: 1.65;
    -webkit-font-smoothing: antialiased;
}

/* ── App shell ── */
.stApp {
    background-color: #060b14;
    background-image:
        radial-gradient(ellipse 110% 55% at 5%  0%,   rgba(14,165,233,0.13)  0%, transparent 55%),
        radial-gradient(ellipse  70% 45% at 95% 5%,   rgba(99, 102,241,0.11) 0%, transparent 50%),
        radial-gradient(ellipse  60% 70% at 50% 100%, rgba(20,184,166,0.09)  0%, transparent 55%),
        radial-gradient(ellipse  40% 35% at 80% 55%,  rgba(244,63,94,0.06)   0%, transparent 50%);
    min-height: 100vh;
}
.main .block-container { padding: 0 2.4rem 3.5rem; max-width: 98%; }

.sidebar-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    margin: 0.6rem 0.8rem 0.9rem;
    background: linear-gradient(135deg, rgba(20,184,166,0.18), rgba(14,165,233,0.12));
    border: 1px solid rgba(20,184,166,0.28);
    border-radius: 6px;
    padding: 3px 9px;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #2dd4bf;
}
.info-badge, .stat-badge {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 9px;
    padding: 0.5rem 0.85rem;
    font-size: 0.75rem;
    color: #7a90a8;
    line-height: 1.55;
    margin: 0 0 5px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    display: inline-block;
}
.info-badge strong, .stat-badge span { color: #b8cce0; font-weight: 600; }

/* ── Page hero ── */
.page-hero {
    background: linear-gradient(135deg, rgba(14,165,233,0.09) 0%, rgba(99,102,241,0.08) 50%, rgba(20,184,166,0.06) 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 1.75rem 2rem;
    margin: 1.2rem 0 1.6rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.page-hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(14,165,233,0.4), rgba(99,102,241,0.4), transparent);
}
.page-hero::after {
    content: '';
    position: absolute; top: -60px; right: -60px; width: 260px; height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(14,165,233,0.10) 0%, transparent 65%);
    pointer-events: none;
}
.page-hero h1 {
    font-family: 'Fraunces', serif !important;
    font-size: 1.65rem !important;
    font-weight: 600 !important;
    color: #f0f6ff !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0.3rem !important;
}
.page-hero p { color: #7a90a8; font-size: 0.875rem; margin: 0; line-height: 1.55; }

.section-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(14,165,233,0.09);
    border: 1px solid rgba(14,165,233,0.20);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; color: #38bdf8;
    margin-bottom: 0.9rem;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(255,255,255,0.052) 0%, rgba(255,255,255,0.022) 100%) !important;
    border: 1px solid rgba(255,255,255,0.075) !important;
    border-radius: 16px !important;
    padding: 1.1rem 1.3rem !important;
    backdrop-filter: blur(24px) saturate(1.4);
    transition: transform 0.22s ease, box-shadow 0.22s ease;
    position: relative; overflow: hidden;
}
div[data-testid="stMetric"]::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #0ea5e9, #6366f1, #14b8a6);
    opacity: 0.65;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 48px rgba(0,0,0,0.5), 0 0 0 1px rgba(14,165,233,0.20);
    border-color: rgba(14,165,233,0.22) !important;
}
div[data-testid="stMetric"] label { color: #7a90a8 !important; font-size: 0.72rem !important; font-weight: 700 !important; letter-spacing: 0.05em !important; text-transform: uppercase !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-size: 1.55rem !important; font-weight: 500 !important; color: #eaf1fb !important; }

/* ── Custom metric cards (risk cards) ── */
.metric-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border-radius: 14px;
    padding: 18px 22px;
    border-left: 4px solid #4a9eff;
    border-top: 1px solid rgba(255,255,255,0.06);
    border-right: 1px solid rgba(255,255,255,0.06);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 8px;
    backdrop-filter: blur(20px);
}
.metric-card.green  { border-left-color: #2ecc71; }
.metric-card.red    { border-left-color: #e74c3c; }
.metric-card.yellow { border-left-color: #f39c12; }
.metric-card.blue   { border-left-color: #4a9eff; }
.metric-label { font-size: 12px; color: #7a90a8; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.04em; }
.metric-value { font-size: 28px; font-weight: 700; color: #f0f6ff; font-family: 'JetBrains Mono', monospace; }
.metric-sub   { font-size: 11px; color: #56698a; margin-top: 2px; }

.prob-bar-wrap { background: rgba(255,255,255,0.06); border-radius: 6px; height: 10px; overflow: hidden; margin: 8px 0 4px; }
.prob-bar { height: 100%; border-radius: 6px; transition: width 0.3s; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.badge-low    { background: rgba(46,204,113,0.15); color: #2ecc71; }
.badge-medium { background: rgba(243,156,18,0.15); color: #f39c12; }
.badge-high   { background: rgba(231,76,60,0.15); color: #e74c3c; }

.section-title { font-size: 16px; font-weight: 600; color: #c8d8ea; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 6px; margin: 20px 0 12px; }

/* ── Charts / Dataframes ── */
.stPlotlyChart {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.065);
    border-radius: 18px; padding: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.28);
}
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.065);
    border-radius: 16px; overflow: hidden;
}

/* ── Form controls (dropdowns) ── */
div[data-testid="stSelectbox"] > div, div[data-testid="stMultiSelect"] > div {
    background: rgba(255,255,255,0.045) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.095) !important;
}

/* ── Alerts / headings / divider ── */
.stAlert { border-radius: 12px !important; border-left-width: 3px !important; font-size: 0.875rem !important; }
h1 { font-family: 'Fraunces', serif !important; font-weight: 600 !important; color: #f0f6ff !important; }
h2 { font-weight: 700 !important; font-size: 1.2rem !important; color: #c8d8ea !important; }
h3 { font-weight: 600 !important; font-size: 0.85rem !important; color: #7a90a8 !important; text-transform: uppercase; letter-spacing: 0.05em; }
h4 { color: #c8d8ea !important; }
hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.065) !important; margin: 1.3rem 0 !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.025); }
::-webkit-scrollbar-thumb { background: rgba(14,165,233,0.28); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ── Load model ──────────────────────────────────────────────────
MODEL_PATH = "MSME_CompletionModel.pkl"

@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(
            f"Model file '{MODEL_PATH}' not found. Make sure it's in the same "
            "directory this app is run from (or update MODEL_PATH)."
        )
        st.stop()
    except Exception as e:
        st.error(
            f"Could not load the model pickle: {e}\n\n"
            "This often means the pickle was created with a different "
            "scikit-learn version than the one installed here — re-pickle "
            "with a matching environment, or pin scikit-learn in "
            "requirements.txt to match the training environment."
        )
        st.stop()

model = load_model()

# ── Helper functions ────────────────────────────────────────────
def get_completion_prob(row):
    """
    Predict probability of completion (class 1) for a single row.
    Row should be a dict with keys matching the model's feature names.
    """
    X = pd.DataFrame([row])
    prob = model.predict_proba(X)[0][1]
    return float(prob)

def risk_tier(prob):
    # These cutoffs (0.60 / 0.40) are illustrative bands for the dashboard UI,
    # centered on the model's 0.50 default decision threshold,
    # not a threshold derived from the model (e.g. via Youden's J or a
    # precision/recall tradeoff analysis) -- the notebook doesn't include
    # that analysis. Treat them as a starting point and revisit once there's
    # a documented threshold-tuning step, ideally against a larger sample
    # than the current 347 records.
    if prob >= 0.60:
        return "Low", "badge-low"
    if prob >= 0.40:
        return "Medium", "badge-medium"
    return "High", "badge-high"

def tier_color(prob):
    return "#2ecc71" if prob >= 0.60 else ("#f39c12" if prob >= 0.40 else "#e74c3c")

# ── Dummy data generators ──────────────────────────────────────
# Categorical Data Dummy
PROVINCES = ["Aklan", "Antique", "Capiz", "Guimaras", "Iloilo", "Negros Occidental"]
SECTORS = [
    "Agriculture/Marine/Aquaculture",
    "Food Processing",
    "Furniture",
    "Gifts, Decors, Handicrafts",
    "Horticulture & Agriculture",
    "Metals & Engineering",
    "Others (grouped)"
]
OWNERSHIP = ["Cooperative", "Corporation", "Partnership", "Single"]
SIZES = ["medium", "micro", "small"]
RISK_TIERS = ["Low", "Medium", "High"]

def generate_msme_data(n, status="approved"):
    """
    Generate dummy MSME records.
    status: 'approved' or 'applicant' – only affects ID prefix.
    """
    rows = []
    for i in range(1, n + 1):
        province = random.choice(PROVINCES)
        year = random.randint(2018, 2025)
        sector = random.choice(SECTORS)
        ownership = random.choice(OWNERSHIP)
        size = random.choice(SIZES)
        project_cost = int(np.random.randint(150_000, 2_500_000))
        has_prior = random.choice([True, False])

        # NOTE: Year is intentionally excluded here/ dropped during training
        row_dict = {
            "Province": province,
            "Sector": sector,
            "Type of Ownership": ownership,
            "Size of Enterprise": size,
            "Project Cost": project_cost,
            "Has_Prior_Funding": has_prior
        }
        prob = get_completion_prob(row_dict)
        tier, _ = risk_tier(prob)

        prefix = "A" if status == "approved" else "P"
        rows.append({
            "ID": f"MSME-{prefix}{i:03d}",
            "Beneficiary": f"Enterprise {chr(64 + i % 26 + 1)}{i:02d}",
            "Province": province,
            "Year": year,
            "Sector": sector,
            "Type of Ownership": ownership,
            "Size": size,
            "Project Cost (₱)": project_cost,
            "Has Prior Funding": has_prior,
            "Completion Prob.": prob,
            "Risk Tier": tier,
        })
    return pd.DataFrame(rows)

@st.cache_data
def generate_dummy_data():
    approved = generate_msme_data(30, "approved")
    applicants = generate_msme_data(20, "applicant")
    return approved, applicants

approved_df, applicant_df = generate_dummy_data()

# ── Shared plotly template ──────────────────────────────────────
PLOT_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Plus Jakarta Sans, sans-serif', color='#7a90a8', size=12),
    margin=dict(l=52, r=28, t=52, b=48),
    xaxis=dict(gridcolor='rgba(255,255,255,0.045)', zeroline=False,
               title_font=dict(size=11, color='#546a82'), tickfont=dict(size=10, color='#546a82'),
               linecolor='rgba(255,255,255,0.07)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.045)', zeroline=False,
               title_font=dict(size=11, color='#546a82'), tickfont=dict(size=10, color='#546a82'),
               linecolor='rgba(255,255,255,0.07)'),
    legend=dict(bgcolor='rgba(6,11,20,0.7)', bordercolor='rgba(255,255,255,0.07)',
                borderwidth=1, font=dict(size=11, color='#94a3b8')),
    hoverlabel=dict(bgcolor='#0b1628', font_family='JetBrains Mono, monospace',
                     font_size=12, bordercolor='rgba(14,165,233,0.3)', font_color='#dde3ee'),
)
RISK_COLOR_MAP = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}

ALL_LABEL = "All"

# ── Top bar: brand header ────────────────────────────────────────
bh1, bh2 = st.columns([0.08, 0.92])
with bh1:
    st.image("https://www.dost.gov.ph/images/DOSTLogo.png", width=52)
with bh2:
    st.markdown("""
    <div style="padding-top:2px;">
        <span style="font-family:'Fraunces',serif;font-size:1.25rem;font-weight:600;color:#f0f6ff;">ASENXO</span>
        <span style="font-size:0.7rem;color:#4a6080;letter-spacing:0.08em;text-transform:uppercase;font-weight:600;margin-left:8px;">Completion Dashboard</span>
        <span class="sidebar-badge" style="margin-left:10px;">⚡ DOST SETUP 4.0 · iFund</span>
    </div>
    """, unsafe_allow_html=True)

# ── Filters (dropdowns, on-page) ─────────────────────────────────
st.markdown('<div class="section-pill">🔍 Filters</div>', unsafe_allow_html=True)
f1, f2, f3 = st.columns(3)
with f1:
    sel_province = st.selectbox("Province", [ALL_LABEL] + PROVINCES, index=0)
with f2:
    sel_sector = st.selectbox("Sector", [ALL_LABEL] + SECTORS, index=0)
with f3:
    sel_tier = st.selectbox("Risk Tier", [ALL_LABEL] + RISK_TIERS, index=0)

st.markdown("---")

# 
province_list = PROVINCES if sel_province == ALL_LABEL else [sel_province]
sector_list = SECTORS if sel_sector == ALL_LABEL else [sel_sector]
tier_list = RISK_TIERS if sel_tier == ALL_LABEL else [sel_tier]

# Apply filters
filt_approved = approved_df[
    approved_df["Province"].isin(province_list) &
    approved_df["Sector"].isin(sector_list) &
    approved_df["Risk Tier"].isin(tier_list)
]
filt_applicant = applicant_df[
    applicant_df["Province"].isin(province_list) &
    applicant_df["Sector"].isin(sector_list) &
    applicant_df["Risk Tier"].isin(tier_list)
]

total_approved   = len(filt_approved)
total_applicants = len(filt_applicant)
high_risk_appr   = (filt_approved["Risk Tier"] == "High").sum()
high_risk_appl   = (filt_applicant["Risk Tier"] == "High").sum()
avg_prob_appr    = filt_approved["Completion Prob."].mean() if total_approved else 0
avg_prob_appl    = filt_applicant["Completion Prob."].mean() if total_applicants else 0

combined = pd.concat([
    filt_approved.assign(Group="Approved"),
    filt_applicant.assign(Group="Applicant"),
])
# ═══════════════════════════════════════════════════════════════
# PORTFOLIO KPIs
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="section-pill">🎯 Portfolio Snapshot</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Approved MSMEs", total_approved)
with k2:
    st.metric("Applying MSMEs", total_applicants)
with k3:
    st.metric("High-Risk (Approved)", high_risk_appr,
              delta=f"-{high_risk_appr} need follow-up", delta_color="inverse")
with k4:
    st.metric("Avg Completion Prob. (Approved)", f"{avg_prob_appr*100:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# OVERVIEW CHARTS
# ═══════════════════════════════════════════════════════════════
c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="section-pill">🗺️ Risk Tier Distribution</div>', unsafe_allow_html=True)
    if not combined.empty:
        tier_counts = combined.groupby(["Group", "Risk Tier"]).size().reset_index(name="Count")
        fig = px.bar(tier_counts, x="Group", y="Count", color="Risk Tier",
                     color_discrete_map=RISK_COLOR_MAP, barmode="stack",
                     category_orders={"Risk Tier": ["Low", "Medium", "High"]},
                     title="Risk Tier by Group")
        fig.update_layout(**PLOT_LAYOUT, title_font=dict(size=13, color='#c8d8ea'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No records match the current filters.")

with c2:
    st.markdown('<div class="section-pill">🌏 Avg Completion Probability by Province</div>', unsafe_allow_html=True)
    if not combined.empty:
        prov_avg = combined.groupby("Province")["Completion Prob."].mean().reset_index()
        prov_avg = prov_avg.sort_values("Completion Prob.", ascending=False)
        fig2 = px.bar(prov_avg, x="Province", y="Completion Prob.",
                      color="Completion Prob.", color_continuous_scale="Teal",
                      title="Avg Completion Probability by Province")
        fig2.update_traces(hovertemplate="%{x}: %{y:.1%}")
        fig2.update_yaxes(tickformat=".0%")
        fig2.update_coloraxes(showscale=False)
        fig2.update_layout(**PLOT_LAYOUT, title_font=dict(size=13, color='#c8d8ea'))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No records match the current filters.")

st.markdown('<div class="section-pill">🏭 Completion Probability by Sector</div>', unsafe_allow_html=True)
if not combined.empty:
    sector_avg = combined.groupby("Sector")["Completion Prob."].mean().reset_index()
    sector_avg = sector_avg.sort_values("Completion Prob.", ascending=True)
    fig3 = px.bar(sector_avg, y="Sector", x="Completion Prob.", orientation="h",
                  color="Completion Prob.", color_continuous_scale="Bluyl",
                  title="Avg Completion Probability by Sector")
    fig3.update_traces(hovertemplate="%{y}: %{x:.1%}")
    fig3.update_xaxes(tickformat=".0%")
    fig3.update_coloraxes(showscale=False)
    fig3.update_layout(**PLOT_LAYOUT, title_font=dict(size=13, color='#c8d8ea'), height=380)
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No records match the current filters.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# APPROVED / FUNDED MSMEs
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🏢 Approved / Funded MSMEs — Completion Probability</div>',
            unsafe_allow_html=True)
st.caption(
    "Completion probability is estimated from the project profile using the trained model. "
    "High risk (prob < 0.40) indicates profiles similar to historically non‑completing projects."
)

sa1, sa2, sa3 = st.columns(3)
with sa1:
    st.markdown(f"""<div class="metric-card green">
        <div class="metric-label">Avg Completion Probability</div>
        <div class="metric-value">{avg_prob_appr*100:.1f}%</div>
        <div class="metric-sub">Across {total_approved} approved MSMEs</div>
    </div>""", unsafe_allow_html=True)
with sa2:
    low_risk = (filt_approved["Risk Tier"] == "Low").sum()
    st.markdown(f"""<div class="metric-card blue">
        <div class="metric-label">Low‑Risk (Approved)</div>
        <div class="metric-value">{low_risk}</div>
        <div class="metric-sub">Prob ≥ 60%</div>
    </div>""", unsafe_allow_html=True)
with sa3:
    high_risk = (filt_approved["Risk Tier"] == "High").sum()
    st.markdown(f"""<div class="metric-card red">
        <div class="metric-label">High‑Risk (Approved)</div>
        <div class="metric-value">{high_risk}</div>
        <div class="metric-sub">Prob < 40% – monitor closely</div>
    </div>""", unsafe_allow_html=True)

st.markdown("#### Approved MSME Detail")
if filt_approved.empty:
    st.info("No approved MSMEs match the current filters.")
else:
    display_appr = filt_approved.copy()
    display_appr["Completion Prob."] = display_appr["Completion Prob."].apply(lambda x: f"{x*100:.1f}%")
    display_appr["Project Cost (₱)"] = display_appr["Project Cost (₱)"].apply(lambda x: f"₱{x:,.0f}")
    display_appr["Has Prior Funding"] = display_appr["Has Prior Funding"].apply(lambda x: "Yes" if x else "No")

    col_order = ["ID", "Beneficiary", "Province", "Year", "Sector",
                 "Type of Ownership", "Size", "Project Cost (₱)",
                 "Has Prior Funding", "Completion Prob.", "Risk Tier"]
    st.dataframe(display_appr[col_order], use_container_width=True, hide_index=True)

    st.markdown("#### 🔎 MSME Deep Dive")
    selected_id = st.selectbox(
        "Select an approved MSME to inspect:",
        filt_approved["ID"].tolist(),
        key="appr_select"
    )
    row = filt_approved[filt_approved["ID"] == selected_id].iloc[0]

    d1, d2 = st.columns(2)
    with d1:
        prob = row["Completion Prob."]
        tier_label, tier_cls = risk_tier(prob)
        color = tier_color(prob)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Completion Probability</div>
            <div class="metric-value" style="color:{color}">{prob*100:.1f}%</div>
            <div class="prob-bar-wrap"><div class="prob-bar"
                style="width:{prob*100:.1f}%; background:{color};"></div></div>
            <div class="metric-sub">Risk Tier:
                <span class="badge {tier_cls}">{tier_label}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with d2:
        st.markdown("**Profile**")
        profile_data = {
            "Field": ["Beneficiary", "Province", "Year", "Sector",
                      "Type of Ownership", "Size", "Project Cost", "Has Prior Funding"],
            "Value": [
                row["Beneficiary"], row["Province"], row["Year"],
                row["Sector"], row["Type of Ownership"], row["Size"],
                f"₱{row['Project Cost (₱)']:,}",
                "Yes" if row["Has Prior Funding"] else "No"
            ]
        }
        st.dataframe(pd.DataFrame(profile_data), hide_index=True, use_container_width=True)

    if prob >= 0.60:
        st.success("✅ Profile aligns with historical completions – low risk.")
    elif prob >= 0.40:
        st.warning("⚠️ Moderate risk – consider additional support or monitoring.")
    else:
        st.error("🔴 High risk – resembles non‑completing projects; Intense monitoring needed.")

st.markdown("---")

