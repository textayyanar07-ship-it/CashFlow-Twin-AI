import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

from simulator import simulate_cashflow
from model import predict_risk, train_risk_model
from ai_advisor import get_explanation_and_recommendations
from recommendations import get_rule_based_recommendations

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CashFlow Twin AI",
    layout="wide",
    page_icon="◈",
    initial_sidebar_state="expanded"
)

DATA_PATH = "data/sample_msme_data.csv"

if "simulation_history" not in st.session_state:
    st.session_state.simulation_history = []

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data(filepath=DATA_PATH):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # Keep as string first, then parse to Timestamp for consistent x-axis
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None

def init_model():
    if not os.path.exists("model.pkl"):
        train_risk_model()

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Base ── */
    [data-testid="stAppViewContainer"] {
        background-color: #070B14;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    header[data-testid="stHeader"] { display: none; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0D1422;
        border-right: 1px solid #1E2C42;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #94A3B8 !important;
        font-size: 1rem !important;
        padding: 8px 12px !important;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(54, 214, 208, 0.05);
        color: #F8FAFC !important;
    }
    /* Active sidebar item styling handled via pseudo-classes or Streamlit defaults combined with CSS */
    div[role="radiogroup"] > label[data-baseweb="radio"] {
        background-color: transparent;
    }

    /* ── Cards ── */
    .glass-card {
        background: #121C2C;
        border: 1px solid #1E2C42;
        border-radius: 16px;
        padding: 24px 28px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        margin-bottom: 16px;
        min-width: 0;
        overflow: hidden;
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
        position: relative;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(54, 214, 208, 0.3);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    }
    .metric-title {
        color: #94A3B8;
        font-size: clamp(0.7rem, 0.9vw, 0.85rem);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
        font-weight: 600;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-value {
        font-size: clamp(1.5rem, 2.5vw, 2.5rem);
        font-weight: 800;
        line-height: 1.1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        word-break: keep-all;
        hyphens: none;
        letter-spacing: -0.5px;
    }

    /* ── Typography ── */
    h1, h2, h3, h4, h5 { color: #F8FAFC !important; font-weight: 700; letter-spacing: -0.5px; }

    /* ── Badges ── */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 24px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        backdrop-filter: blur(10px);
    }
    .badge-teal  { background: rgba(54, 214, 208, 0.1); color: #36D6D0; border: 1px solid rgba(54, 214, 208, 0.2); }
    .badge-violet{ background: rgba(139,124,255,.1); color:#8B7CFF; border:1px solid rgba(139,124,255,.2); }

    /* ── Divider ── */
    hr { border: none; border-top: 1px solid #1E2C42; margin: 20px 0; }

    /* ── Status dot ── */
    .sdot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px; }
    .sdot-green  { background:#4ADE80; box-shadow:0 0 10px #4ADE80; animation: pulse-green 2s infinite; }
    .sdot-teal   { background:#36D6D0; box-shadow:0 0 10px #36D6D0; animation: pulse-teal 2s infinite; }
    .sdot-violet { background:#8B7CFF; box-shadow:0 0 10px #8B7CFF; animation: pulse-violet 2s infinite; }
    .sdot-warning{ background:#FBBF24; box-shadow:0 0 10px #FBBF24; }

    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.4); }
        70% { box-shadow: 0 0 0 6px rgba(74, 222, 128, 0); }
        100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
    }
    @keyframes pulse-teal {
        0% { box-shadow: 0 0 0 0 rgba(54, 214, 208, 0.4); }
        70% { box-shadow: 0 0 0 6px rgba(54, 214, 208, 0); }
        100% { box-shadow: 0 0 0 0 rgba(54, 214, 208, 0); }
    }
    @keyframes pulse-violet {
        0% { box-shadow: 0 0 0 0 rgba(139, 124, 255, 0.4); }
        70% { box-shadow: 0 0 0 6px rgba(139, 124, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(139, 124, 255, 0); }
    }

    /* ── Slider accent ── */
    .stSlider > div > div > div > div { background-color: #36D6D0 !important; }

    /* ── Action cards ── */
    .action-card {
        background: #121C2C;
        border: 1px solid #1E2C42;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        transition: transform 0.2s;
    }
    .action-card:hover {
        transform: scale(1.01);
    }
    .risk-driver-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
        font-size: 0.95rem;
    }
    .risk-bar-bg { flex:1; background:#1E2C42; border-radius:8px; height:10px; overflow: hidden; }
    .risk-bar-fill { height:10px; border-radius:8px; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1); }
</style>
""",
    unsafe_allow_html=True,
)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
<div style="padding:24px 0 16px 0;">
  <div style="color:#36D6D0;font-size:1.5rem;font-weight:800;letter-spacing:1px;line-height:1.2;">
    CASHFLOW<br>TWIN AI
  </div>
  <div style="color:#94A3B8;font-size:0.8rem;font-style:italic;margin-top:8px;line-height:1.4;">
    Your Business. Simulated Before It's Lived.
  </div>
</div>
<div style="color:#94A3B8;font-size:0.7rem;font-weight:700;letter-spacing:1.5px;margin:24px 0 12px 0;">NAVIGATION</div>
""",
        unsafe_allow_html=True,
    )

    nav_items = [
        "◉ Overview",
        "◇ Financial Twin",
        "◇ Time Machine",
        "◇ Risk Intelligence",
        "◇ Twin Advisor",
        "◇ Action Center",
        "◇ Data",
    ]
    selection = st.radio("nav", nav_items, label_visibility="collapsed")

    st.markdown(
        """
<div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #1E2C42;">
    <div style="color:#94A3B8;font-size:0.7rem;font-weight:700;letter-spacing:1.5px;margin-bottom:12px;">SYSTEM STATUS</div>
    <div style="font-size:0.85rem;margin-bottom:12px;display:flex;align-items:center;">
      <span class="sdot sdot-green"></span>Twin Engine <strong style="color:#F8FAFC;margin-left:6px;">ONLINE</strong>
    </div>
    <div style="font-size:0.85rem;margin-bottom:24px;display:flex;align-items:center;">
      <span class="sdot sdot-teal"></span>Prediction Engine <strong style="color:#F8FAFC;margin-left:6px;">ACTIVE</strong>
    </div>
    <div style="color:#94A3B8;font-size:0.7rem;font-weight:700;letter-spacing:1.5px;margin-bottom:8px;">DATA</div>
    <div style="font-size:0.82rem;color:#94A3B8;display:flex;align-items:center;">
      <span class="sdot sdot-violet" style="animation:none;box-shadow:none;"></span>Synthetic Demo
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

page_name = selection[2:].strip()

# ─── INIT DATA ────────────────────────────────────────────────────────────────
init_model()
df = load_data()
if df is None:
    st.error("⚠ Data missing. Run: python data_generator.py")
    st.stop()

historical_df = df[df["type"] == "historical"].copy()
expected_df   = df[df["type"] == "expected"].copy()

# ─── MANUAL CURRENT CASH STATE ────────────────────────────────────────────────
# The dataset value is used only as the initial default. After that, the user's
# manual value is the source of truth for all Digital Twin calculations.
default_current_cash = float(historical_df.iloc[-1]["closing_balance"])

if "manual_current_cash" not in st.session_state:
    st.session_state.manual_current_cash = default_current_cash

current_cash = float(st.session_state.manual_current_cash)

avg_inflow   = float(historical_df["customer_payments_in"].mean())
avg_outflow  = float(
    (
        historical_df["supplier_payments"]
        + historical_df["payroll"]
        + historical_df["rent"]
        + historical_df["utilities"]
        + historical_df["loan_emi"]
        + historical_df["other_expenses"]
    ).mean()
)

def build_risk_features(sup_pct=0.0, cus_delay=0, exp_pct=0.0, runway=60, deficit=0.0):
    """Build a consistent feature dict for predict_risk."""
    return {
        "current_cash":        current_cash,
        "supplier_growth":     sup_pct / 100.0,
        "customer_delay":      cus_delay,
        "expense_pressure":    exp_pct / 100.0,
        "runway_days":         runway,
        "deficit_amount":      deficit if deficit else 0.0,
    }

baseline_sim, b_runway, b_date, b_deficit = simulate_cashflow(
    expected_df, current_cash,
    supplier_pct_change=0, customer_delay_days=0,
    expense_pct_change=0, new_order_amount=0,
    new_order_day=0, emi_shift_days=0,
)
b_risk = predict_risk(build_risk_features(
    sup_pct=0, cus_delay=0, exp_pct=0,
    runway=b_runway, deficit=b_deficit if b_deficit else 0,
))

today_ts = historical_df.iloc[-1]["date"]   # pd.Timestamp

if "sim_sup" not in st.session_state: st.session_state.sim_sup = 0
if "sim_cus" not in st.session_state: st.session_state.sim_cus = 0
if "sim_exp" not in st.session_state: st.session_state.sim_exp = 0
if "sim_new" not in st.session_state: st.session_state.sim_new = 0

def update_sim_from_ov():
    st.session_state.sim_sup = st.session_state.ov_sup
    st.session_state.sim_cus = st.session_state.ov_cus
    st.session_state.sim_exp = st.session_state.ov_exp
    st.session_state.sim_new = st.session_state.ov_new
    st.session_state.tm_sup = st.session_state.ov_sup
    st.session_state.tm_cus = st.session_state.ov_cus
    st.session_state.tm_exp = st.session_state.ov_exp
    st.session_state.tm_new = int(st.session_state.ov_new / 100000)

def update_sim_from_tm():
    st.session_state.sim_sup = st.session_state.tm_sup
    st.session_state.sim_cus = st.session_state.tm_cus
    st.session_state.sim_exp = st.session_state.tm_exp
    st.session_state.sim_new = st.session_state.tm_new * 100000
    st.session_state.ov_sup = st.session_state.tm_sup
    st.session_state.ov_cus = st.session_state.tm_cus
    st.session_state.ov_exp = st.session_state.tm_exp
    st.session_state.ov_new = st.session_state.tm_new * 100000

scenario_sim, s_runway, s_date, s_deficit = simulate_cashflow(
    expected_df, current_cash,
    supplier_pct_change=st.session_state.sim_sup,
    customer_delay_days=st.session_state.sim_cus,
    expense_pct_change=st.session_state.sim_exp,
    new_order_amount=float(st.session_state.sim_new),
    emi_shift_days=0,
)
s_risk = predict_risk(build_risk_features(
    sup_pct=st.session_state.sim_sup, cus_delay=st.session_state.sim_cus, exp_pct=st.session_state.sim_exp,
    runway=s_runway, deficit=s_deficit if s_deficit else 0,
))

simulation = {
    "current_cash": current_cash,
    "scenario_sim": scenario_sim,
    "scenario_runway": s_runway,
    "projected_deficit": s_deficit if s_deficit else 0.0,
    "deficit_date": s_date,
    "risk_score": s_risk["score"],
    "risk_level": s_risk["band"],
    "runway_days": s_runway,
    "risk_band": s_risk["band"],
    "deficit_amount": s_deficit if s_deficit else 0.0,
    "risk_drivers": {
        "sup_ratio": min(1.0, float(historical_df["supplier_payments"].mean()) * (1 + st.session_state.sim_sup/100.0) / max(avg_inflow, 1)),
        "emi_ratio": min(1.0, float(expected_df["loan_emi"].max()) / max(current_cash, 1) * 2),
        "exp_ratio": min(1.0, (avg_outflow * (1 + st.session_state.sim_exp/100.0)) / max(avg_inflow, 1)),
        "delay_norm": min(1.0, st.session_state.sim_cus / 60.0)
    }
}


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def metric_html(title, value, color="#F8FAFC"):
    return (
        f'<div class="glass-card">'
        f'<div class="metric-title">{title}</div>'
        f'<div class="metric-value" style="color:{color};">{value}</div>'
        f'</div>'
    )

def band_color(band):
    return {"LOW": "#22C55E", "MEDIUM": "#FFB547", "HIGH": "#FF5C6C"}.get(band, "#94A3B8")

def runway_color(days):
    if days >= 45: return "#22C55E"
    if days >= 20: return "#FFB547"
    return "#FF5C6C"

def make_forecast_chart(hist_df, base_sim, scen_sim=None, show_today=True, height=400):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["closing_balance"],
        mode="lines", name="Historical Cash",
        line=dict(color="#94A3B8", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=base_sim["date"], y=base_sim["closing_balance"],
        mode="lines", name="Baseline Forecast",
        line=dict(color="rgba(25,211,197,0.45)", dash="dot", width=2)
    ))
    if scen_sim is not None:
        fig.add_trace(go.Scatter(
            x=scen_sim["date"], y=scen_sim["closing_balance"],
            mode="lines", name="Simulated Scenario",
            line=dict(color="#19D3C5", width=3)
        ))
    # Zero danger line
    fig.add_hline(y=0, line_dash="dash", line_color="#FF5C6C",
                  annotation_text="CASH ZERO", annotation_position="bottom right",
                  annotation=dict(font_color="#FF5C6C", font_size=11))
    # TODAY marker
    if show_today:
        fig.add_shape(
            type="line",
            x0=today_ts, x1=today_ts,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(dash="dot", color="rgba(248,250,252,0.5)")
        )
        fig.add_annotation(
            x=today_ts, y=1,
            xref="x", yref="paper",
            text="TODAY",
            showarrow=False,
            font=dict(color="#F8FAFC", size=11),
            xanchor="right", yanchor="top"
        )
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#101827", bordercolor="#1a2235"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color="#94A3B8")
    fig.update_yaxes(showgrid=True, gridcolor="#101827", zeroline=False, color="#94A3B8",
                     tickprefix="₹")
    return fig

def risk_meter_html(score, band):
    pct = min(max(float(score), 0), 100)
    color = band_color(band)
    return f"""
<div style="text-align:center;padding:36px;background:#101827;border-radius:12px;margin-bottom:24px;">
  <div style="color:#94A3B8;letter-spacing:2px;font-size:0.85rem;margin-bottom:8px;">RISK SCORE</div>
  <div style="font-size:5rem;font-weight:800;color:{color};line-height:1;">
    {score:.0f}<span style="font-size:2rem;color:#94A3B8;"> / 100</span>
  </div>
  <div style="color:{color};font-size:1.1rem;font-weight:700;margin-top:8px;letter-spacing:2px;">{band} RISK</div>
  <div style="margin:24px auto 0;width:70%;height:10px;
       background:linear-gradient(90deg,#22C55E 0%,#FFB547 50%,#FF5C6C 100%);
       border-radius:10px;position:relative;">
    <div style="position:absolute;top:-5px;left:calc({pct:.1f}% - 10px);
         width:20px;height:20px;background:#F8FAFC;border-radius:50%;
         box-shadow:0 0 8px rgba(0,0,0,0.5);"></div>
  </div>
  <div style="display:flex;justify-content:space-between;width:70%;margin:8px auto 0;
       font-size:0.75rem;color:#94A3B8;">
    <span>LOW</span><span>HIGH</span>
  </div>
</div>"""

# ─── TOP STATUS BAR ───────────────────────────────────────────────────────────
st.markdown(
    """
<div style="display:flex;gap:24px;font-size:0.75rem;color:#94A3B8;
     margin-bottom:24px;text-transform:uppercase;letter-spacing:1px;flex-wrap:wrap;
     background:#121C2C;padding:12px 24px;border-radius:12px;border:1px solid #1E2C42;">
  <div style="display:flex;align-items:center;"><span class="sdot sdot-green"></span> TWIN ACTIVE</div>
  <div style="display:flex;align-items:center;"><span class="sdot sdot-teal"></span> PREDICTION ENGINE ACTIVE</div>
  <div style="display:flex;align-items:center;"><span class="sdot sdot-violet"></span> SIMULATION READY</div>
</div>""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page_name == "Overview":
    st.markdown(
        """
        <div style="margin-bottom: 32px;">
            <h1 style='font-weight:800;font-size:3.2rem;margin-bottom:0;letter-spacing:-1px;'>YOUR BUSINESS.</h1>
            <h1 style='font-weight:800;font-size:3.2rem;margin-top:-12px;color:#36D6D0;letter-spacing:-1px;'>SIMULATED BEFORE IT'S LIVED.</h1>
            <p style='color:#94A3B8;font-size:1.1rem;margin-top:16px;max-width:800px;line-height:1.6;'>
                See where your cash is heading, understand financial risk, and test decisions before making them.
                Your AI-powered Financial Command Center.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Manual Current Cash input ──
    input_col, info_col, reset_col = st.columns([1.4, 2.2, 0.8])

    with input_col:
        st.number_input(
            "💰 Enter Current Cash (₹)",
            min_value=0.0,
            step=10000.0,
            key="manual_current_cash",
            help="Enter the actual current cash available in the business.",
        )

    with info_col:
        st.markdown(
            "<div style='margin-top:30px;color:#94A3B8;font-size:0.88rem;'>"
            "This manual value drives the Digital Twin. Cash runway, risk score, "
            "projected deficit and the 60-day forecast update automatically."
            "</div>",
            unsafe_allow_html=True,
        )

    def reset_current_cash():
        st.session_state.manual_current_cash = default_current_cash

    with reset_col:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        st.button(
            "↺ Reset",
            key="reset_current_cash_button",
            use_container_width=True,
            on_click=reset_current_cash,
        )

    # ── Hero KPI row ──
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        metric_html("CURRENT CASH", f"₹{current_cash/100000:,.2f} L"),
        unsafe_allow_html=True,
    )
    c2.markdown(metric_html("CASH RUNWAY",  f"{b_runway} DAYS",  runway_color(b_runway)),  unsafe_allow_html=True)
    c3.markdown(metric_html("RISK SCORE",   f"{b_risk['score']:.0f} / 100", band_color(b_risk["band"])), unsafe_allow_html=True)
    def_val = f"₹{b_deficit/100000:,.2f} L" if b_deficit else "NONE"
    c4.markdown(metric_html("PROJECTED DEFICIT", def_val, "#FF5C6C" if b_deficit else "#22C55E"), unsafe_allow_html=True)

    # ── Forecast chart ──
    st.markdown("### 📊 60-DAY CASH REALITY")
    st.plotly_chart(make_forecast_chart(historical_df[-30:], baseline_sim), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Time Machine sliders ──
    st.markdown("### ⏳ FINANCIAL TIME MACHINE")
    st.markdown(
        "<p style='color:#94A3B8;margin-bottom:16px;'>"
        "Change one business condition. Watch your financial future react.</p>",
        unsafe_allow_html=True,
    )

    sl1, sl2, sl3, sl4 = st.columns(4)
    with sl1: s_sup = st.slider("Supplier Payment Increase (%)", 0, 100, st.session_state.sim_sup, key="ov_sup", on_change=update_sim_from_ov)
    with sl2: s_cus = st.slider("Customer Payment Delay (Days)", 0, 60,  st.session_state.sim_cus, key="ov_cus", on_change=update_sim_from_ov)
    with sl3: s_exp = st.slider("Monthly Expense Increase (%)",  0, 100, st.session_state.sim_exp, key="ov_exp", on_change=update_sim_from_ov)
    with sl4: s_new = st.number_input("New Order Value (₹)", min_value=0, value=int(st.session_state.sim_new), step=100000, key="ov_new", on_change=update_sim_from_ov)

    s_runway = simulation["scenario_runway"]
    s_deficit = simulation["projected_deficit"]
    s_risk = {"band": simulation["risk_level"]}

    # ── Live chart with scenario overlay ──
    st.plotly_chart(
        make_forecast_chart(historical_df[-30:], baseline_sim, simulation["scenario_sim"]),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # ── Before / After cards ──
    diff = s_runway - b_runway
    diff_color = "#FF5C6C" if diff < 0 else ("#22C55E" if diff > 0 else "#94A3B8")
    diff_arrow = "▼" if diff < 0 else ("▲" if diff > 0 else "—")
    diff_label = f"{diff_arrow} {abs(diff)} DAYS {'LOST' if diff < 0 else 'GAINED'}" if diff != 0 else "NO CHANGE"

    ba1, ba2, ba3 = st.columns(3)
    with ba1:
        st.markdown(
            f"""<div class="glass-card">
              <div style="color:#94A3B8;font-size:0.75rem;margin-bottom:4px;letter-spacing:1px;font-weight:600;">BASELINE</div>
              <div style="font-size:2.2rem;font-weight:800;color:#4ADE80;line-height:1.2;">{b_runway} DAYS</div>
              <div style="font-size:0.9rem;color:{band_color(b_risk['band'])};margin-top:4px;">{b_risk['band']} RISK</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with ba2:
        sc_col = runway_color(s_runway)
        st.markdown(
            f"""<div class="glass-card" style="border-color:{sc_col}50; box-shadow: 0 8px 32px {sc_col}15;">
              <div style="color:#94A3B8;font-size:0.75rem;margin-bottom:4px;letter-spacing:1px;font-weight:600;">STRESS SCENARIO</div>
              <div style="font-size:2.2rem;font-weight:800;color:{sc_col};line-height:1.2;">{s_runway} DAYS</div>
              <div style="font-size:0.9rem;color:{band_color(s_risk['band'])};margin-top:4px;">{s_risk['band']} RISK</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with ba3:
        def_str = f"₹{s_deficit/100000:,.2f} L deficit" if s_deficit else "No deficit projected"
        st.markdown(
            f"""<div class="glass-card" style="display:flex;flex-direction:column;justify-content:center;min-height:120px;">
              <div style="font-size:1.6rem;font-weight:800;color:{diff_color};">{diff_label}</div>
              <div style="font-size:0.9rem;color:#94A3B8;margin-top:8px;">{def_str}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        
    st.markdown("<hr style='margin: 40px 0;'/>", unsafe_allow_html=True)
    st.markdown("### 🧬 DIGITAL TWIN STORY")
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:center; background:#121C2C; padding:32px 40px; border-radius:16px; border:1px solid #1E2C42; margin-top:24px; flex-wrap:wrap; gap:16px;">
            <div style="text-align:center; flex:1;">
                <div style="font-size:1.5rem; margin-bottom:8px;">📊</div>
                <div style="font-weight:700; font-size:0.85rem; letter-spacing:1px; color:#F8FAFC;">BUSINESS DATA</div>
            </div>
            <div style="color:#36D6D0; font-size:1.2rem;">→</div>
            <div style="text-align:center; flex:1;">
                <div style="font-size:1.5rem; margin-bottom:8px;">💎</div>
                <div style="font-weight:700; font-size:0.85rem; letter-spacing:1px; color:#36D6D0; text-shadow: 0 0 10px rgba(54,214,208,0.4);">DIGITAL FINANCIAL TWIN</div>
            </div>
            <div style="color:#36D6D0; font-size:1.2rem;">→</div>
            <div style="text-align:center; flex:1;">
                <div style="font-size:1.5rem; margin-bottom:8px;">🧠</div>
                <div style="font-weight:700; font-size:0.85rem; letter-spacing:1px; color:#8B7CFF;">AI PREDICTION ENGINE</div>
            </div>
            <div style="color:#36D6D0; font-size:1.2rem;">→</div>
            <div style="text-align:center; flex:1;">
                <div style="font-size:1.5rem; margin-bottom:8px;">⚠️</div>
                <div style="font-weight:700; font-size:0.85rem; letter-spacing:1px; color:#FBBF24;">RISK INTELLIGENCE</div>
            </div>
            <div style="color:#36D6D0; font-size:1.2rem;">→</div>
            <div style="text-align:center; flex:1;">
                <div style="font-size:1.5rem; margin-bottom:8px;">🎯</div>
                <div style="font-weight:700; font-size:0.85rem; letter-spacing:1px; color:#4ADE80;">ACTION RECOMMENDATIONS</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — FINANCIAL TWIN
# ══════════════════════════════════════════════════════════════════════════════
elif page_name == "Financial Twin":
    st.markdown("### ◈ DIGITAL FINANCIAL TWIN")
    st.markdown(
        "<p style='color:#94A3B8;margin-bottom:24px;font-size:1.05rem;'>A living simulation of the business's financial reality.</p>",
        unsafe_allow_html=True,
    )

    # Data flow diagram
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:center; background:#121C2C; padding:32px 40px; border-radius:16px; border:1px solid #1E2C42; margin-bottom:32px; flex-wrap:wrap; gap:16px;">
            <div style="text-align:center; flex:1;">
                <div style="font-size:1.5rem; margin-bottom:8px;">📊</div>
                <div style="font-weight:700; font-size:0.85rem; letter-spacing:1px; color:#F8FAFC;">REAL BUSINESS DATA</div>
            </div>
            <div style="color:#36D6D0; font-size:1.2rem;">→</div>
            <div style="text-align:center; flex:1;">
                <div style="font-size:1.5rem; margin-bottom:8px;">💎</div>
                <div style="font-weight:700; font-size:0.85rem; letter-spacing:1px; color:#36D6D0; text-shadow: 0 0 10px rgba(54,214,208,0.4);">DIGITAL FINANCIAL TWIN</div>
            </div>
            <div style="color:#36D6D0; font-size:1.2rem;">→</div>
            <div style="text-align:center; flex:1;">
                <div style="font-size:1.5rem; margin-bottom:8px;">🧠</div>
                <div style="font-weight:700; font-size:0.85rem; letter-spacing:1px; color:#8B7CFF;">PREDICTION ENGINE</div>
            </div>
            <div style="color:#36D6D0; font-size:1.2rem;">→</div>
            <div style="text-align:center; flex:1;">
                <div style="font-size:1.5rem; margin-bottom:8px;">⚠️</div>
                <div style="font-weight:700; font-size:0.85rem; letter-spacing:1px; color:#FBBF24;">RISK INTELLIGENCE</div>
            </div>
            <div style="color:#36D6D0; font-size:1.2rem;">→</div>
            <div style="text-align:center; flex:1;">
                <div style="font-size:1.5rem; margin-bottom:8px;">🤖</div>
                <div style="font-weight:700; font-size:0.85rem; letter-spacing:1px; color:#4ADE80;">AI RECOMMENDATIONS</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input source cards
    i1, i2, i3 = st.columns(3)
    i1.markdown(metric_html("🏦 BANK TRANSACTIONS", "Connected", "#22C55E"),  unsafe_allow_html=True)
    i2.markdown(metric_html("🧾 INVOICES",          "Syncing",   "#19D3C5"),  unsafe_allow_html=True)
    i3.markdown(metric_html("📑 GST / BUSINESS DATA","Validated", "#22C55E"), unsafe_allow_html=True)

    i4, i5, i6 = st.columns(3)
    monthly_exp = (historical_df["rent"].mean() + historical_df["utilities"].mean()) * 30
    monthly_sup = historical_df["supplier_payments"].mean() * 30
    i4.markdown(metric_html("💸 EXPENSES",          f"₹{monthly_exp:,.0f}/mo"), unsafe_allow_html=True)
    i5.markdown(metric_html("🤝 SUPPLIER PAYMENTS", f"₹{monthly_sup:,.0f}/mo"), unsafe_allow_html=True)
    i6.markdown(metric_html("📈 SEASONAL PATTERNS", "Detected", "#19D3C5"),     unsafe_allow_html=True)

    st.markdown("#### CURRENT BUSINESS STATE")
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(metric_html("CURRENT CASH",       f"₹{current_cash:,.0f}"),  unsafe_allow_html=True)
    s2.markdown(metric_html("AVG DAILY INFLOW",   f"₹{avg_inflow:,.0f}"),    unsafe_allow_html=True)
    s3.markdown(metric_html("AVG DAILY OUTFLOW",  f"₹{avg_outflow:,.0f}"),   unsafe_allow_html=True)
    net = avg_inflow - avg_outflow
    net_col = "#22C55E" if net >= 0 else "#FF5C6C"
    s4.markdown(metric_html("NET DAILY FLOW",     f"₹{net:+,.0f}", net_col),  unsafe_allow_html=True)

    # Twin status
    st.markdown(
        '<div class="badge badge-teal">'
        '<span class="sdot sdot-teal"></span>TWIN STATUS: ACTIVE</div>',
        unsafe_allow_html=True,
    )

    # Historical chart
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### HISTORICAL CASH FLOW")
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(
        x=historical_df["date"], y=historical_df["closing_balance"],
        mode="lines", name="Closing Balance",
        fill="tozeroy", fillcolor="rgba(25,211,197,0.05)",
        line=dict(color="#19D3C5", width=2),
    ))
    fig_h.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8"), showlegend=False,
    )
    fig_h.update_xaxes(showgrid=False, zeroline=False)
    fig_h.update_yaxes(showgrid=True, gridcolor="#101827", zeroline=False, tickprefix="₹")
    st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — TIME MACHINE
# ══════════════════════════════════════════════════════════════════════════════
elif page_name == "Time Machine":
    st.markdown("### ⏳ FINANCIAL TIME MACHINE")
    st.markdown(
        "<p style='color:#94A3B8;margin-bottom:24px;font-size:1.05rem;'>Stress-test your business before reality does.</p>",
        unsafe_allow_html=True,
    )

    col_ctrl, col_res = st.columns([1, 2])

    with col_ctrl:
        st.markdown("#### SCENARIO CONTROLS")
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        tm_sup = st.slider("Supplier Payment Increase (%)", 0, 100, st.session_state.sim_sup, key="tm_sup", on_change=update_sim_from_tm)
        tm_cus = st.slider("Customer Payment Delay (Days)", 0, 60,   st.session_state.sim_cus, key="tm_cus", on_change=update_sim_from_tm)
        tm_exp = st.slider("Expense Increase (%)",          0, 100,  st.session_state.sim_exp, key="tm_exp", on_change=update_sim_from_tm)
        tm_new = st.slider("New Order (₹ Lakhs)",           0,  50,  int(st.session_state.sim_new/100000), key="tm_new", on_change=update_sim_from_tm)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 RUN SIMULATION", use_container_width=True):
            st.success("Simulation parameters applied.")
            
        if st.button("💾 Save Scenario", use_container_width=True):
            st.session_state.simulation_history.append(
                {"sup": tm_sup, "cus": tm_cus, "exp": tm_exp, "new": tm_new * 100000}
            )
            st.success("Scenario saved.")
        st.markdown("</div>", unsafe_allow_html=True)

    s_runway = simulation["scenario_runway"]
    s_deficit = simulation["projected_deficit"]
    s_risk_tm = {"score": simulation["risk_score"], "band": simulation["risk_level"]}

    with col_res:
        st.markdown("#### LIVE RESULT")
        r1, r2, r3 = st.columns(3)
        r1.markdown(metric_html("RUNWAY",  f"{s_runway} DAYS", runway_color(s_runway)), unsafe_allow_html=True)
        r2.markdown(metric_html("RISK",    f"{s_risk_tm['score']:.0f} / 100", band_color(s_risk_tm["band"])), unsafe_allow_html=True)
        def_val_tm = f"₹{s_deficit/100000:,.2f} L" if s_deficit else "NONE"
        r3.markdown(metric_html("DEFICIT", def_val_tm, "#FF5C6C" if s_deficit else "#4ADE80"), unsafe_allow_html=True)

        st.markdown("#### BASELINE VS SCENARIO")
        st.markdown("<div class='glass-card' style='padding: 16px;'>", unsafe_allow_html=True)
        fig_tm = make_forecast_chart(historical_df[-30:], baseline_sim, simulation["scenario_sim"], height=320)
        st.plotly_chart(fig_tm, use_container_width=True, config={"displayModeBar": False})
        
        # Runway impact
        diff_tm = s_runway - b_runway
        d_col = "#FF5C6C" if diff_tm < 0 else ("#4ADE80" if diff_tm > 0 else "#94A3B8")
        d_sym = "▼" if diff_tm < 0 else ("▲" if diff_tm > 0 else "—")
        st.markdown(
            f"<div style='font-size:1.1rem;font-weight:800;color:{d_col};margin-top:12px;text-align:center;letter-spacing:1px;'>"
            f"RUNWAY IMPACT: {d_sym} {abs(diff_tm)} DAYS "
            f"({'LOST' if diff_tm < 0 else 'GAINED' if diff_tm > 0 else 'NO CHANGE'})"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Simulation history
    if st.session_state.simulation_history:
        st.markdown("---")
        st.markdown("#### SIMULATION HISTORY")
        h_cols = st.columns(min(len(st.session_state.simulation_history), 4))
        for idx, h in enumerate(reversed(st.session_state.simulation_history[-4:])):
            with h_cols[idx]:
                st.markdown(
                    f"<div class='glass-card' style='padding:14px 16px;'>"
                    f"<div style='color:#94A3B8;font-size:0.7rem;margin-bottom:4px;'>SCENARIO {len(st.session_state.simulation_history)-idx}</div>"
                    f"<div style='font-size:0.82rem;'>Supplier +{h['sup']}%</div>"
                    f"<div style='font-size:0.82rem;'>Delay {h['cus']}d</div>"
                    f"<div style='font-size:0.82rem;'>Expense +{h['exp']}%</div>"
                    f"<div style='font-size:0.82rem;'>Order ₹{h['new']/100000:.1f}L</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
        if st.button("Clear History"):
            st.session_state.simulation_history = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — RISK INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page_name == "Risk Intelligence":
    st.markdown("### ◉ RISK INTELLIGENCE")
    st.markdown(
        "<p style='color:#94A3B8;margin-bottom:24px;font-size:1.05rem;'>Understand what is driving your liquidity risk.</p>",
        unsafe_allow_html=True,
    )

    # Risk meter (rendered as a single HTML block — no Streamlit widgets inside)
    st.markdown(risk_meter_html(simulation["risk_score"], simulation["risk_level"]), unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    # ── Risk drivers ──
    sup_ratio  = simulation["risk_drivers"]["sup_ratio"]
    emi_ratio  = simulation["risk_drivers"]["emi_ratio"]
    exp_ratio  = simulation["risk_drivers"]["exp_ratio"]
    delay_norm = simulation["risk_drivers"]["delay_norm"]

    def driver_bar(label, value, color_start, color_end):
        pct = round(value * 100)
        return (
            f"<div style='margin-bottom:20px;'>"
            f"<div style='display:flex;justify-content:space-between;font-size:0.9rem;font-weight:600;margin-bottom:8px;'>"
            f"<span>{label}</span><span style='color:{color_end};'>{pct}%</span></div>"
            f"<div style='background:#1a2235;border-radius:8px;height:12px;box-shadow:inset 0 1px 3px rgba(0,0,0,0.3);'>"
            f"<div style='width:{pct}%;height:12px;border-radius:8px;background:linear-gradient(90deg, {color_start}, {color_end});box-shadow: 0 0 10px {color_end}40;'></div>"
            f"</div></div>"
        )

    with col_l:
        st.markdown("#### TOP RISK DRIVERS")
        drivers_html = (
            "<div class='glass-card' style='padding: 24px;'>"
            + driver_bar("Supplier Obligations",     sup_ratio,  "#FF5C6C", "#FF2A3A")
            + driver_bar("Customer Payment Delays",  delay_norm, "#FBBF24", "#F59E0B")
            + driver_bar("Expense vs Inflow",        exp_ratio,  "#FBBF24", "#F59E0B")
            + driver_bar("Upcoming EMI Obligations", emi_ratio,  "#A78BFA", "#8B5CF6")
            + "</div>"
        )
        st.markdown(drivers_html, unsafe_allow_html=True)

    with col_r:
        st.markdown("#### WHY IS THE RISK CHANGING?")
        st.markdown(
            """<div class='glass-card'>
              <div style='color:#19D3C5;font-weight:800;font-size:1.1rem;margin-bottom:4px;'>01</div>
              <div style='font-weight:700;margin-bottom:6px;'>SUPPLIER PRESSURE</div>
              <div style='color:#94A3B8;font-size:0.88rem;line-height:1.5;'>
                Supplier payments are consuming a large share of daily inflows.
                Any increase in procurement costs directly reduces your cash buffer.
              </div>
            </div>
            <div class='glass-card'>
              <div style='color:#FFB547;font-weight:800;font-size:1.1rem;margin-bottom:4px;'>02</div>
              <div style='font-weight:700;margin-bottom:6px;'>COLLECTION PRESSURE</div>
              <div style='color:#94A3B8;font-size:0.88rem;line-height:1.5;'>
                Customer payments arriving later than expected create a gap between
                outflows and inflows, shrinking your runway.
              </div>
            </div>
            <div class='glass-card'>
              <div style='color:#8B7CFF;font-weight:800;font-size:1.1rem;margin-bottom:4px;'>03</div>
              <div style='font-weight:700;margin-bottom:6px;'>EMI OBLIGATIONS</div>
              <div style='color:#94A3B8;font-size:0.88rem;line-height:1.5;'>
                Recurring loan EMI payments create fixed, unavoidable outflows.
                Without sufficient inflows on those dates, cash can turn negative.
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — TWIN ADVISOR
# ══════════════════════════════════════════════════════════════════════════════
elif page_name == "Twin Advisor":
    st.markdown("### 🤖 TWIN ADVISOR")
    st.markdown(
        "<p style='color:#94A3B8;margin-bottom:24px;font-size:1.05rem;'>Your AI financial co-pilot.</p>",
        unsafe_allow_html=True,
    )

    # Detect whether key exists (used only after call to set has_api for fallback label)
    try:
        has_api = bool(os.environ.get("GEMINI_API_KEY") or "GEMINI_API_KEY" in st.secrets)
    except Exception:
        has_api = bool(os.environ.get("GEMINI_API_KEY"))

    context = {
        "current_cash":   simulation["current_cash"],
        "runway_days":    simulation["scenario_runway"],
        "risk_score":     simulation["risk_score"],
        "risk_band":      simulation["risk_level"],
        "deficit_amount": simulation["projected_deficit"],
        "deficit_date":   simulation["deficit_date"],
    }

    try:
        with st.spinner("AI generating intelligence brief..."):
            ai_resp = get_explanation_and_recommendations(context)
    except Exception as e:
        print(f"[Twin Advisor] AI error: {e}")
        ai_resp = {
            "explanation": (
                f"Your business currently has ₹{current_cash/100000:,.2f}L in cash with "
                f"a runway of {b_runway} days. The risk score is {b_risk['score']:.0f}/100 "
                f"({b_risk['band']} risk). "
                "Key drivers are supplier payment obligations and upcoming EMI commitments. "
                "Proactive action now can significantly extend your financial runway."
            ),
            "actions": get_rule_based_recommendations(context),
            "gemini_used": False,
            "error": str(e)[:200],
        }

    # ── True status badge — rendered AFTER the call so it reflects reality ────
    gemini_used = ai_resp.get("gemini_used", False)
    api_error   = ai_resp.get("error")

    if gemini_used:
        badge_label = "Gemini Connected"
        badge_dot   = "#22C55E"
    elif has_api and api_error:
        badge_label = "Gemini API Error — Simulation Advisor Mode"
        badge_dot   = "#FF5C6C"
    elif has_api:
        badge_label = "Gemini API Error — Simulation Advisor Mode"
        badge_dot   = "#FBBF24"
    else:
        badge_label = "Simulation Advisor Mode"
        badge_dot   = "#FBBF24"

    st.markdown(
        f"<div class='badge badge-violet' style='margin-bottom:24px;border:1px solid #8B7CFF40;box-shadow: 0 0 12px rgba(139, 124, 255, 0.2);'>"
        f"<span class='sdot' style='background:{badge_dot};box-shadow:0 0 8px {badge_dot};'></span>"
        f"ANALYSIS COMPLETE &nbsp;|&nbsp; {badge_label}</div>",
        unsafe_allow_html=True,
    )

    # ── Safe error debug card (visible only when API failed with a key present) ─
    if api_error and not gemini_used and has_api:
        st.markdown(
            f"<div style='background:#1a0a0a;border:1px solid #FF5C6C33;border-radius:12px;"
            f"padding:16px;margin-bottom:24px;font-size:0.85rem;color:#FF5C6C;'>"
            f"<strong>DEBUG:</strong> {api_error}</div>",
            unsafe_allow_html=True,
        )

    # Explanation card
    explanation = ai_resp.get("explanation", "The financial twin has analyzed your situation.")
    st.markdown("#### WHAT I SEE")
    st.markdown(
        f"<div class='glass-card' style='border-top:3px solid #8B7CFF;"
        f"background:linear-gradient(180deg,rgba(139,124,255,.08) 0%,#101827 100%);"
        f"box-shadow: 0 4px 20px rgba(139,124,255,0.1); padding: 32px; border-radius: 12px; margin-bottom: 32px;'>"
        f"<p style='font-size:1.15rem;line-height:1.7;color:#F8FAFC;margin:0;letter-spacing: 0.3px;'>{explanation}</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Action cards
    st.markdown("#### WHAT I WOULD DO")
    actions = ai_resp.get("actions", [])
    for idx, action in enumerate(actions[:4]):
        st.markdown(
            f"<div class='glass-card' style='border-left:4px solid #8B7CFF; padding: 20px 24px; margin-bottom: 16px; transition: transform 0.2s ease; cursor: pointer;' onmouseover='this.style.transform=\"scale(1.02)\"' onmouseout='this.style.transform=\"scale(1)\"'>"
            f"<div style='display: flex; align-items: center; gap: 16px;'>"
            f"<div style='color:#8B7CFF;font-weight:800;font-size:1.6rem;opacity:0.8;'>0{idx+1}</div>"
            f"<div style='flex: 1;'>"
            f"<div style='font-weight:800;font-size:1.1rem;margin-bottom:6px;text-transform:uppercase;letter-spacing:1px;color:#F8FAFC;'>{action.get('action','')}</div>"
            f"<div style='color:#94A3B8;font-size:0.95rem;margin-bottom:8px;line-height:1.5;'>{action.get('problem','')}</div>"
            f"<div style='color:#36D6D0;font-weight:700;font-size:0.95rem;'>⚡ {action.get('impact','')}</div>"
            f"</div></div></div>",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — ACTION CENTER
# ══════════════════════════════════════════════════════════════════════════════
elif page_name == "Action Center":
    st.markdown("### 🎯 ACTION CENTER")
    st.markdown(
        "<p style='color:#94A3B8;margin-bottom:20px;'>Turn financial risk into concrete next steps.</p>",
        unsafe_allow_html=True,
    )

    context = {
        "current_cash":   simulation["current_cash"],
        "runway_days":    simulation["scenario_runway"],
        "risk_score":     simulation["risk_score"],
        "risk_band":      simulation["risk_level"],
        "deficit_amount": simulation["projected_deficit"],
        "deficit_date":   simulation["deficit_date"],
    }
    recs = get_rule_based_recommendations(context)

    col_cards, col_outcome = st.columns([3, 2])

    with col_cards:
        st.markdown("#### RECOMMENDED ACTIONS")
        for rec in recs:
            pri_emoji = rec.get("priority", "🟢")
            if   pri_emoji == "🔴": color, p_label = "#FF5C6C", "HIGH"
            elif pri_emoji == "🟡": color, p_label = "#FBBF24", "MEDIUM"
            else:                   color, p_label = "#4ADE80", "LOW"

            st.markdown(
                f"<div class='glass-card' style='border-left:4px solid {color}; padding: 20px 24px; margin-bottom: 16px; transition: transform 0.2s ease; cursor: pointer;' onmouseover='this.style.transform=\"scale(1.02)\"' onmouseout='this.style.transform=\"scale(1)\"'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>"
                f"<div style='background:{color}20; color:{color}; padding:4px 10px; border-radius:4px; font-size:0.75rem; font-weight:800; letter-spacing:1px;'>"
                f"{pri_emoji} {p_label} PRIORITY</div>"
                f"</div>"
                f"<div style='color:#94A3B8;font-size:0.85rem;margin-bottom:6px;font-weight:600;'>PROBLEM: {rec.get('problem','')}</div>"
                f"<div style='font-weight:800;font-size:1.15rem;margin-bottom:8px;color:#F8FAFC;'>ACTION: {rec.get('action','')}</div>"
                f"<div style='color:#36D6D0;font-weight:700;font-size:0.95rem;'>"
                f"⚡ EXPECTED IMPACT: {rec.get('impact','')}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    with col_outcome:
        st.markdown("#### IF YOU ACT NOW")
        improved_runway = int(simulation["scenario_runway"] * 1.25)
        b_band = simulation["risk_level"]
        b_band_col = band_color(b_band)
        st.markdown(
            f"<div class='glass-card' style='padding: 32px; background:linear-gradient(180deg, #101827 0%, #162032 100%);'>"
            f"<div style='color:#94A3B8;font-size:0.8rem;font-weight:700;letter-spacing:1.5px;margin-bottom:8px;'>WITHOUT ACTION</div>"
            f"<div style='font-size:2.4rem;font-weight:800;color:{b_band_col}; line-height: 1.1;'>{simulation['scenario_runway']} DAYS</div>"
            f"<div style='font-size:0.9rem;color:{b_band_col};margin-bottom:24px; font-weight:600;'>{b_band} RISK</div>"
            f"<hr style='border:none; border-top:1px dashed #334155; margin:24px 0;'/>"
            f"<div style='color:#94A3B8;font-size:0.8rem;font-weight:700;letter-spacing:1.5px;margin-bottom:8px;'>WITH RECOMMENDED ACTIONS</div>"
            f"<div style='font-size:2.8rem;font-weight:800;color:#4ADE80; line-height: 1.1; text-shadow: 0 0 20px rgba(74,222,128,0.3);'>{improved_runway} DAYS</div>"
            f"<div style='font-size:0.95rem;color:#4ADE80;margin-bottom:24px; font-weight:700; letter-spacing:1px;'>🟢 IMPROVED OUTLOOK</div>"
            f"<div style='font-size:0.75rem;color:#64748B;font-style:italic; line-height:1.4;'>*Illustrative scenario. "
            f"Actual results depend on business execution.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 7 — DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page_name == "Data":
    st.markdown("### 📂 BUSINESS DATA")
    st.markdown(
        "<p style='color:#94A3B8;margin-bottom:24px;font-size:1.05rem;'>Understand the data powering your Digital Twin.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='badge badge-teal' style='margin-bottom:24px;border:1px solid #19D3C540;box-shadow: 0 0 12px rgba(25, 211, 197, 0.2);'>"
        "<span class='sdot sdot-green'></span>DATA QUALITY: READY</div>",
        unsafe_allow_html=True,
    )

    total_in  = float(historical_df["customer_payments_in"].sum())
    total_out = float(
        historical_df["supplier_payments"].sum()
        + historical_df["payroll"].sum()
        + historical_df["rent"].sum()
        + historical_df["utilities"].sum()
        + historical_df["loan_emi"].sum()
        + historical_df["other_expenses"].sum()
    )
    
    # Use glass cards for data metrics
    st.markdown("<div class='glass-card' style='padding: 24px; margin-bottom: 32px;'>", unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.markdown(metric_html("TOTAL RECORDS",     str(len(df))),                          unsafe_allow_html=True)
    d2.markdown(metric_html("HISTORICAL DAYS",   str(len(historical_df))),               unsafe_allow_html=True)
    d3.markdown(metric_html("TOTAL INFLOW",      f"₹{total_in/100000:,.2f} L"),          unsafe_allow_html=True)
    d4.markdown(metric_html("TOTAL OUTFLOW",     f"₹{total_out/100000:,.2f} L"),         unsafe_allow_html=True)

    st.markdown("<hr style='border:none; border-top:1px dashed #334155; margin:16px 0;'/>", unsafe_allow_html=True)

    d5, d6 = st.columns(2)
    d5.markdown(metric_html("DATE RANGE (HIST)",
        f"{historical_df['date'].min().strftime('%d %b %Y')} → {historical_df['date'].max().strftime('%d %b %Y')}"),
        unsafe_allow_html=True)
    d6.markdown(metric_html("FORECAST PERIOD",
        f"{expected_df['date'].min().strftime('%d %b %Y')} → {expected_df['date'].max().strftime('%d %b %Y')}"),
        unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────
    # SAMPLE TEMPLATE + DATA UPLOAD
    # ──────────────────────────────────────────────────────────────────────
    st.markdown("#### UPLOAD NEW DATA")
    st.markdown(
        "<p style='color:#94A3B8;margin-top:-8px;'>"
        "Before uploading, download the sample template and structure your CSV using the same columns."
        "</p>",
        unsafe_allow_html=True,
    )

    req_cols = [
        "date", "type", "opening_balance", "customer_payments_in",
        "supplier_payments", "payroll", "rent", "utilities",
        "loan_emi", "other_expenses", "closing_balance",
    ]

    # A ready-to-download example that users can edit with their own values.
    sample_template = pd.DataFrame([
        {
            "date": "2026-09-01",
            "type": "historical",
            "opening_balance": 500000.00,
            "customer_payments_in": 75000.00,
            "supplier_payments": 25000.00,
            "payroll": 30000.00,
            "rent": 5000.00,
            "utilities": 2500.00,
            "loan_emi": 7500.00,
            "other_expenses": 3000.00,
            "closing_balance": 502000.00,
        },
        {
            "date": "2026-09-02",
            "type": "historical",
            "opening_balance": 502000.00,
            "customer_payments_in": 60000.00,
            "supplier_payments": 20000.00,
            "payroll": 0.00,
            "rent": 0.00,
            "utilities": 1800.00,
            "loan_emi": 0.00,
            "other_expenses": 2500.00,
            "closing_balance": 537700.00,
        },
        {
            "date": "2026-09-03",
            "type": "expected",
            "opening_balance": 537700.00,
            "customer_payments_in": 65000.00,
            "supplier_payments": 30000.00,
            "payroll": 0.00,
            "rent": 0.00,
            "utilities": 2000.00,
            "loan_emi": 0.00,
            "other_expenses": 3500.00,
            "closing_balance": 567200.00,
        },
    ], columns=req_cols)

    st.markdown(
        """
        <div style="
            border:1px solid rgba(45,212,191,.35);
            background:rgba(20,184,166,.06);
            border-radius:12px;
            padding:16px 18px;
            margin:12px 0 16px 0;">
            <div style="font-weight:800;color:#5EEAD4;font-size:1rem;">
                📄 CSV SAMPLE TEMPLATE
            </div>
            <div style="color:#94A3B8;font-size:.86rem;margin-top:6px;">
                Download this template, replace the example values with your business data,
                and upload the completed CSV. Keep all column names exactly the same.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    template_col, rules_col = st.columns([1.05, 1.95])

    with template_col:
        st.download_button(
            label="⬇️ Download Sample CSV Template",
            data=sample_template.to_csv(index=False).encode("utf-8"),
            file_name="cashflow_twin_data_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with rules_col:
        st.markdown(
            """
            <div style="color:#CBD5E1;font-size:.84rem;line-height:1.8;">
                <b style="color:#E2E8F0;">Quick rules:</b>
                <br>• <b>date</b>: YYYY-MM-DD
                <br>• <b>type</b>: historical or expected
                <br>• All money fields must contain numbers only
                <br>• Keep the exact column names from the template
                <br>• Do not change the CSV file format
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("👁️ Preview sample template columns and example data"):
        st.dataframe(sample_template, use_container_width=True, hide_index=True)

    uploaded = st.file_uploader(
        "Upload your completed CSV",
        type=["csv"],
        help="Download the sample template above and use the same schema.",
    )

    if uploaded:
        try:
            new_df = pd.read_csv(uploaded)

            # 1. Required column validation
            missing = [c for c in req_cols if c not in new_df.columns]
            extra = [c for c in new_df.columns if c not in req_cols]

            if missing:
                st.error(f"❌ Missing required columns: {', '.join(missing)}")
                st.info("Download the sample CSV template above and use its exact structure.")
            else:
                # Keep only the required columns and preserve the required order.
                new_df = new_df[req_cols].copy()

                # 2. Date validation
                new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce")
                invalid_dates = int(new_df["date"].isna().sum())

                # 3. Type validation
                new_df["type"] = new_df["type"].astype(str).str.strip().str.lower()
                invalid_types = int(
                    (~new_df["type"].isin(["historical", "expected"])).sum()
                )

                # 4. Numeric validation
                numeric_cols = [c for c in req_cols if c not in ["date", "type"]]
                numeric_errors = {}
                for col in numeric_cols:
                    converted = pd.to_numeric(new_df[col], errors="coerce")
                    numeric_errors[col] = int(converted.isna().sum())
                    new_df[col] = converted

                invalid_numeric = sum(numeric_errors.values())

                if invalid_dates > 0:
                    st.error(
                        f"❌ {invalid_dates} invalid date value(s) found. "
                        "Use the format YYYY-MM-DD."
                    )
                elif invalid_types > 0:
                    st.error(
                        "❌ Invalid values found in the 'type' column. "
                        "Use only: historical or expected."
                    )
                elif invalid_numeric > 0:
                    bad_cols = [
                        col for col, count in numeric_errors.items() if count > 0
                    ]
                    st.error(
                        "❌ Invalid or empty numeric values found in: "
                        + ", ".join(bad_cols)
                    )
                else:
                    new_df["date"] = new_df["date"].dt.strftime("%Y-%m-%d")

                    if extra:
                        st.warning(
                            "⚠️ Extra columns were found and ignored: "
                            + ", ".join(extra)
                        )

                    # Show the user exactly what will be saved.
                    st.success(
                        f"✅ CSV validated successfully — {len(new_df)} record(s) ready to use."
                    )
                    st.dataframe(
                        new_df.head(10),
                        use_container_width=True,
                        hide_index=True,
                    )

                    if st.button(
                        "🚀 Apply Uploaded Data to Digital Twin",
                        key="apply_uploaded_data",
                        use_container_width=True,
                    ):
                        new_df.to_csv(DATA_PATH, index=False)
                        st.cache_data.clear()
                        st.success(
                            "✅ New data applied successfully. Reloading the Digital Twin..."
                        )
                        st.rerun()

        except Exception as e:
            print(f"[Data Upload] Error: {e}")
            st.error(f"❌ Upload failed: {e}")

    st.markdown("#### DATA PREVIEW")
    st.dataframe(df.head(50), use_container_width=True)
