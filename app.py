"""
╔══════════════════════════════════════════════════════════╗
║          SpendWise AI — Expense Intelligence App         ║
║   Streamlit · Pandas · Scikit-learn · Plotly             ║
╚══════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

import io
import base64
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    clean_dataframe,
    train_categoriser,
    predict_categories,
    detect_anomalies,
    forecast_spending,
    generate_insights,
    budget_health_score,
)
from data_gen import generate_transactions
from dotenv import load_dotenv
import os
load_dotenv()
# Then in session state init:
# st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
# ── New feature modules (Features 1–8) ───────────────────────
from utils.recommendations  import generate_recommendations
from utils.goals            import calculate_goal_status
from utils.anomaly_alerts   import build_anomaly_alerts, weekly_anomaly_summary
from utils.health_breakdown import compute_health_breakdown
from utils.benchmark        import compare_with_benchmark, benchmark_summary_line
from utils.persistence      import save_session, list_sessions, load_session, delete_session
from utils.chatbot          import answer_query
from utils.gemini_chat      import gemini_chat 

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SpendWise AI",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  CUSTOM CSS  — dark fintech aesthetic
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=Inter:wght@300;400;500;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0d14;
    color: #e2e8f0;
}
.stApp { background: #0a0d14; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f1420 !important;
    border-right: 1px solid #1e2535;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
    border: 1px solid #1e2d45;
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
div[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #f0f9ff !important;
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem !important;
    font-weight: 800;
}

/* ── Headers ── */
h1 { font-family: 'Syne', sans-serif !important; font-size: 2.4rem !important;
     background: linear-gradient(90deg, #38bdf8, #818cf8);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
h2, h3 { font-family: 'Syne', sans-serif !important; color: #cbd5e1 !important; }

/* ── Plotly container ── */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* ── Insight cards ── */
.insight-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border-left: 3px solid #38bdf8;
    border-radius: 8px;
    padding: 12px 18px;
    margin-bottom: 10px;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ── Button ── */
.stDownloadButton > button {
    background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.4rem !important;
    transition: opacity .2s;
}
.stDownloadButton > button:hover { opacity: 0.85; }

/* ── Anomaly table ── */
.anomaly-row { color: #f87171; font-weight: 600; }

/* ── Divider ── */
hr { border-color: #1e2535 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  SESSION STATE — persist data across reruns
# ══════════════════════════════════════════════════════════════
if "df" not in st.session_state:
    st.session_state.df            = None
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df   = None
if "insights" not in st.session_state:
    st.session_state.insights      = None
if "health" not in st.session_state:
    st.session_state.health        = None
if "model_acc" not in st.session_state:
    st.session_state.model_acc     = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded   = False
if "monthly_budget" not in st.session_state:
    st.session_state.monthly_budget = 50_000

# ── New feature session keys ──────────────────────────────────
if "recommendations"  not in st.session_state:
    st.session_state.recommendations  = []
if "goal_data"        not in st.session_state:
    st.session_state.goal_data        = None
if "anomaly_alerts"   not in st.session_state:
    st.session_state.anomaly_alerts   = []
if "weekly_anomalies" not in st.session_state:
    st.session_state.weekly_anomalies = []
if "health_breakdown" not in st.session_state:
    st.session_state.health_breakdown = None
if "benchmarking"     not in st.session_state:
    st.session_state.benchmarking     = []
if "chat_history"     not in st.session_state:
    st.session_state.chat_history     = []
if "gemini_api_key"   not in st.session_state:
    st.session_state.gemini_api_key   = ""
if "gemini_enabled"   not in st.session_state:
    st.session_state.gemini_enabled   = False
if "gemini_model"     not in st.session_state:
    st.session_state.gemini_model     = "gemini-2.0-flash"

# ══════════════════════════════════════════════════════════════
#  SIDEBAR — Controls
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    data_source = st.radio(
        "Data Source",
        ["Generate Synthetic Data", "Upload CSV"],
        index=0,
    )

    uploaded = None
    if data_source == "Upload CSV":
        uploaded = st.file_uploader(
            "📂 Upload transactions.csv",
            type=["csv"],
            help="Required columns: Date, Description, Amount, Payment_Mode",
        )
        if uploaded is not None:
            # Preview uploaded file immediately
            preview = pd.read_csv(uploaded)
            uploaded.seek(0)   # reset pointer for later processing
            st.success(f"✅ {len(preview):,} rows detected")
            with st.expander("👀 Preview (first 5 rows)"):
                st.dataframe(preview.head(), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎛️ Settings")
    monthly_budget = st.number_input(
        "Monthly Budget (₹)",
        min_value=5_000, max_value=5_00_000,
        value=st.session_state.monthly_budget, step=1_000,
    )
    contamination = st.slider(
        "Anomaly Sensitivity", 0.01, 0.15, 0.05, 0.01,
        help="Higher = more transactions flagged as anomalies",
    )
    forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30, 7)

    st.markdown("---")

    # ── SUBMIT BUTTON ────────────────────────────────────────
    ready = (data_source == "Generate Synthetic Data") or (uploaded is not None)

    submit = st.button(
        "🚀 Analyse My Spending",
        use_container_width=True,
        disabled=not ready,
        type="primary",
    )

    if data_source == "Upload CSV" and uploaded is None:
        st.caption("⬆️ Upload a CSV file to enable analysis.")

    st.markdown("---")

    # ── Reset button ─────────────────────────────────────────
    if st.button("🔄 Reset Dashboard", use_container_width=True):
        for key in ["df", "forecast_df", "insights", "health", "model_acc", "data_loaded"]:
            st.session_state[key] = None
        st.session_state.data_loaded = False
        st.rerun()

    # ── Gemini API Key ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 Gemini AI")
    gemini_key_input = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=st.session_state.gemini_api_key,
        placeholder="Paste your key from aistudio.google.com",
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )
    gemini_model_input = st.selectbox(
        "Gemini Model",
        ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"].index(
            st.session_state.gemini_model
        ),
    )
    if st.button("🔑 Connect Gemini", use_container_width=True):
        if gemini_key_input.strip():
            ok = (gemini_key_input.strip())
            if ok:
                st.session_state.gemini_api_key = gemini_key_input.strip()
                st.session_state.gemini_model   = gemini_model_input
                st.session_state.gemini_enabled = True
                st.success("✅ Gemini connected!")
            else:
                st.session_state.gemini_enabled = False
                st.error("❌ Could not connect. Check your API key.")
        else:
            st.warning("⬆️ Paste your Gemini API key first.")

    if st.session_state.gemini_enabled:
        st.markdown(
            "<small style='color:#22c55e'>● Gemini AI active</small>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<small style='color:#64748b'>● Using rule-based chatbot</small>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<small style='color:#475569'>SpendWise AI v2.0 · Built with ❤️ using "
        "Streamlit · Scikit-learn · Plotly</small>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════
#  DATA LOADING — only runs when Submit is clicked
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_and_process(source: str, budget: float, cont: float, fdays: int, _file=None):
    if source == "Upload CSV" and _file is not None:
        raw = pd.read_csv(_file)
    else:
        raw = generate_transactions(600)

    df = clean_dataframe(raw)

    # Guard: clean_dataframe drops rows with unparseable dates/amounts.
    # If nothing survives, raise a user-friendly error immediately
    # so the empty-vocabulary crash in TfidfVectorizer never happens.
    if len(df) == 0:
        raise ValueError(
            "\n\n🚨 SpendWise AI — Empty Data After Cleaning\n"
            "────────────────────────────────────────────────\n"
            "No valid transactions remain after cleaning your file.\n\n"
            "Common reasons:\n"
            "  • Date column values are not recognisable dates\n"
            "    (e.g. '01/31/24' is fine, 'N/A' or '--' is not)\n"
            "  • Amount column contains only text or blanks\n"
            "  • All rows were filtered out during header detection\n\n"
            "Tip: Open your CSV, check that at least one row has a valid\n"
            "date (any common format) and a numeric amount, then re-upload."
        )

    pipeline, le, acc = train_categoriser(df)
    df = predict_categories(df, pipeline, le)
    df = detect_anomalies(df, contamination=cont)
    forecast, slope = forecast_spending(df, days_ahead=fdays)
    insights = generate_insights(df, forecast)
    health   = budget_health_score(df, monthly_budget=budget)
    return df, forecast, insights, health, acc


if submit:
    st.session_state.monthly_budget = monthly_budget
    # NEW ▸ Wrap entire pipeline in try/except so column-mapping
    #        errors surface as a clean Streamlit message, not a
    #        raw Python traceback. No other logic changed.
    try:
        with st.spinner("🧠 Running AI models — this takes a few seconds..."):
            (
                st.session_state.df,
                st.session_state.forecast_df,
                st.session_state.insights,
                st.session_state.health,
                st.session_state.model_acc,
            ) = load_and_process(
                data_source, monthly_budget, contamination, forecast_days,
                _file=uploaded,
            )
        st.session_state.data_loaded = True
    except ValueError as col_err:
        # Column-mapping error from _resolve_columns()
        st.error(str(col_err))
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error while processing your file:\n\n{e}")
        st.stop()

    # ── Compute new features immediately after core pipeline ────
    _df  = st.session_state.df
    _bud = monthly_budget

    st.session_state.recommendations  = generate_recommendations(_df, _bud)
    st.session_state.anomaly_alerts   = build_anomaly_alerts(_df)
    st.session_state.weekly_anomalies = weekly_anomaly_summary(_df)
    st.session_state.health_breakdown = compute_health_breakdown(_df, _bud)
    st.session_state.benchmarking     = compare_with_benchmark(_df, _bud)

    # Auto-save session to SQLite
    meta = {
        "monthly_budget": _bud,
        "total_spend":    float(_df["Amount"].sum()),
        "rows":           len(_df),
    }
    save_session(_df, meta, label=f"Session {pd.Timestamp.now().strftime('%d %b %Y %H:%M')}")

    st.toast("✅ Analysis complete!", icon="🎉")

# ── Guard: show landing screen if no data yet ─────────────────
if not st.session_state.data_loaded:
    st.markdown("# 💸 SpendWise AI")
    st.markdown(
        "<p style='color:#64748b;font-size:1rem;'>"
        "Expense Intelligence · ML Categorisation · Anomaly Detection · Forecasting"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    cola, colb, colc = st.columns(3)
    cola.info("**Step 1**\n\nChoose *Generate Synthetic Data* or upload your own CSV in the sidebar.")
    colb.info("**Step 2**\n\nAdjust your monthly budget and settings to match your situation.")
    colc.info("**Step 3**\n\nClick **🚀 Analyse My Spending** to run the AI models and see your dashboard.")
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#334155;font-size:0.85rem'>"
        "Supports CSV files with columns: <code>Date · Description · Amount · Payment_Mode</code>"
        "</p>",
        unsafe_allow_html=True,
    )
    st.stop()   # ← nothing below renders until Submit is clicked

# ── Pull from session state for the rest of the dashboard ─────
df           = st.session_state.df
forecast_df  = st.session_state.forecast_df
insights     = st.session_state.insights
health       = st.session_state.health
model_acc    = st.session_state.model_acc

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
h_left, h_right = st.columns([3, 1])
with h_left:
    st.markdown("# 💸 SpendWise AI")
    st.markdown(
        "<p style='color:#64748b;font-size:0.95rem;margin-top:-14px'>"
        "Expense Intelligence · ML Categorisation · Anomaly Detection · Forecasting"
        "</p>",
        unsafe_allow_html=True,
    )
with h_right:
    source_label = "📂 Uploaded CSV" if data_source == "Upload CSV" else "🤖 Synthetic Data"
    st.markdown(f"<br><span style='color:#38bdf8;font-size:0.85rem'>Source: {source_label}</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:#64748b;font-size:0.8rem'>{len(df):,} transactions loaded</span>", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  ROW 1 — METRIC CARDS
# ══════════════════════════════════════════════════════════════
total_spend    = df["Amount"].sum()
avg_daily      = total_spend / max(df["Date"].nunique(), 1)
n_anomalies    = int(df["Is_Anomaly"].sum())
forecast_total = forecast_df["Predicted_Spend"].sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("💰 Total Spend",         f"₹{total_spend:,.0f}")
c2.metric("📅 Daily Average",       f"₹{avg_daily:,.0f}")
c3.metric("🚨 Anomalies Found",     n_anomalies)
c4.metric(f"🔮 {forecast_days}d Forecast", f"₹{forecast_total:,.0f}")
c5.metric("🤖 Model Accuracy",      f"{model_acc}%" if str(model_acc) != "nan" else "N/A (small dataset)")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  ROW 2 — BUDGET HEALTH GAUGE  +  CATEGORY PIE
# ══════════════════════════════════════════════════════════════
col_gauge, col_pie = st.columns([1, 2])

with col_gauge:
    st.markdown("### 🏥 Budget Health")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health["score"],
        delta={"reference": 70, "valueformat": ".0f"},
        title={"text": health["status"], "font": {"size": 15, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155"},
            "bar":  {"color": health["color"]},
            "bgcolor": "#111827",
            "bordercolor": "#1e293b",
            "steps": [
                {"range": [0, 30],   "color": "#1f1225"},
                {"range": [30, 60],  "color": "#1a1f2e"},
                {"range": [60, 100], "color": "#0f1a24"},
            ],
        },
        number={"font": {"color": "#e2e8f0", "size": 42}},
    ))
    gauge.update_layout(
        height=260, paper_bgcolor="#0a0d14", font_color="#e2e8f0",
        margin=dict(l=20, r=20, t=40, b=0),
    )
    st.plotly_chart(gauge, use_container_width=True)
    st.caption(
        f"Monthly spend: ₹{health['monthly_spend']:,.0f} / "
        f"₹{health['monthly_budget']:,.0f} budget ({health['ratio']}%)"
    )

with col_pie:
    st.markdown("### 🗂️ Spending by Category")
    cat_df = (
        df.groupby("Category")["Amount"]
          .sum()
          .reset_index()
          .sort_values("Amount", ascending=False)
    )
    pie = px.pie(
        cat_df, names="Category", values="Amount",
        hole=0.52,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    pie.update_traces(
        textposition="outside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>",
    )
    pie.update_layout(
        paper_bgcolor="#0a0d14", plot_bgcolor="#0a0d14",
        font_color="#cbd5e1",
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
        height=310, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(pie, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  ROW 3 — SPENDING TREND  +  FORECAST
# ══════════════════════════════════════════════════════════════
st.markdown("### 📈 Spending Trend & 30-Day Forecast")

daily_actual = (
    df.groupby("Date")["Amount"]
      .sum()
      .reset_index()
      .rename(columns={"Amount": "Spend"})
)

trend_fig = go.Figure()

# Actual spend area
trend_fig.add_trace(go.Scatter(
    x=daily_actual["Date"], y=daily_actual["Spend"],
    name="Actual Spend", mode="lines",
    line=dict(color="#38bdf8", width=2),
    fill="tozeroy",
    fillcolor="rgba(56,189,248,0.08)",
    hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.0f}<extra></extra>",
))

# Forecast
trend_fig.add_trace(go.Scatter(
    x=forecast_df["Date"], y=forecast_df["Predicted_Spend"],
    name="Forecast", mode="lines",
    line=dict(color="#818cf8", width=2, dash="dot"),
    hovertemplate="<b>%{x|%d %b %Y}</b> (forecast)<br>₹%{y:,.0f}<extra></extra>",
))

# Anomaly markers
anomalies = df[df["Is_Anomaly"]]
anom_daily = anomalies.groupby("Date")["Amount"].sum().reset_index()
trend_fig.add_trace(go.Scatter(
    x=anom_daily["Date"], y=anom_daily["Amount"],
    name="Anomaly", mode="markers",
    marker=dict(color="#f87171", size=9, symbol="x", line=dict(width=2, color="#fca5a5")),
    hovertemplate="<b>%{x|%d %b %Y}</b> ⚠️<br>₹%{y:,.0f}<extra></extra>",
))

trend_fig.update_layout(
    paper_bgcolor="#0a0d14", plot_bgcolor="#0f1420",
    font_color="#94a3b8",
    xaxis=dict(gridcolor="#1e2535", showgrid=True),
    yaxis=dict(gridcolor="#1e2535", showgrid=True, tickprefix="₹"),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08),
    height=360,
    margin=dict(l=0, r=0, t=30, b=0),
    hovermode="x unified",
)
st.plotly_chart(trend_fig, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  ROW 4 — CATEGORY BAR  +  PAYMENT MODE
# ══════════════════════════════════════════════════════════════
col_bar, col_pay = st.columns([3, 2])

with col_bar:
    st.markdown("### 🏷️ Category Breakdown")
    cat_bar = px.bar(
        cat_df.sort_values("Amount"),
        x="Amount", y="Category", orientation="h",
        color="Amount",
        color_continuous_scale="Blues",
        text_auto=".2s",
    )
    cat_bar.update_traces(
        textfont_color="#e2e8f0",
        hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}<extra></extra>",
    )
    cat_bar.update_layout(
        paper_bgcolor="#0a0d14", plot_bgcolor="#0f1420",
        font_color="#94a3b8",
        xaxis=dict(gridcolor="#1e2535", tickprefix="₹"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        coloraxis_showscale=False,
        height=340, margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(cat_bar, use_container_width=True)

with col_pay:
    st.markdown("### 💳 Payment Mode Split")
    pay_df = df["Payment_Mode"].value_counts().reset_index()
    pay_df.columns = ["Mode", "Count"]
    pay_bar = px.bar(
        pay_df, x="Count", y="Mode", orientation="h",
        color="Mode",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    pay_bar.update_layout(
        paper_bgcolor="#0a0d14", plot_bgcolor="#0f1420",
        font_color="#94a3b8",
        xaxis=dict(gridcolor="#1e2535"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
        height=340, margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(pay_bar, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  ROW 5 — ANOMALY TABLE
# ══════════════════════════════════════════════════════════════
st.markdown(f"### 🚨 Flagged Anomalies ({n_anomalies} transactions)")
if n_anomalies:
    anom_view = (
        df[df["Is_Anomaly"]]
          [["Date", "Description", "Amount", "Category", "Payment_Mode"]]
          .sort_values("Amount", ascending=False)
          .head(20)
    )
    anom_view["Date"] = anom_view["Date"].dt.strftime("%d %b %Y")
    anom_view["Amount"] = anom_view["Amount"].apply(lambda x: f"₹{x:,.2f}")
    st.dataframe(
        anom_view.reset_index(drop=True),
        use_container_width=True,
        height=min(36 * len(anom_view) + 40, 380),
    )
else:
    st.info("✅ No anomalies detected at current sensitivity level.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  ROW 6 — AI INSIGHTS
# ══════════════════════════════════════════════════════════════
st.markdown("### 🧠 AI-Generated Insights")
for tip in insights:
    st.markdown(f"<div class='insight-card'>{tip}</div>", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  ROW 7 — MONTHLY HEATMAP
# ══════════════════════════════════════════════════════════════
st.markdown("### 🗓️ Monthly Spending Heatmap")
heatmap_df = (
    df.groupby(["Month", "Category"])["Amount"]
      .sum()
      .reset_index()
      .pivot(index="Category", columns="Month", values="Amount")
      .fillna(0)
)
hm = px.imshow(
    heatmap_df,
    color_continuous_scale="Blues",
    aspect="auto",
    text_auto=".2s",
)
hm.update_layout(
    paper_bgcolor="#0a0d14", plot_bgcolor="#0f1420",
    font_color="#94a3b8",
    height=360, margin=dict(l=0, r=0, t=0, b=0),
    coloraxis_colorbar=dict(tickprefix="₹"),
)
st.plotly_chart(hm, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  ROW 8 — RAW DATA EXPLORER
# ══════════════════════════════════════════════════════════════
st.markdown("### 🔍 Transaction Explorer")

# Filter controls
fe1, fe2, fe3 = st.columns(3)
with fe1:
    search_term = st.text_input("🔎 Search description", placeholder="e.g. Swiggy, Amazon...")
with fe2:
    all_cats = ["All"] + sorted(df["Category"].dropna().unique().tolist())
    filter_cat = st.selectbox("Filter by Category", all_cats)
with fe3:
    filter_anomaly = st.selectbox("Filter by Type", ["All", "Normal only", "Anomalies only"])

# Apply filters
filtered = df.copy()
if search_term:
    filtered = filtered[filtered["Description"].str.contains(search_term, case=False, na=False)]
if filter_cat != "All":
    filtered = filtered[filtered["Category"] == filter_cat]
if filter_anomaly == "Anomalies only":
    filtered = filtered[filtered["Is_Anomaly"]]
elif filter_anomaly == "Normal only":
    filtered = filtered[~filtered["Is_Anomaly"]]

# Display table
view_cols = ["Date", "Description", "Amount", "Category", "Payment_Mode", "Is_Anomaly"]
display_df = filtered[view_cols].copy()
display_df["Date"]   = display_df["Date"].dt.strftime("%d %b %Y")
display_df["Amount"] = display_df["Amount"].apply(lambda x: f"₹{x:,.2f}")
display_df = display_df.rename(columns={"Is_Anomaly": "⚠️ Anomaly"})

st.caption(f"Showing **{len(display_df):,}** of {len(df):,} transactions")
st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=320)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  ROW 9 — EXPORT
# ══════════════════════════════════════════════════════════════
st.markdown("### 📥 Export Report")
st.caption("All exports reflect current filter selections where applicable.")

exp1, exp2, exp3 = st.columns(3)

# CSV export
csv_bytes = df.drop(columns=["Anomaly_Score"], errors="ignore").to_csv(index=False).encode()
exp1.download_button(
    "⬇️ Download Full CSV",
    data=csv_bytes,
    file_name="spendwise_transactions.csv",
    mime="text/csv",
)

# Anomaly CSV
anom_csv = df[df["Is_Anomaly"]].drop(columns=["Anomaly_Score"], errors="ignore").to_csv(index=False).encode()
exp2.download_button(
    "⚠️ Download Anomalies CSV",
    data=anom_csv,
    file_name="spendwise_anomalies.csv",
    mime="text/csv",
)

# Forecast CSV
fc_csv = forecast_df.to_csv(index=False).encode()
exp3.download_button(
    "🔮 Download Forecast CSV",
    data=fc_csv,
    file_name="spendwise_forecast.csv",
    mime="text/csv",
)

# ── Full HTML Report ──────────────────────────────────────────
html_report = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SpendWise AI Report</title>
<style>
  body {{ font-family: Arial, sans-serif; background: #f8fafc; color: #1e293b; margin: 40px; }}
  h1 {{ color: #0ea5e9; }} h2 {{ color: #334155; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 24px; font-size: 0.85rem; }}
  th {{ background: #0ea5e9; color: white; padding: 8px 12px; text-align: left; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #e2e8f0; }}
  tr:nth-child(even) {{ background: #f1f5f9; }}
  .metric {{ display: inline-block; background: #0f172a; color: #e2e8f0;
             padding: 16px 24px; border-radius: 10px; margin: 8px; text-align: center; }}
  .metric h3 {{ margin: 0; font-size: 0.75rem; color: #94a3b8; }}
  .metric p  {{ margin: 4px 0 0; font-size: 1.6rem; font-weight: 700; }}
  .insight {{ background: #f0f9ff; border-left: 4px solid #0ea5e9;
              padding: 10px 16px; margin: 8px 0; border-radius: 4px; }}
</style>
</head>
<body>
<h1>💸 SpendWise AI — Expense Intelligence Report</h1>
<p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%d %b %Y %H:%M')}</p>

<h2>📊 Summary Metrics</h2>
<div class="metric"><h3>Total Spend</h3><p>₹{total_spend:,.0f}</p></div>
<div class="metric"><h3>Daily Average</h3><p>₹{avg_daily:,.0f}</p></div>
<div class="metric"><h3>Anomalies</h3><p>{n_anomalies}</p></div>
<div class="metric"><h3>Budget Health</h3><p>{health['status']}</p></div>
<div class="metric"><h3>Model Accuracy</h3><p>{model_acc}%</p></div>

<h2>🧠 AI Insights</h2>
{''.join(f'<div class="insight">{i}</div>' for i in insights)}

<h2>🗂️ Category Summary</h2>
{cat_df.to_html(index=False, float_format='₹{:,.2f}'.format)}

<h2>🔮 30-Day Forecast (first 10 rows)</h2>
{forecast_df.head(10).to_html(index=False)}

<h2>🚨 Anomalous Transactions</h2>
{df[df['Is_Anomaly']][['Date','Description','Amount','Category']].head(20).to_html(index=False)}
</body>
</html>
"""
st.download_button(
    "📄 Download Full HTML Report",
    data=html_report.encode(),
    file_name="spendwise_report.html",
    mime="text/html",
)


# ══════════════════════════════════════════════════════════════
#  FEATURE 1 — SMART RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
st.markdown("### 💡 Smart Recommendations")
recs = st.session_state.recommendations
if recs:
    priority_colors = {"high": "#ef4444", "medium": "#f97316", "low": "#22c55e"}
    priority_icons  = {"high": "🔴", "medium": "🟠", "low": "🟢"}
    total_saving_potential = sum(r["saving_potential"] for r in recs if r["saving_potential"] > 0)
    st.caption(f"**{len(recs)} recommendations** found · Estimated savings potential: **₹{total_saving_potential:,.0f}/month**")
    for rec in recs:
        color = priority_colors[rec["priority"]]
        icon  = priority_icons[rec["priority"]]
        badge = f"<span style='background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:0.7rem;font-weight:700'>{rec['priority'].upper()}</span>"
        st.markdown(
            f"<div class='insight-card' style='border-left-color:{color}'>"
            f"{icon} {badge} &nbsp;<strong>{rec['category']}</strong><br>{rec['message']}"
            + (f"<br><small style='color:#64748b'>💰 Potential saving: ₹{rec['saving_potential']:,.0f}/month</small>" if rec["saving_potential"] > 0 else "")
            + "</div>",
            unsafe_allow_html=True,
        )
else:
    st.success("✅ No major overspending patterns detected. Keep it up!")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  FEATURE 2 — GOAL-BASED FINANCIAL PLANNING
# ══════════════════════════════════════════════════════════════
st.markdown("### 🎯 Financial Goal Planner")
g1, g2 = st.columns(2)
with g1:
    goal_target = st.number_input("🏆 Target Savings Amount (₹)", min_value=1_000, max_value=50_00_000, value=1_00_000, step=5_000)
with g2:
    goal_months = st.slider("⏱️ Time to achieve (months)", 1, 60, 12)

if st.button("📊 Calculate Goal Feasibility", type="secondary"):
    goal_result = calculate_goal_status(df, monthly_budget, goal_target, goal_months)
    st.session_state.goal_data = goal_result

if st.session_state.goal_data:
    g = st.session_state.goal_data
    gm1, gm2, gm3 = st.columns(3)
    gm1.metric("Required/Month",  f"₹{g['required_monthly']:,.0f}")
    gm2.metric("Current Savings", f"₹{g['avg_monthly_savings']:,.0f}/month")
    gm3.metric("Feasibility",     "✅ Yes" if g["feasible"] else "❌ Not Yet")

    st.progress(min(int(g["progress_pct"]), 100), text=f"Progress: {g['progress_pct']}% of ₹{g['target_savings']:,.0f} goal (estimated from savings trend)")

    if g["feasible"]:
        st.success(g["recommendation"])
    else:
        st.warning(g["recommendation"])
        st.caption(f"Gap: ₹{g['gap']:,.0f}/month extra savings needed.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  FEATURE 4 — ADVANCED ANOMALY ALERT SYSTEM
# ══════════════════════════════════════════════════════════════
st.markdown("### 🚨 Advanced Anomaly Alerts")
alerts = st.session_state.anomaly_alerts
weekly = st.session_state.weekly_anomalies

sev_colors = {"critical": "#ef4444", "high": "#f97316", "medium": "#eab308"}
sev_icons  = {"critical": "🔴", "high": "🟠", "medium": "🟡"}

if alerts:
    aa1, aa2 = st.columns([2, 1])
    with aa1:
        st.caption(f"**{len(alerts)} anomalous transactions** with explanations:")
        for a in alerts[:10]:
            c = sev_colors.get(a["severity"], "#64748b")
            i = sev_icons.get(a["severity"], "⚪")
            date_str = a["date"].strftime("%d %b %Y") if hasattr(a["date"], "strftime") else str(a["date"])
            st.markdown(
                f"<div class='insight-card' style='border-left-color:{c}'>"
                f"{i} <strong>{a['description']}</strong> — ₹{a['amount']:,.0f} "
                f"<small style='color:#64748b'>({date_str} · {a['category']})</small><br>"
                f"<small>🔍 {a['reason']}</small></div>",
                unsafe_allow_html=True,
            )
    with aa2:
        st.caption("📅 Weekly Summary")
        if weekly:
            wk_df = pd.DataFrame(weekly)
            wk_df["total_amount"] = wk_df["total_amount"].apply(lambda x: f"₹{x:,.0f}")
            wk_df.columns = ["Week", "Count", "Total", "Top Category"]
            st.dataframe(wk_df, use_container_width=True, height=280)
        else:
            st.info("No weekly data available.")
else:
    st.success("✅ No anomalies to alert. Spending looks normal.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  FEATURE 5 — EXPLAINABLE HEALTH SCORE
# ══════════════════════════════════════════════════════════════
st.markdown("### 🔬 Explainable Financial Health Score")
breakdown = st.session_state.health_breakdown
if breakdown:
    comp_labels = {
        "savings_ratio":     "💰 Savings Ratio",
        "expense_stability": "📊 Expense Stability",
        "essential_ratio":   "🛒 Essential vs Non-Essential",
        "investment_ratio":  "📈 Investment Ratio",
    }
    bh1, bh2 = st.columns([1, 2])
    with bh1:
        fig_score = go.Figure(go.Indicator(
            mode="gauge+number",
            value=breakdown["final_score"],
            title={"text": breakdown["final_label"], "font": {"size": 13, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#334155"},
                "bar":  {"color": breakdown["final_color"]},
                "bgcolor": "#111827",
                "bordercolor": "#1e293b",
                "steps": [
                    {"range": [0, 45],  "color": "#1f1225"},
                    {"range": [45, 65], "color": "#1a1f2e"},
                    {"range": [65, 100],"color": "#0f1a24"},
                ],
            },
            number={"font": {"color": "#e2e8f0", "size": 38}},
        ))
        fig_score.update_layout(height=220, paper_bgcolor="#0a0d14", margin=dict(l=10,r=10,t=30,b=0))
        st.plotly_chart(fig_score, use_container_width=True)

    with bh2:
        for key, comp in breakdown["components"].items():
            label = comp_labels.get(key, key)
            score = comp["score"]
            wt    = int(comp["weight"] * 100)
            bar_color = "#22c55e" if score >= 75 else "#eab308" if score >= 50 else "#ef4444"
            st.markdown(
                f"<div style='margin-bottom:10px'>"
                f"<div style='display:flex;justify-content:space-between'>"
                f"<span style='font-size:0.85rem'>{label}</span>"
                f"<span style='color:#64748b;font-size:0.75rem'>{comp['label']} · weight {wt}%</span>"
                f"</div>"
                f"<div style='background:#1e293b;border-radius:6px;height:8px;margin-top:4px'>"
                f"<div style='background:{bar_color};width:{score}%;height:8px;border-radius:6px'></div>"
                f"</div>"
                f"<small style='color:#475569'>{comp['description']}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  FEATURE 6 — BENCHMARKING
# ══════════════════════════════════════════════════════════════
st.markdown("### 📊 Peer Benchmarking")
bench_data = st.session_state.benchmarking
if bench_data:
    st.caption(benchmark_summary_line(bench_data))
    bench_df = pd.DataFrame(bench_data)
    bench_df["user_monthly"]      = bench_df["user_monthly"].apply(lambda x: f"₹{x:,.0f}")
    bench_df["benchmark_monthly"] = bench_df["benchmark_monthly"].apply(lambda x: f"₹{x:,.0f}")
    bench_df["diff_pct"]          = bench_df["diff_pct"].apply(lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%")
    bench_df["status"]            = bench_df["status"].map({"above": "⬆️ Above avg", "below": "⬇️ Below avg", "on_par": "✅ On par"})
    bench_df.columns = ["Category", "Your Monthly", "Peer Average", "Difference", "Status"]

    # Bar chart: user vs benchmark
    raw_bench = pd.DataFrame(st.session_state.benchmarking)
    bench_fig = go.Figure()
    bench_fig.add_trace(go.Bar(name="You", x=raw_bench["category"], y=raw_bench["user_monthly"],
                               marker_color="#38bdf8"))
    bench_fig.add_trace(go.Bar(name="Peer Avg", x=raw_bench["category"], y=raw_bench["benchmark_monthly"],
                               marker_color="#818cf8", opacity=0.7))
    bench_fig.update_layout(
        barmode="group", paper_bgcolor="#0a0d14", plot_bgcolor="#0f1420",
        font_color="#94a3b8", height=300, margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(gridcolor="#1e2535"), yaxis=dict(gridcolor="#1e2535", tickprefix="₹"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(bench_fig, use_container_width=True)
    st.dataframe(bench_df.reset_index(drop=True), use_container_width=True, height=320)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  FEATURE 7 — SESSION PERSISTENCE
# ══════════════════════════════════════════════════════════════
st.markdown("### 💾 Saved Sessions")
sessions = list_sessions()
if sessions:
    sess_df = pd.DataFrame(sessions)
    sess_df.columns = ["ID", "Saved At", "Label"]
    st.dataframe(sess_df, use_container_width=True, height=180)

    col_load, col_del = st.columns(2)
    with col_load:
        load_id = st.number_input("Load session by ID", min_value=1, step=1, value=sessions[0]["id"])
        if st.button("📂 Load Session"):
            try:
                loaded_df, loaded_meta = load_session(int(load_id))
                st.info(f"Loaded session {load_id}: {loaded_meta.get('rows', '?')} rows, budget ₹{loaded_meta.get('monthly_budget', '?'):,}")
            except Exception as e:
                st.error(f"Could not load: {e}")
    with col_del:
        del_id = st.number_input("Delete session by ID", min_value=1, step=1, value=sessions[0]["id"])
        if st.button("🗑️ Delete Session"):
            delete_session(int(del_id))
            st.success(f"Session {del_id} deleted.")
            st.rerun()
else:
    st.info("No saved sessions yet. Sessions are saved automatically when you run the analysis.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  FEATURE 8 — CHATBOT: ASK YOUR DATA (Gemini AI + Rule fallback)
# ══════════════════════════════════════════════════════════════
st.markdown("### 🤖 Ask Your Data")

# Status badge
if st.session_state.gemini_enabled:
    st.markdown(
        "<span style='background:#064e3b;color:#6ee7b7;padding:3px 10px;"
        "border-radius:10px;font-size:0.75rem;font-weight:600'>"
        "● Gemini AI Active</span>&nbsp;"
        "<small style='color:#64748b'>Powered by Google Gemini — answers use your real spending data</small>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<span style='background:#1e293b;color:#94a3b8;padding:3px 10px;"
        "border-radius:10px;font-size:0.75rem'>"
        "● Rule-Based Mode</span>&nbsp;"
        "<small style='color:#64748b'>Add your Gemini API key in the sidebar to enable AI answers</small>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Suggested questions
with st.expander("💡 Suggested questions — click to copy"):
    suggestions = [
        "Where did I spend the most this month?",
        "How can I reduce my Food expenses?",
        "Am I on track to meet my savings goal?",
        "Which transactions look suspicious?",
        "What will my spending be next month?",
        "How does my spending compare to average?",
        "Give me a 3-step plan to save ₹10,000 next month.",
        "Which day of the week do I spend the most?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(suggestions):
        cols[i % 2].code(q, language=None)

# Chat input
chat_input = st.chat_input(
    "Ask anything about your finances…"
    if st.session_state.gemini_enabled
    else "e.g. Where did I spend the most? How can I save money?"
)

if chat_input:
    with st.spinner("🤔 Thinking…"):
        if st.session_state.gemini_enabled:
            response, used_gemini = gemini_chat(
                user_message   = chat_input,
                chat_history   = st.session_state.chat_history,
                df             = df,
                forecast_df    = forecast_df,
                health         = health,
                insights       = insights,
                monthly_budget = monthly_budget,
                api_key        = st.session_state.gemini_api_key,
                model_name     = st.session_state.gemini_model,
            )
        else:
            response    = answer_query(chat_input, df, forecast_df, health, insights, monthly_budget)
            used_gemini = False

    st.session_state.chat_history.append({"role": "user",      "content": chat_input})
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Render history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Controls row
ctrl1, ctrl2 = st.columns([1, 4])
with ctrl1:
    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()
with ctrl2:
    if st.session_state.chat_history:
        chat_export = "\n\n".join(
            f"{'You' if m['role']=='user' else 'SpendWise AI'}: {m['content']}"
            for m in st.session_state.chat_history
        )
        st.download_button(
            "📥 Export Chat",
            data=chat_export.encode(),
            file_name="spendwise_chat.txt",
            mime="text/plain",
            key="export_chat",
        )

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#334155;font-size:0.8rem'>"
    "SpendWise AI · Random Forest · Isolation Forest · Linear Regression · "
    "Built with Streamlit & Plotly</p>",
    unsafe_allow_html=True,
)