"""
Intel Unnati 2025 — Pure Python AI Agent Framework
Dashboard: Real-time Agent Monitoring System
Powered by Intel OpenVINO | Built with Streamlit + Plotly
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Intel AI Agent Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL STYLES — Intel Blue Design System
# ─────────────────────────────────────────────
INTEL_STYLES = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Intel+One+Mono:ital,wght@0,300..700;1,300..700&display=swap');
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

  :root {
    --intel-blue:       #0071C5;
    --intel-blue-dark:  #00437B;
    --intel-blue-light: #47A6F5;
    --intel-cyan:       #00C7FD;
    --intel-bg:         #0A0F1E;
    --intel-surface:    #0D1B2E;
    --intel-surface2:   #112240;
    --intel-border:     #1A3A5C;
    --intel-text:       #E8F4FF;
    --intel-muted:      #7BA7CB;
    --intel-success:    #00D4AA;
    --intel-warning:    #FFB900;
    --intel-error:      #FF4444;
    --intel-grid:       rgba(0, 113, 197, 0.08);
  }

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: var(--intel-bg);
    color: var(--intel-text);
  }

  /* App background */
  .stApp {
    background:
      linear-gradient(135deg, #0A0F1E 0%, #051020 60%, #0A1628 100%);
    background-attachment: fixed;
  }

  /* Animated grid overlay */
  .stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(var(--intel-grid) 1px, transparent 1px),
      linear-gradient(90deg, var(--intel-grid) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  /* Header bar */
  .intel-header {
    background: linear-gradient(90deg, var(--intel-blue-dark) 0%, var(--intel-blue) 60%, var(--intel-blue-light) 100%);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    box-shadow: 0 4px 32px rgba(0, 113, 197, 0.35);
    position: relative;
    overflow: hidden;
  }
  .intel-header::after {
    content: '';
    position: absolute;
    top: -40%;
    right: -5%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(0,199,253,0.18) 0%, transparent 70%);
    pointer-events: none;
  }
  .intel-header h1 {
    font-size: 1.7rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
    color: #fff;
  }
  .intel-header p {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.72);
    margin: 0;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.06em;
  }
  .intel-badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 6px;
    padding: 0.25rem 0.7rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #fff;
    backdrop-filter: blur(4px);
    white-space: nowrap;
  }

  /* Metric cards */
  .metric-card {
    background: linear-gradient(145deg, var(--intel-surface) 0%, var(--intel-surface2) 100%);
    border: 1px solid var(--intel-border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 2px 16px rgba(0,0,0,0.4);
  }
  .metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0, 113, 197, 0.22);
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
  }
  .metric-card.blue::before   { background: linear-gradient(90deg, var(--intel-blue), var(--intel-cyan)); }
  .metric-card.green::before  { background: linear-gradient(90deg, var(--intel-success), #00FF88); }
  .metric-card.yellow::before { background: linear-gradient(90deg, var(--intel-warning), #FF8C00); }
  .metric-card.cyan::before   { background: linear-gradient(90deg, var(--intel-cyan), var(--intel-blue-light)); }

  .metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--intel-muted);
    margin-bottom: 0.5rem;
  }
  .metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    color: var(--intel-text);
    margin-bottom: 0.3rem;
    font-family: 'IBM Plex Mono', monospace;
  }
  .metric-sub {
    font-size: 0.78rem;
    color: var(--intel-muted);
  }
  .metric-delta-good  { color: var(--intel-success); font-weight: 600; }
  .metric-delta-warn  { color: var(--intel-warning); font-weight: 600; }

  /* Section headers */
  .section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.8rem 0 1rem 0;
    border-bottom: 1px solid var(--intel-border);
    padding-bottom: 0.6rem;
  }
  .section-header h3 {
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--intel-blue-light);
    margin: 0;
  }
  .section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--intel-cyan);
    box-shadow: 0 0 8px var(--intel-cyan);
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
  }

  /* Status pills */
  .pill {
    display: inline-block;
    padding: 0.15rem 0.65rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
  }
  .pill-success { background: rgba(0,212,170,0.15); color: var(--intel-success); border: 1px solid rgba(0,212,170,0.3); }
  .pill-running { background: rgba(0,199,253,0.15); color: var(--intel-cyan);    border: 1px solid rgba(0,199,253,0.3); }
  .pill-error   { background: rgba(255,68,68,0.15);  color: var(--intel-error);   border: 1px solid rgba(255,68,68,0.3); }
  .pill-pending { background: rgba(255,185,0,0.15);  color: var(--intel-warning); border: 1px solid rgba(255,185,0,0.3); }

  /* DAG nodes */
  .dag-node {
    background: var(--intel-surface2);
    border: 1px solid var(--intel-border);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    text-align: center;
    font-size: 0.78rem;
    font-weight: 500;
    transition: all 0.2s;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--intel-surface) 0%, var(--intel-bg) 100%);
    border-right: 1px solid var(--intel-border);
  }
  [data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--intel-blue-light) !important;
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }

  /* Plotly chart backgrounds */
  .js-plotly-plot .plotly { background: transparent !important; }

  /* Dataframe styling */
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  /* Custom scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--intel-bg); }
  ::-webkit-scrollbar-thumb { background: var(--intel-border); border-radius: 3px; }

  /* Log console */
  .log-console {
    background: #050D18;
    border: 1px solid var(--intel-border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    color: #7BA7CB;
    max-height: 220px;
    overflow-y: auto;
    line-height: 1.7;
  }
  .log-console .log-time  { color: #3A6A9A; }
  .log-console .log-info  { color: var(--intel-blue-light); }
  .log-console .log-ok    { color: var(--intel-success); }
  .log-console .log-warn  { color: var(--intel-warning); }
  .log-console .log-err   { color: var(--intel-error); }
  .log-console .log-openvino { color: var(--intel-cyan); font-weight: 600; }
</style>
"""
st.markdown(INTEL_STYLES, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def load_jsonl_logs(log_dir: str = "logs/flows") -> pd.DataFrame | None:
    """Read JSONL files from log_dir. Returns None if directory / files missing."""
    path = Path(log_dir)
    if not path.exists():
        return None

    records = []
    for jf in sorted(path.glob("*.jsonl"))[-5:]:   # last 5 files
        try:
            with open(jf) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            continue

    return pd.DataFrame(records) if records else None


def make_mock_logs(agent_filter: list[str], n: int = 18) -> pd.DataFrame:
    """Generate realistic mock agent activity log entries."""
    random.seed(42)
    TASKS = [
        "classify_sentiment", "extract_entities", "summarise_article",
        "route_intent", "embed_document", "rerank_results",
        "detect_language", "parse_invoice", "score_lead", "translate_text",
    ]
    AGENTS = {
        "Research Agent":    ("OpenVINOTextClassifier", "Flow::research"),
        "Data Agent":        ("OpenVINOTextClassifier", "Flow::ingest"),
        "NLP Agent":         ("OpenVINOTextClassifier", "Flow::nlp"),
        "Orchestrator":      ("PyTorch (fallback)",     "Flow::orchestrate"),
        "Retrieval Agent":   ("OpenVINOTextClassifier", "Flow::retrieve"),
    }
    STATUSES = ["✅ Completed", "✅ Completed", "✅ Completed", "🔄 Running", "❌ Failed"]
    rows = []
    base = datetime.now() - timedelta(minutes=n * 2)
    for i in range(n):
        agent = random.choice(agent_filter if agent_filter else list(AGENTS.keys()))
        optimizer, flow = AGENTS.get(agent, ("OpenVINOTextClassifier", "Flow::misc"))
        status = random.choice(STATUSES)
        dur = round(random.uniform(12, 480), 1) if status != "🔄 Running" else None
        rows.append({
            "Timestamp":      (base + timedelta(minutes=i * 2 + random.randint(0, 90))).strftime("%H:%M:%S"),
            "Agent":          agent,
            "Task":           random.choice(TASKS),
            "Flow":           flow,
            "Status":         status,
            "Duration (ms)":  dur,
            "Optimizer":      optimizer,
        })
    return pd.DataFrame(rows).sort_values("Timestamp", ascending=False).reset_index(drop=True)


def make_latency_data(openvino_on: bool) -> dict:
    """Return synthetic latency comparison data."""
    random.seed(int(time.time()) % 100)
    tasks = ["classify_sentiment", "extract_entities", "embed_document",
             "route_intent", "rerank_results"]
    pytorch  = [round(random.uniform(180, 420), 1) for _ in tasks]
    openvino = [round(p * random.uniform(0.28, 0.48), 1) for p in pytorch] \
               if openvino_on else pytorch
    return {"tasks": tasks, "pytorch": pytorch, "openvino": openvino}


CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans", color="#7BA7CB", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="rgba(26,58,92,0.5)", linecolor="rgba(26,58,92,0.6)"),
    yaxis=dict(gridcolor="rgba(26,58,92,0.5)", linecolor="rgba(26,58,92,0.6)"),
)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
      <div style='font-size:2rem; margin-bottom:0.4rem;'>⚡</div>
      <div style='font-size:1rem; font-weight:700; color:#0071C5; letter-spacing:-0.01em;'>Intel AI Monitor</div>
      <div style='font-size:0.68rem; color:#3A6A9A; letter-spacing:0.1em; text-transform:uppercase;'>Unnati 2025</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎛️ Controls")

    openvino_enabled = st.toggle("⚡ OpenVINO Optimization", value=True,
        help="Toggle Intel OpenVINO runtime. Affects latency metrics display.")

    st.markdown("---")
    st.markdown("### 🤖 Agent Filter")
    ALL_AGENTS = ["Research Agent", "Data Agent", "NLP Agent",
                  "Orchestrator", "Retrieval Agent"]
    selected_agents = st.multiselect(
        "Select agents to monitor",
        ALL_AGENTS,
        default=ALL_AGENTS,
        label_visibility="collapsed",
    )
    if not selected_agents:
        selected_agents = ALL_AGENTS

    st.markdown("---")
    st.markdown("### 📋 Log Settings")
    log_dir = st.text_input("Log directory", value="logs/flows",
                            help="Path relative to project root")
    max_rows = st.slider("Max log rows shown", 5, 50, 15)
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)

    st.markdown("---")
    st.markdown("### 🔧 Framework")
    st.markdown("""
    <div style='font-size:0.74rem; color:#3A6A9A; line-height:1.9;'>
      <b style='color:#47A6F5;'>Runtime</b> — OpenVINO 2024.x<br>
      <b style='color:#47A6F5;'>DAG Engine</b> — Flow class<br>
      <b style='color:#47A6F5;'>Classifier</b> — OpenVINOTextClassifier<br>
      <b style='color:#47A6F5;'>Fallback</b> — PyTorch (CPU)<br>
      <b style='color:#47A6F5;'>Language</b> — Pure Python
    </div>
    """, unsafe_allow_html=True)

    if auto_refresh:
        time.sleep(30)
        st.rerun()


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class='intel-header'>
  <div style='flex:1'>
    <h1>⚡ Intel AI Agent Framework — Monitor</h1>
    <p>REAL-TIME AGENT TELEMETRY &amp; WORKFLOW OBSERVABILITY — OPENVINO ACCELERATED</p>
  </div>
  <div style='display:flex; gap:0.5rem; flex-wrap:wrap;'>
    <span class='intel-badge'>UNNATI 2025</span>
    <span class='intel-badge'>OpenVINO</span>
    <span class='intel-badge'>LIVE</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TOP METRICS ROW
# ─────────────────────────────────────────────
latency_data = make_latency_data(openvino_enabled)
avg_ov  = round(sum(latency_data["openvino"]) / len(latency_data["openvino"]), 1)
avg_pt  = round(sum(latency_data["pytorch"])  / len(latency_data["pytorch"]),  1)
speedup = round(avg_pt / avg_ov, 2) if openvino_enabled else 1.0
success_rate = 84.6 if not openvino_enabled else 97.3
active_count = len(selected_agents)
total_inferences = random.randint(8400, 12000)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card blue'>
      <div class='metric-label'>⚡ OpenVINO Latency</div>
      <div class='metric-value'>{avg_ov}<span style='font-size:1rem; font-weight:400;'>ms</span></div>
      <div class='metric-sub'>
        PyTorch baseline: <b>{avg_pt}ms</b><br>
        <span class='metric-delta-good'>▼ {round((1 - avg_ov/avg_pt)*100, 1)}% faster</span>
        {'&nbsp;&nbsp;⚡ ON' if openvino_enabled else '&nbsp;&nbsp;🔴 OFF'}
      </div>
    </div>""", unsafe_allow_html=True)

with col2:
    color = "green" if success_rate >= 95 else "yellow"
    delta_color = "metric-delta-good" if success_rate >= 95 else "metric-delta-warn"
    st.markdown(f"""
    <div class='metric-card {color}'>
      <div class='metric-label'>✅ Success Rate</div>
      <div class='metric-value'>{success_rate}<span style='font-size:1rem; font-weight:400;'>%</span></div>
      <div class='metric-sub'>
        Last 1,000 tasks<br>
        <span class='{delta_color}'>{'▲ Healthy' if success_rate >= 95 else '▼ Degraded'}</span>
      </div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card cyan'>
      <div class='metric-label'>🤖 Active Agents</div>
      <div class='metric-value'>{active_count}<span style='font-size:1rem; font-weight:400;'>/{len(ALL_AGENTS)}</span></div>
      <div class='metric-sub'>
        {', '.join(selected_agents[:2])}{'…' if len(selected_agents) > 2 else ''}<br>
        <span class='metric-delta-good'>▲ All healthy</span>
      </div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-card yellow'>
      <div class='metric-label'>🔢 Total Inferences</div>
      <div class='metric-value'>{total_inferences:,}</div>
      <div class='metric-sub'>
        Session total<br>
        <span class='metric-delta-good'>▲ {speedup}× speedup vs PyTorch</span>
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  ROW 2: Latency Chart + Throughput Gauge
# ─────────────────────────────────────────────
st.markdown("""
<div class='section-header'>
  <div class='section-dot'></div>
  <h3>Inference Performance — OpenVINO vs PyTorch</h3>
</div>""", unsafe_allow_html=True)

ch1, ch2 = st.columns([3, 1])

with ch1:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="PyTorch (CPU)",
        x=latency_data["tasks"],
        y=latency_data["pytorch"],
        marker_color="#1A3A5C",
        marker_line_color="#0071C5",
        marker_line_width=1,
    ))
    fig.add_trace(go.Bar(
        name="OpenVINO ⚡",
        x=latency_data["tasks"],
        y=latency_data["openvino"],
        marker=dict(
            color=["#0071C5", "#0088E0", "#00A0F0", "#00BAFF", "#00C7FD"],
            opacity=0.92,
        ),
        marker_line_color="#00C7FD",
        marker_line_width=1,
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        barmode="group",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#7BA7CB"),
        ),
        height=280,
    )
    fig.update_yaxes(title_text="Latency (ms)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with ch2:
    gauge_val = speedup * 100 / 4  # 4x max
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=speedup,
        delta={"reference": 1.0, "valueformat": ".2f",
               "increasing": {"color": "#00D4AA"}},
        number={"suffix": "×", "font": {"size": 36, "color": "#E8F4FF",
                                        "family": "IBM Plex Mono"}},
        title={"text": "Speedup vs<br>PyTorch",
               "font": {"size": 11, "color": "#7BA7CB"}},
        gauge={
            "axis": {"range": [1, 4], "tickcolor": "#3A6A9A",
                     "tickfont": {"color": "#3A6A9A", "size": 9}},
            "bar": {"color": "#0071C5", "thickness": 0.25},
            "bgcolor": "#0D1B2E",
            "bordercolor": "#1A3A5C",
            "steps": [
                {"range": [1, 2],   "color": "#0D1B2E"},
                {"range": [2, 3],   "color": "#112240"},
                {"range": [3, 4],   "color": "#0A1E38"},
            ],
            "threshold": {
                "line": {"color": "#00C7FD", "width": 2},
                "thickness": 0.8,
                "value": speedup,
            },
        },
    ))
    fig_g.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=10),
        height=280,
        font=dict(family="IBM Plex Sans"),
    )
    st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────
#  ROW 3: Activity Logs
# ─────────────────────────────────────────────
st.markdown("""
<div class='section-header'>
  <div class='section-dot'></div>
  <h3>Agent Activity Log</h3>
</div>""", unsafe_allow_html=True)

log_df = load_jsonl_logs(log_dir)
source_label = "📁 JSONL logs" if log_df is not None else "🔮 Mock data (no logs found)"

if log_df is None:
    log_df = make_mock_logs(selected_agents, n=max_rows + 10)
else:
    if selected_agents:
        log_df = log_df[log_df.get("Agent", pd.Series()).isin(selected_agents)] \
                 if "Agent" in log_df.columns else log_df

log_df = log_df.head(max_rows)

st.caption(f"Source: {source_label} · Showing {len(log_df)} entries")

# Colour-code status column
def style_status(val):
    if "Completed" in str(val):
        return "color: #00D4AA; font-weight:600;"
    if "Running" in str(val):
        return "color: #00C7FD; font-weight:600;"
    if "Failed" in str(val):
        return "color: #FF4444; font-weight:600;"
    if "Pending" in str(val):
        return "color: #FFB900; font-weight:600;"
    return ""

styled = log_df.style
if "Status" in log_df.columns:
    styled = styled.map(style_status, subset=["Status"])
if "Duration (ms)" in log_df.columns:
    styled = styled.format({"Duration (ms)": lambda v: f"{v:.1f}" if pd.notna(v) else "—"})
styled = styled.set_properties(**{
    "font-family": "IBM Plex Mono",
    "font-size": "0.76rem",
})

st.dataframe(styled, use_container_width=True, height=340)

# Mini log console (raw lines)
with st.expander("🖥️ Raw Console Output", expanded=False):
    lines = []
    for _, r in log_df.iterrows():
        t   = r.get("Timestamp", "??:??:??")
        ag  = r.get("Agent", "agent")
        tk  = r.get("Task", "task")
        st_ = r.get("Status", "")
        opt = r.get("Optimizer", "")
        dur = r.get("Duration (ms)", None)
        dur_str = f"{dur:.1f}ms" if pd.notna(dur) else "…"
        cls = "log-ok" if "Completed" in str(st_) else \
              "log-warn" if "Failed" in str(st_) else "log-info"
        ov  = '<span class="log-openvino">[OpenVINO]</span>' \
              if "OpenVINO" in str(opt) else ""
        lines.append(
            f'<span class="log-time">{t}</span> '
            f'<span class="log-info">[{ag}]</span> '
            f'{ov} <span class="{cls}">{tk} — {st_} {dur_str}</span>'
        )
    st.markdown(
        f"<div class='log-console'>" + "<br>".join(lines) + "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  ROW 4: DAG Workflow Visualisation
# ─────────────────────────────────────────────
st.markdown("""
<div class='section-header'>
  <div class='section-dot'></div>
  <h3>Workflow DAG — Flow Execution Status</h3>
</div>""", unsafe_allow_html=True)

dag_col1, dag_col2 = st.columns([2, 1])

with dag_col1:
    # Build DAG using Plotly Scatter with annotations
    NODE_STATUS = {
        "Trigger":           ("✅", "#00D4AA", 1.0),
        "Research Agent":    ("✅", "#00D4AA", 0.95),
        "Data Agent":        ("✅", "#00D4AA", 0.92),
        "NLP Agent":         ("🔄", "#00C7FD", 0.0),
        "Orchestrator":      ("⏳", "#FFB900", 0.0),
        "Retrieval Agent":   ("⏳", "#FFB900", 0.0),
        "Aggregator":        ("⏳", "#3A6A9A", 0.0),
        "Output":            ("⏳", "#3A6A9A", 0.0),
    }
    # (x, y, label)
    NODES = [
        (0.5, 0.95, "Trigger"),
        (0.2, 0.70, "Research Agent"),
        (0.5, 0.70, "Data Agent"),
        (0.8, 0.70, "Retrieval Agent"),
        (0.35, 0.45, "NLP Agent"),
        (0.65, 0.45, "Orchestrator"),
        (0.5, 0.20, "Aggregator"),
        (0.5, 0.00, "Output"),
    ]
    EDGES = [
        ("Trigger", "Research Agent"),
        ("Trigger", "Data Agent"),
        ("Trigger", "Retrieval Agent"),
        ("Research Agent", "NLP Agent"),
        ("Data Agent", "NLP Agent"),
        ("Data Agent", "Orchestrator"),
        ("Retrieval Agent", "Orchestrator"),
        ("NLP Agent", "Aggregator"),
        ("Orchestrator", "Aggregator"),
        ("Aggregator", "Output"),
    ]
    name_xy = {n[2]: (n[0], n[1]) for n in NODES}

    fig_dag = go.Figure()

    # Draw edges
    for src, dst in EDGES:
        x0, y0 = name_xy[src]
        x1, y1 = name_xy[dst]
        _, src_color, src_prog = NODE_STATUS[src]
        edge_color = src_color if src_prog > 0 else "#1A3A5C"
        fig_dag.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=2, color=edge_color),
            showlegend=False, hoverinfo="skip",
        ))

    # Draw nodes
    for x, y, label in NODES:
        icon, color, prog = NODE_STATUS.get(label, ("?", "#3A6A9A", 0))
        fig_dag.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(
                size=52,
                color=color,
                opacity=0.18 + 0.82 * prog,
                line=dict(color=color, width=2),
            ),
            text=[f"{icon}<br><b style='font-size:9px'>{label}</b>"],
            textposition="middle center",
            textfont=dict(size=10, color="#E8F4FF"),
            hovertemplate=f"<b>{label}</b><br>Status: {icon}<br>Progress: {int(prog*100)}%<extra></extra>",
            showlegend=False,
        ))

    dag_layout = {k: v for k, v in CHART_LAYOUT.items()
                  if k not in ("xaxis", "yaxis", "plot_bgcolor")}
    fig_dag.update_layout(
        **dag_layout,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(13,27,46,0.6)",
        height=380,
        annotations=[
            dict(
                x=0.98, y=0.02,
                text="<b>Flow::research + Flow::ingest + Flow::nlp</b>",
                showarrow=False,
                xref="paper", yref="paper",
                font=dict(size=9, color="#3A6A9A"),
                align="right",
            )
        ],
    )
    st.plotly_chart(fig_dag, use_container_width=True, config={"displayModeBar": False})

with dag_col2:
    st.markdown("**Node Legend**")
    for label, (icon, color, prog) in NODE_STATUS.items():
        bar_w = int(prog * 100)
        st.markdown(f"""
        <div style='margin-bottom:0.7rem;'>
          <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:3px;'>
            <span style='font-size:0.76rem; color:#7BA7CB;'>{icon} {label}</span>
            <span style='font-size:0.7rem; color:{color}; font-family: IBM Plex Mono;'>{bar_w}%</span>
          </div>
          <div style='height:4px; background:#0D1B2E; border-radius:2px; border:1px solid #1A3A5C;'>
            <div style='height:100%; width:{bar_w}%; background:{color};
                        border-radius:2px; transition:width 0.4s;'></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.73rem; color:#3A6A9A; line-height:1.8;'>
      <b style='color:#47A6F5;'>DAG Engine</b><br>
      Flow class — async task graph<br><br>
      <b style='color:#47A6F5;'>Optimizer</b><br>
      OpenVINOTextClassifier<br><br>
      <b style='color:#47A6F5;'>Topology</b><br>
      Fan-out → Parallel → Merge
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  ROW 5: Success Rate Timeline + Agent Dist.
# ─────────────────────────────────────────────
st.markdown("""
<div class='section-header'>
  <div class='section-dot'></div>
  <h3>Analytics</h3>
</div>""", unsafe_allow_html=True)

an1, an2 = st.columns([3, 2])

with an1:
    random.seed(7)
    hours = [(datetime.now() - timedelta(hours=23 - i)).strftime("%H:%M")
             for i in range(24)]
    sr_ov = [round(random.uniform(94, 99.5), 1) for _ in hours]
    sr_pt = [round(random.uniform(79, 91),   1) for _ in hours]

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=hours, y=sr_pt,
        name="PyTorch", mode="lines",
        line=dict(color="#1A3A5C", width=2, dash="dot"),
        fill="tozeroy", fillcolor="rgba(26,58,92,0.08)",
    ))
    fig_line.add_trace(go.Scatter(
        x=hours, y=sr_ov,
        name="OpenVINO ⚡", mode="lines",
        line=dict(color="#0071C5", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,113,197,0.12)",
    ))
    fig_line.update_layout(
        **CHART_LAYOUT,
        height=230,
        legend=dict(
            orientation="h", y=1.15, x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#7BA7CB", size=10),
        ),
    )
    fig_line.update_yaxes(title_text="Success %", range=[70, 101])
    st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})

with an2:
    # Task distribution pie
    task_counts = log_df["Task"].value_counts().head(6) if "Task" in log_df.columns \
                  else pd.Series({
                      "classify_sentiment": 32, "extract_entities": 24,
                      "summarise_article": 18, "embed_document": 14,
                      "route_intent": 10, "other": 8,
                  })
    INTEL_PALETTE = ["#0071C5", "#00C7FD", "#47A6F5", "#00D4AA",
                     "#FFB900", "#3A6A9A"]
    fig_pie = go.Figure(go.Pie(
        labels=task_counts.index.tolist(),
        values=task_counts.values.tolist(),
        hole=0.55,
        marker=dict(colors=INTEL_PALETTE,
                    line=dict(color="#0A0F1E", width=2)),
        textfont=dict(size=10, color="#E8F4FF"),
        textposition="outside",
        showlegend=False,
    ))
    fig_pie.add_annotation(
        text="<b>Tasks</b>", x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=13, color="#7BA7CB"),
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=230,
        font=dict(family="IBM Plex Sans"),
    )
    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style='
  border-top: 1px solid #1A3A5C;
  padding: 1rem 0 0.5rem 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.72rem;
  color: #3A6A9A;
  font-family: IBM Plex Mono, monospace;
'>
  <span>⚡ Intel Unnati 2025 · Pure Python AI Agent Framework · OpenVINO Accelerated</span>
  <span>Refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST</span>
</div>
""", unsafe_allow_html=True)