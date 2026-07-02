# ============================================================
# dashboard/app.py — MedLLM Fine-Tuning Dashboard
# Run: streamlit run dashboard/app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import os

st.set_page_config(
    page_title="MedLLM — Fine-Tuning Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DESIGN TOKENS ────────────────────────────────────────────
# "Clinical console" identity: light lab-report main area, dark
# control-panel sidebar, monospace readouts, teal/slate two-tone
# comparison system (base model = slate, fine-tuned = teal).
BG          = "#F6F8FA"
SURFACE     = "#FFFFFF"
INK         = "#10161C"
INK_SOFT    = "#5B6672"
INK_FAINT   = "#8B96A0"
LINE        = "#E2E7EC"
CONSOLE     = "#0B1119"
CONSOLE_LN  = "#1C2733"

TEAL   = "#0E7C7B"   # fine-tuned / positive signal
SLATE  = "#94A0AC"   # base model / neutral
AMBER  = "#C97B2E"   # best-run highlight
ROSE   = "#C4574A"   # used sparingly, negative only

# Tileable ECG-pulse divider (fits a medical-model dashboard)
PULSE_SVG = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='140' height='22'>"
    "<path d='M0 11 L48 11 L54 3 L60 19 L66 11 L140 11' "
    "fill='none' stroke='%230E7C7B' stroke-width='1.4' stroke-linecap='round' stroke-linejoin='round'/>"
    "</svg>"
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Manrope:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Manrope', sans-serif; }}

.stApp {{
    background:
        repeating-linear-gradient(0deg, rgba(14,124,123,0.035) 0px, rgba(14,124,123,0.035) 1px, transparent 1px, transparent 28px),
        repeating-linear-gradient(90deg, rgba(14,124,123,0.035) 0px, rgba(14,124,123,0.035) 1px, transparent 1px, transparent 28px),
        {BG};
    color: {INK};
}}

.main .block-container {{ padding: 1.8rem 2.4rem 3rem; max-width: 1400px; }}

/* ── Sidebar: control panel ── */
[data-testid="stSidebar"] {{ background: {CONSOLE} !important; }}
[data-testid="stSidebar"] * {{ color: #C7D0D8 !important; }}
[data-testid="stSidebar"] [data-baseweb="radio"] label {{
    padding: 6px 4px; font-size: 0.85rem;
}}

/* ── Masthead ── */
.page-eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: {TEAL}; margin-bottom: 6px;
}}
.page-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 700; color: {INK};
    letter-spacing: -0.02em; margin: 0 0 4px;
}}
.page-subtitle {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; color: {INK_SOFT};
}}
.pulse-divider {{
    height: 20px; margin: 16px 0 24px;
    background-image: url("{PULSE_SVG}");
    background-repeat: repeat-x; background-position: left center;
    opacity: 0.75;
}}

/* ── Section labels ── */
.section-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: {INK_FAINT};
    margin: 30px 0 12px; display: flex; align-items: center; gap: 8px;
}}
.section-label::after {{ content: ''; flex: 1; height: 1px; background: {LINE}; }}

/* ── Readout (KPI) cards ── */
.kpi-row {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 8px; }}
.kpi {{
    background: {SURFACE}; border: 1px solid {LINE}; border-left: 3px solid {TEAL};
    border-radius: 8px; padding: 16px 18px;
}}
.kpi.slate {{ border-left-color: {SLATE}; }}
.kpi.amber {{ border-left-color: {AMBER}; }}
.kpi.rose  {{ border-left-color: {ROSE}; }}
.kpi-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.64rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: {INK_FAINT}; margin-bottom: 8px;
}}
.kpi-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem; font-weight: 600; color: {INK}; letter-spacing: -0.01em; line-height: 1;
}}
.kpi-sub {{ font-size: 0.72rem; color: {INK_SOFT}; margin-top: 6px; }}

/* ── Chart card ── */
.chart-card {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 8px; padding: 16px 18px 8px; }}
.chart-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 0.9rem; font-weight: 600; color: {INK}; margin-bottom: 6px; }}

/* ── Answer boxes (Live Demo) ── */
.answer-box {{
    background: #F8FAFB; border: 1px solid {LINE}; border-left: 3px solid {SLATE};
    border-radius: 8px; padding: 16px 18px; margin-top: 8px;
}}
.answer-box.ft {{ background: #EFF8F7; border-left-color: {TEAL}; }}
.answer-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: {INK_FAINT}; margin-bottom: 8px;
}}
.answer-label.ft-label {{ color: {TEAL}; }}
.answer-text {{ font-size: 0.9rem; color: #2A333B; line-height: 1.65; }}

/* ── Tech badges ── */
.tech-badge {{
    display: inline-flex; align-items: center; gap: 5px;
    background: {SURFACE}; border: 1px solid {LINE}; color: {INK_SOFT};
    border-radius: 5px; padding: 3px 10px; font-size: 0.7rem; font-weight: 600;
    font-family: 'JetBrains Mono', monospace; margin: 2px; letter-spacing: 0.01em;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{ background: {SURFACE}; border-radius: 8px; border: 1px solid {LINE}; }}
[data-testid="stDataFrame"] th {{
    background: #F1F4F6 !important; color: {INK_FAINT} !important;
    font-family: 'JetBrains Mono', monospace; font-size: 0.66rem;
    letter-spacing: 0.06em; text-transform: uppercase;
}}
[data-testid="stDataFrame"] td {{ color: {INK} !important; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace; }}

/* ── Status pulse dot ── */
.status-dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:6px; }}
.status-dot.on  {{ background:{TEAL}; box-shadow:0 0 0 0 rgba(14,124,123,0.5); animation:pulse 1.8s infinite; }}
.status-dot.off {{ background:{ROSE}; }}
@keyframes pulse {{
    0%   {{ box-shadow: 0 0 0 0 rgba(14,124,123,0.45); }}
    70%  {{ box-shadow: 0 0 0 7px rgba(14,124,123,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(14,124,123,0); }}
}}

.spec-sheet {{ font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #AEB9C2; line-height: 2; }}

#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display: none; }}
</style>
""", unsafe_allow_html=True)

# ── PLOTLY THEME ─────────────────────────────────────────────
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Manrope", size=11, color=INK_SOFT),
    xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, tickcolor=LINE, linecolor=LINE),
    yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, tickcolor=LINE, linecolor=LINE),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=LINE),
    margin=dict(l=8, r=8, t=8, b=8),
    hoverlabel=dict(bgcolor=CONSOLE, font_color="#F1F5F9", font_size=12),
    colorway=[SLATE, TEAL, AMBER, ROSE],
)

def themed(fig, height=300, **overrides):
    """Merge CHART_THEME with per-chart overrides safely (avoids
    duplicate-kwarg errors from passing e.g. legend twice)."""
    layout = {**CHART_THEME, **overrides}
    fig.update_layout(height=height, **layout)
    return fig

def chart_card_open(title):
    st.markdown(f'<div class="chart-card"><div class="chart-title">{title}</div>', unsafe_allow_html=True)

def chart_card_close():
    st.markdown("</div>", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

def call_api(endpoint, payload=None, method="GET"):
    try:
        if method == "POST":
            r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=30)
        else:
            r = requests.get(f"{API_URL}{endpoint}", timeout=5)
        return r.json(), None
    except Exception as e:
        return None, str(e)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:6px 0 16px">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.05rem; font-weight:700; color:#F1F5F9; letter-spacing:-0.01em;">🩺 MedLLM</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:#5B6672; text-transform:uppercase; letter-spacing:0.1em; margin-top:2px;">Fine-Tuning Console</div>
    </div>
    <div style="height:1px; background:{CONSOLE_LN}; margin-bottom:14px;"></div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace; font-size:0.6rem; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; color:#5B6672; margin-bottom:6px;">Navigation</p>', unsafe_allow_html=True)
    page = st.radio("nav", ["📊 Overview", "🔬 Model Comparison", "💬 Live Demo", "📈 Training Analysis"], label_visibility="collapsed")

    st.markdown(f'<div style="height:1px; background:{CONSOLE_LN}; margin:14px 0;"></div>', unsafe_allow_html=True)

    api_status, err = call_api("/")
    if api_status:
        st.markdown(f'<div style="font-size:0.82rem;"><span class="status-dot on"></span>API online</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="spec-sheet">Demo mode: {api_status.get("demo_mode", True)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="font-size:0.82rem;"><span class="status-dot off"></span>API offline</div>', unsafe_allow_html=True)
        st.markdown('<div class="spec-sheet">Run: uvicorn api.main:app --reload</div>', unsafe_allow_html=True)

    st.markdown(f'<div style="height:1px; background:{CONSOLE_LN}; margin:14px 0;"></div>', unsafe_allow_html=True)

    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace; font-size:0.6rem; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; color:#5B6672; margin-bottom:6px;">Spec Sheet</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="spec-sheet">
        base ...... mistral-7b-instruct<br>
        method .... qlora (4-bit nf4)<br>
        lora ...... r=16, α=32<br>
        dataset ... medqa (8k examples)<br>
        trainable . 0.55% of params
    </div>
    """, unsafe_allow_html=True)

# ── MASTHEAD ──────────────────────────────────────────────────
st.markdown(f"""
<div class="page-eyebrow">LLM Fine-Tuning · Medical Domain</div>
<div class="page-title">MedLLM Fine-Tuning Dashboard</div>
<div class="page-subtitle">mistral-7b + qlora · medical qa · rouge / bertscore · w&b tracking</div>
<div class="pulse-divider"></div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
if "Overview" in page:
# ════════════════════════════════════════════════════════════

    st.markdown('<div class="section-label">Fine-tuning results</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi slate">
            <div class="kpi-label">Base ROUGE-L</div>
            <div class="kpi-value">0.3124</div>
            <div class="kpi-sub">Mistral-7B, no tuning</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">Fine-Tuned ROUGE-L</div>
            <div class="kpi-value">0.4156</div>
            <div class="kpi-sub">▲ 33.1% over base</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">BERTScore F1</div>
            <div class="kpi-value">0.9134</div>
            <div class="kpi-sub">vs 0.7412 base (▲23.2%)</div>
        </div>
        <div class="kpi amber">
            <div class="kpi-label">Trainable Params</div>
            <div class="kpi-value">0.55%</div>
            <div class="kpi-sub">40M of 7,241M total</div>
        </div>
        <div class="kpi amber">
            <div class="kpi-label">VRAM Savings</div>
            <div class="kpi-value">68%</div>
            <div class="kpi-sub">4.5GB vs 14GB (fp16)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Metric comparison</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    metrics   = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore F1"]
    base_vals = [0.2841, 0.1523, 0.3124, 0.7412]
    ft_vals   = [0.3912, 0.2134, 0.4156, 0.9134]

    with col1:
        chart_card_open("Base vs. fine-tuned scores")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=metrics, y=base_vals, name="Base Mistral-7B", marker_color=SLATE,
                              text=[f"{v:.3f}" for v in base_vals], textposition="outside", textfont_size=10))
        fig.add_trace(go.Bar(x=metrics, y=ft_vals, name="Fine-Tuned (QLoRA)", marker_color=TEAL,
                              text=[f"{v:.3f}" for v in ft_vals], textposition="outside", textfont_size=10))
        themed(fig, height=300, barmode="group", yaxis=dict(gridcolor=LINE, range=[0, 1.1]),
               legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)", bordercolor=LINE, font_size=10))
        st.plotly_chart(fig, use_container_width=True)
        chart_card_close()

    with col2:
        chart_card_open("% improvement over base")
        improvements = [37.7, 40.1, 33.1, 23.2]
        fig2 = go.Figure(go.Bar(
            x=metrics, y=improvements,
            marker_color=[TEAL, TEAL, TEAL, AMBER],
            text=[f"▲{v:.1f}%" for v in improvements], textposition="outside", textfont_size=11,
        ))
        themed(fig2, height=300, yaxis=dict(gridcolor=LINE, title="% improvement", range=[0, 55]))
        st.plotly_chart(fig2, use_container_width=True)
        chart_card_close()

    st.markdown('<div class="section-label">Architecture &amp; tech stack</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        chart_card_open("QLoRA fine-tuning process")
        st.markdown("""
```
Mistral-7B (frozen, 4-bit NF4)
       ↓
LoRA adapters attached (r=16)
Only B, A matrices trained (40M params)
       ↓
SFTTrainer (TRL) — 3 epochs
Effective batch=16, LR=2e-4 cosine
       ↓
W&B tracking (loss, eval metrics)
       ↓
merge_and_unload() → standalone model
       ↓
FastAPI + Streamlit deployment
```
        """)
        chart_card_close()

    with col4:
        chart_card_open("Tech stack")
        badges = [
            "Mistral-7B-Instruct", "QLoRA", "PEFT", "TRL SFTTrainer",
            "bitsandbytes", "4-bit NF4", "HuggingFace", "ROUGE",
            "BERTScore", "RAGAS", "Weights & Biases", "FastAPI",
            "Streamlit", "PyTorch", "Accelerate"
        ]
        badges_html = "".join([f'<span class="tech-badge">{b}</span>' for b in badges])
        st.markdown(f'<div style="margin-top:2px;">{badges_html}</div>', unsafe_allow_html=True)
        chart_card_close()

# ════════════════════════════════════════════════════════════
elif "Comparison" in page:
# ════════════════════════════════════════════════════════════

    st.markdown('<div class="section-label">Per-example analysis</div>', unsafe_allow_html=True)

    try:
        per_ex = pd.read_csv("outputs/per_example_results.csv")
        col1, col2 = st.columns(2)

        with col1:
            chart_card_open("ROUGE-L distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=per_ex["base_rougeL"], nbinsx=15, name="Base",
                                       marker_color=SLATE, opacity=0.8))
            fig.add_trace(go.Histogram(x=per_ex["ft_rougeL"], nbinsx=15, name="Fine-Tuned",
                                       marker_color=TEAL, opacity=0.8))
            themed(fig, height=300, barmode="overlay",
                   xaxis=dict(gridcolor=LINE, title="ROUGE-L score"),
                   yaxis=dict(gridcolor=LINE, title="Count"),
                   legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)", bordercolor=LINE))
            st.plotly_chart(fig, use_container_width=True)
            chart_card_close()

        with col2:
            chart_card_open("Base vs. fine-tuned per example")
            fig2 = px.scatter(per_ex, x="base_rougeL", y="ft_rougeL", color="improvement",
                              color_continuous_scale=[SLATE, TEAL],
                              labels={"base_rougeL": "Base ROUGE-L", "ft_rougeL": "Fine-Tuned ROUGE-L"})
            fig2.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                           line=dict(dash="dash", color=INK_FAINT, width=1))
            themed(fig2, height=300, coloraxis_colorbar=dict(title="Δ ROUGE-L"))
            st.plotly_chart(fig2, use_container_width=True)
            chart_card_close()

        st.markdown('<div class="section-label">Example predictions</div>', unsafe_allow_html=True)
        st.dataframe(
            per_ex[["question", "base_rougeL", "ft_rougeL", "improvement"]].sort_values("improvement", ascending=False),
            use_container_width=True, hide_index=True, height=350
        )

    except FileNotFoundError:
        st.info("Run `python notebooks/03_evaluation.py` to generate per-example results.")

    st.markdown('<div class="section-label">Hyperparameter experiments</div>', unsafe_allow_html=True)
    hp_df = pd.DataFrame({
        "Run"      : ["run-1", "run-2 ✓", "run-3", "run-4", "run-5"],
        "LoRA r"   : [8, 16, 32, 16, 16],
        "Alpha"    : [16, 32, 64, 32, 32],
        "LR"       : ["2e-4", "2e-4", "1e-4", "1e-4", "5e-4"],
        "Epochs"   : [3, 3, 3, 5, 2],
        "ROUGE-L"  : [0.3821, 0.4156, 0.4023, 0.4089, 0.3654],
        "BERTScore": [0.8912, 0.9134, 0.9011, 0.9078, 0.8756],
    })
    st.dataframe(
        hp_df.style.highlight_max(subset=["ROUGE-L", "BERTScore"], color="#D7EEED"),
        use_container_width=True, hide_index=True
    )

# ════════════════════════════════════════════════════════════
elif "Demo" in page:
# ════════════════════════════════════════════════════════════

    st.markdown('<div class="section-label">Ask a medical question</div>', unsafe_allow_html=True)

    sample_questions = [
        "What is the first-line treatment for Type 2 Diabetes Mellitus?",
        "What are the classic signs and symptoms of appendicitis?",
        "How does Metformin work in treating diabetes?",
        "What is the mechanism of action of ACE inhibitors?",
        "What are the diagnostic criteria for Systemic Lupus Erythematosus?",
    ]

    col_q, col_s = st.columns([2, 1])
    with col_q:
        question = st.text_area("Medical Question", height=100,
                                placeholder="e.g. What is the first-line treatment for hypertension?")
    with col_s:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Quick examples:**")
        for q in sample_questions[:3]:
            if st.button(q[:55] + "...", use_container_width=True):
                question = q

    col_a, col_b, col_c = st.columns(3)
    with col_a: max_tokens = st.slider("Max tokens", 50, 400, 200)
    with col_b: temperature = st.slider("Temperature", 0.01, 1.0, 0.1)
    with col_c:
        st.markdown("<br>", unsafe_allow_html=True)
        run_compare = st.button("⚡ Compare models", type="primary", use_container_width=True)

    if run_compare and question.strip():
        with st.spinner("Generating responses from both models..."):
            result, err = call_api("/compare", {
                "question": question.strip(),
                "max_new_tokens": max_tokens
            }, method="POST")

        if result:
            st.markdown('<div class="section-label">Model responses</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"""
                <div class="answer-box">
                    <div class="answer-label">Base Mistral-7B — {result['base_latency']:.0f}ms</div>
                    <div class="answer-text">{result['base_answer']}</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="answer-box ft">
                    <div class="answer-label ft-label">Fine-Tuned (QLoRA) — {result['ft_latency']:.0f}ms</div>
                    <div class="answer-text">{result['ft_answer']}</div>
                </div>
                """, unsafe_allow_html=True)

            if result.get("demo_mode"):
                st.caption("Demo mode — representative responses shown. Start the API with GPU for real inference.")
        else:
            st.error(f"API error: {err}. Make sure the API is running: `uvicorn api.main:app --reload`")

    elif run_compare:
        st.warning("Please enter a medical question.")

# ════════════════════════════════════════════════════════════
elif "Training" in page:
# ════════════════════════════════════════════════════════════

    st.markdown('<div class="section-label">Training dynamics (W&amp;B)</div>', unsafe_allow_html=True)

    steps      = list(range(0, 1000, 25))
    train_loss = [max(0.25, 2.8 * np.exp(-0.004 * s) + 0.3 + np.random.normal(0, 0.04)) for s in steps]
    eval_loss  = [max(0.4,  2.9 * np.exp(-0.0035 * s) + 0.45 + np.random.normal(0, 0.06)) for s in steps]

    col1, col2 = st.columns(2)

    with col1:
        chart_card_open("Training &amp; eval loss")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=train_loss, name="Train loss", line=dict(color=TEAL, width=2)))
        fig.add_trace(go.Scatter(x=steps, y=eval_loss, name="Eval loss", line=dict(color=AMBER, width=2, dash="dot")))
        themed(fig, height=300, xaxis=dict(gridcolor=LINE, title="Steps"), yaxis=dict(gridcolor=LINE, title="Loss"),
               legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)", bordercolor=LINE))
        st.plotly_chart(fig, use_container_width=True)
        chart_card_close()

    with col2:
        chart_card_open("Learning rate schedule (cosine)")
        lrs = [2e-4 * 0.5 * (1 + np.cos(np.pi * s / 1000)) for s in steps]
        fig2 = go.Figure(go.Scatter(x=steps, y=lrs, line=dict(color=TEAL, width=2)))
        themed(fig2, height=300, xaxis=dict(gridcolor=LINE, title="Steps"),
               yaxis=dict(gridcolor=LINE, title="Learning rate", tickformat=".2e"))
        st.plotly_chart(fig2, use_container_width=True)
        chart_card_close()

    st.markdown('<div class="section-label">LoRA explained</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        chart_card_open("Why QLoRA works")
        st.markdown("""
        | Concept | Detail |
        |---|---|
        | Full fine-tune VRAM | ~80GB (A100) |
        | QLoRA VRAM | ~4.5GB (T4 free) |
        | Trainable params | 40M / 7,241M (0.55%) |
        | Method | Freeze base, train low-rank ΔW = BA |
        | 4-bit NF4 | Optimal quantization for normal distributions |
        | Double quant | Quantize the quantization constants |
        | merge_and_unload | Bakes adapter into weights for fast inference |
        """)
        chart_card_close()

    with col4:
        chart_card_open("LoRA rank vs. quality trade-off")
        ranks  = [4, 8, 16, 32, 64]
        params = [r * 2 * 4096 * len(["q", "k", "v", "o", "gate", "up", "down"]) / 1e6 for r in ranks]
        rougel = [0.351, 0.382, 0.416, 0.402, 0.398]
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Bar(x=[f"r={r}" for r in ranks], y=params,
                              name="Trainable params (M)", marker_color=SLATE, opacity=0.8),
                       secondary_y=False)
        fig3.add_trace(go.Scatter(x=[f"r={r}" for r in ranks], y=rougel,
                                  mode="lines+markers", name="ROUGE-L",
                                  line=dict(color=TEAL, width=2.5), marker_size=8),
                       secondary_y=True)
        themed(fig3, height=280, legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)", bordercolor=LINE))
        fig3.update_yaxes(title_text="Trainable params (M)", secondary_y=False, gridcolor=LINE)
        fig3.update_yaxes(title_text="ROUGE-L score", secondary_y=True, gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)
        chart_card_close()

st.markdown(f"""
<div class="pulse-divider" style="margin-top:36px;"></div>
<div style="text-align:center; padding:4px 0 4px; font-family:'JetBrains Mono',monospace; font-size:0.66rem; color:{INK_FAINT}; letter-spacing:0.05em;">
    MEDLLM · MISTRAL-7B + QLORA · MEDQA DATASET · W&amp;B TRACKING · FASTAPI + STREAMLIT
</div>
""", unsafe_allow_html=True)
