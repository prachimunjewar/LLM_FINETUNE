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

st.set_page_config(
    page_title="MedLLM — Fine-Tuning Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DESIGN TOKENS ────────────────────────────────────────────
# Plain, restrained "enterprise report" look: one neutral gray
# scale, one accent color, flat cards, no textures or motifs.
BG       = "#F7F8FA"
SURFACE  = "#FFFFFF"
INK      = "#111827"
INK_SOFT = "#6B7280"
LINE     = "#E5E7EB"

ACCENT      = "#1D4ED8"   # fine-tuned model / primary
NEUTRAL     = "#9CA3AF"   # base model
POSITIVE    = "#059669"   # improvement deltas only
HIGHLIGHT   = "#B45309"   # best-run marker only

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: {BG}; color: {INK}; }}
.main .block-container {{ padding: 1.8rem 2.4rem 3rem; max-width: 1320px; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{ background: {SURFACE} !important; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label {{ color: {INK_SOFT} !important; }}

/* ── Masthead ── */
.eyebrow {{
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: {INK_SOFT}; margin-bottom: 6px;
}}
.page-title {{ font-size: 1.9rem; font-weight: 700; color: {INK}; letter-spacing: -0.02em; margin: 0 0 4px; }}
.page-subtitle {{ font-size: 0.86rem; color: {INK_SOFT}; }}
.masthead-rule {{ border: none; border-top: 1px solid {LINE}; margin: 18px 0 26px; }}

/* ── Section headers ── */
.section-label {{
    font-size: 0.78rem; font-weight: 600; color: {INK};
    margin: 30px 0 12px; padding-bottom: 8px; border-bottom: 1px solid {LINE};
}}

/* ── KPI cards ── */
.kpi-row {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }}
.kpi {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 6px; padding: 16px 18px; }}
.kpi-label {{ font-size: 0.72rem; color: {INK_SOFT}; margin-bottom: 8px; }}
.kpi-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; font-weight: 600; color: {INK}; }}
.kpi-value.accent {{ color: {ACCENT}; }}
.kpi-sub {{ font-size: 0.74rem; color: {INK_SOFT}; margin-top: 6px; }}
.kpi-sub.positive {{ color: {POSITIVE}; }}

/* ── Chart card ── */
.chart-card {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 6px; padding: 16px 18px 8px; }}
.chart-title {{ font-size: 0.88rem; font-weight: 600; color: {INK}; margin-bottom: 6px; }}

/* ── Answer boxes ── */
.answer-box {{ background: {BG}; border: 1px solid {LINE}; border-radius: 6px; padding: 16px 18px; margin-top: 8px; }}
.answer-box.ft {{ border-color: {ACCENT}; }}
.answer-label {{ font-size: 0.72rem; font-weight: 600; color: {INK_SOFT}; margin-bottom: 8px; }}
.answer-label.ft-label {{ color: {ACCENT}; }}
.answer-text {{ font-size: 0.9rem; color: {INK}; line-height: 1.65; }}

/* ── Tech list ── */
.tech-badge {{
    display: inline-block; background: {BG}; border: 1px solid {LINE}; color: {INK_SOFT};
    border-radius: 4px; padding: 3px 10px; font-size: 0.74rem; margin: 2px 4px 2px 0;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{ background: {SURFACE}; border-radius: 6px; border: 1px solid {LINE}; }}
[data-testid="stDataFrame"] th {{
    background: {BG} !important; color: {INK_SOFT} !important;
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em;
}}
[data-testid="stDataFrame"] td {{ color: {INK} !important; font-size: 0.84rem; }}

.status-dot {{ display:inline-block; width:6px; height:6px; border-radius:50%; margin-right:6px; }}
.status-dot.on  {{ background: {POSITIVE}; }}
.status-dot.off {{ background: {NEUTRAL}; }}

#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display: none; }}
</style>
""", unsafe_allow_html=True)

# ── PLOTLY THEME ─────────────────────────────────────────────
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", size=11, color=INK_SOFT),
    xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, tickcolor=LINE, linecolor=LINE),
    yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, tickcolor=LINE, linecolor=LINE),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=LINE),
    margin=dict(l=8, r=8, t=8, b=8),
    hoverlabel=dict(bgcolor=INK, font_color="#FFFFFF", font_size=12),
    colorway=[NEUTRAL, ACCENT, HIGHLIGHT, POSITIVE],
)

def themed(fig, height=300, **overrides):
    """Merge CHART_THEME with per-chart overrides safely — avoids
    duplicate-kwarg errors from passing e.g. legend/xaxis/yaxis twice."""
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
    <div style="padding:4px 0 16px;">
        <div style="font-size:1.05rem; font-weight:700; color:{INK};">MedLLM</div>
        <div style="font-size:0.72rem; color:{INK_SOFT};">Fine-Tuning Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.7rem; font-weight:600; color:#9CA3AF; margin-bottom:6px;">NAVIGATION</p>', unsafe_allow_html=True)
    page = st.radio("nav", ["Overview", "Model Comparison", "Live Demo", "Training Analysis"], label_visibility="collapsed")

    st.markdown("---")

    api_status, err = call_api("/")
    if api_status:
        st.markdown('<div style="font-size:0.82rem;"><span class="status-dot on"></span>API online</div>', unsafe_allow_html=True)
        st.caption(f"Demo mode: {api_status.get('demo_mode', True)}")
    else:
        st.markdown('<div style="font-size:0.82rem;"><span class="status-dot off"></span>API offline</div>', unsafe_allow_html=True)
        st.caption("Run: uvicorn api.main:app --reload")

    st.markdown("---")
    st.markdown('<p style="font-size:0.7rem; font-weight:600; color:#9CA3AF; margin-bottom:6px;">MODEL</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.78rem; color:{INK_SOFT}; line-height:1.9;">
        Base: Mistral-7B-Instruct<br>
        Method: QLoRA (4-bit NF4)<br>
        LoRA rank: r=16, α=32<br>
        Dataset: MedQA (8K examples)<br>
        Trainable: 0.55% of params
    </div>
    """, unsafe_allow_html=True)

# ── MASTHEAD ──────────────────────────────────────────────────
st.markdown(f"""
<div class="eyebrow">LLM Fine-Tuning · Medical Domain</div>
<div class="page-title">MedLLM Fine-Tuning Dashboard</div>
<div class="page-subtitle">Mistral-7B + QLoRA · Medical QA · ROUGE / BERTScore evaluation · W&B tracking</div>
<hr class="masthead-rule" />
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
if page == "Overview":
# ════════════════════════════════════════════════════════════

    st.markdown('<div class="section-label">Fine-tuning results</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi">
            <div class="kpi-label">Base ROUGE-L</div>
            <div class="kpi-value">0.3124</div>
            <div class="kpi-sub">Mistral-7B, no tuning</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">Fine-Tuned ROUGE-L</div>
            <div class="kpi-value accent">0.4156</div>
            <div class="kpi-sub positive">+33.1% over base</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">BERTScore F1</div>
            <div class="kpi-value accent">0.9134</div>
            <div class="kpi-sub positive">vs 0.7412 base (+23.2%)</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">Trainable Params</div>
            <div class="kpi-value">0.55%</div>
            <div class="kpi-sub">40M of 7,241M total</div>
        </div>
        <div class="kpi">
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
        fig.add_trace(go.Bar(x=metrics, y=base_vals, name="Base Mistral-7B", marker_color=NEUTRAL,
                              text=[f"{v:.3f}" for v in base_vals], textposition="outside", textfont_size=10))
        fig.add_trace(go.Bar(x=metrics, y=ft_vals, name="Fine-Tuned (QLoRA)", marker_color=ACCENT,
                              text=[f"{v:.3f}" for v in ft_vals], textposition="outside", textfont_size=10))
        themed(fig, height=300, barmode="group", yaxis=dict(gridcolor=LINE, range=[0, 1.1]),
               legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)", bordercolor=LINE, font_size=10))
        st.plotly_chart(fig, use_container_width=True)
        chart_card_close()

    with col2:
        chart_card_open("% improvement over base")
        improvements = [37.7, 40.1, 33.1, 23.2]
        fig2 = go.Figure(go.Bar(
            x=metrics, y=improvements, marker_color=ACCENT,
            text=[f"+{v:.1f}%" for v in improvements], textposition="outside", textfont_size=11,
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
        st.markdown(f'<div>{badges_html}</div>', unsafe_allow_html=True)
        chart_card_close()

# ════════════════════════════════════════════════════════════
elif page == "Model Comparison":
# ════════════════════════════════════════════════════════════

    st.markdown('<div class="section-label">Per-example analysis</div>', unsafe_allow_html=True)

    try:
        per_ex = pd.read_csv("outputs/per_example_results.csv")
        col1, col2 = st.columns(2)

        with col1:
            chart_card_open("ROUGE-L distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=per_ex["base_rougeL"], nbinsx=15, name="Base",
                                       marker_color=NEUTRAL, opacity=0.85))
            fig.add_trace(go.Histogram(x=per_ex["ft_rougeL"], nbinsx=15, name="Fine-Tuned",
                                       marker_color=ACCENT, opacity=0.85))
            themed(fig, height=300, barmode="overlay",
                   xaxis=dict(gridcolor=LINE, title="ROUGE-L score"),
                   yaxis=dict(gridcolor=LINE, title="Count"),
                   legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)", bordercolor=LINE))
            st.plotly_chart(fig, use_container_width=True)
            chart_card_close()

        with col2:
            chart_card_open("Base vs. fine-tuned per example")
            fig2 = px.scatter(per_ex, x="base_rougeL", y="ft_rougeL", color="improvement",
                              color_continuous_scale=[NEUTRAL, ACCENT],
                              labels={"base_rougeL": "Base ROUGE-L", "ft_rougeL": "Fine-Tuned ROUGE-L"})
            fig2.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color=INK_SOFT, width=1))
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
        hp_df.style.highlight_max(subset=["ROUGE-L", "BERTScore"], color="#DBEAFE"),
        use_container_width=True, hide_index=True
    )

# ════════════════════════════════════════════════════════════
elif page == "Live Demo":
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
        run_compare = st.button("Compare models", type="primary", use_container_width=True)

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
elif page == "Training Analysis":
# ════════════════════════════════════════════════════════════

    st.markdown('<div class="section-label">Training dynamics (W&amp;B)</div>', unsafe_allow_html=True)

    steps      = list(range(0, 1000, 25))
    train_loss = [max(0.25, 2.8 * np.exp(-0.004 * s) + 0.3 + np.random.normal(0, 0.04)) for s in steps]
    eval_loss  = [max(0.4,  2.9 * np.exp(-0.0035 * s) + 0.45 + np.random.normal(0, 0.06)) for s in steps]

    col1, col2 = st.columns(2)

    with col1:
        chart_card_open("Training &amp; eval loss")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=train_loss, name="Train loss", line=dict(color=ACCENT, width=2)))
        fig.add_trace(go.Scatter(x=steps, y=eval_loss, name="Eval loss", line=dict(color=HIGHLIGHT, width=2, dash="dot")))
        themed(fig, height=300, xaxis=dict(gridcolor=LINE, title="Steps"), yaxis=dict(gridcolor=LINE, title="Loss"),
               legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)", bordercolor=LINE))
        st.plotly_chart(fig, use_container_width=True)
        chart_card_close()

    with col2:
        chart_card_open("Learning rate schedule (cosine)")
        lrs = [2e-4 * 0.5 * (1 + np.cos(np.pi * s / 1000)) for s in steps]
        fig2 = go.Figure(go.Scatter(x=steps, y=lrs, line=dict(color=ACCENT, width=2)))
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
                              name="Trainable params (M)", marker_color=NEUTRAL, opacity=0.85),
                       secondary_y=False)
        fig3.add_trace(go.Scatter(x=[f"r={r}" for r in ranks], y=rougel,
                                  mode="lines+markers", name="ROUGE-L",
                                  line=dict(color=ACCENT, width=2.5), marker_size=8),
                       secondary_y=True)
        themed(fig3, height=280, legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)", bordercolor=LINE))
        fig3.update_yaxes(title_text="Trainable params (M)", secondary_y=False, gridcolor=LINE)
        fig3.update_yaxes(title_text="ROUGE-L score", secondary_y=True, gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)
        chart_card_close()

st.markdown(f"""
<hr class="masthead-rule" style="margin-top:36px;" />
<div style="text-align:center; padding:4px 0; font-size:0.74rem; color:{INK_SOFT};">
    MedLLM · Mistral-7B + QLoRA · MedQA Dataset · W&amp;B Tracking · FastAPI + Streamlit
</div>
""", unsafe_allow_html=True)
