import streamlit as st
import numpy     as np
import pandas    as pd
import plotly.graph_objects as go
import plotly.express       as px
from plotly.subplots import make_subplots

PALETTE = {
    "primary"      : "#6C63FF",   
    "secondary"    : "#FF6584",   
    "accent"       : "#43D9AD",   
    "warning"      : "#FFC300",   
    "danger"       : "#FF4560",   
    "success"      : "#00E396",   
    "background"   : "#0F1117",   
    "card_bg"      : "#1A1C24",   
    "card_border"  : "#2E3044",   
    "text_primary" : "#EAEAEA",
    "text_muted"   : "#9B9BB4",
    "gradient_start": "#6C63FF",
    "gradient_end"  : "#FF6584",
}

RISK_COLORS = {
    "Low"   : PALETTE["success"],
    "Medium": PALETTE["warning"],
    "High"  : PALETTE["danger"],
}

RISK_ICONS = {
    "Low"   : "✅",
    "Medium": "⚠️",
    "High"  : "🚨",
}

def inject_css() -> None:
    """Inject all custom CSS into the Streamlit page."""
    css = f"""
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Root & Global ── */
    :root {{
        --primary     : {PALETTE['primary']};
        --secondary   : {PALETTE['secondary']};
        --accent      : {PALETTE['accent']};
        --warning     : {PALETTE['warning']};
        --danger      : {PALETTE['danger']};
        --success     : {PALETTE['success']};
        --bg          : {PALETTE['background']};
        --card-bg     : {PALETTE['card_bg']};
        --border      : {PALETTE['card_border']};
        --text        : {PALETTE['text_primary']};
        --text-muted  : {PALETTE['text_muted']};
    }}

    html, body, [class*="css"] {{
        font-family : 'Inter', sans-serif !important;
        color       : var(--text) !important;
    }}

    .stApp {{
        background  : var(--bg) !important;
    }}

    /* ── Hide Streamlit Branding ── */
    #MainMenu, footer, header {{ visibility: hidden; }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background     : var(--card-bg) !important;
        border-right   : 1px solid var(--border) !important;
        padding-top    : 1rem;
    }}
    [data-testid="stSidebar"] * {{
        color : var(--text) !important;
    }}

    /* ── Hero Header ── */
    .hero-header {{
        background      : linear-gradient(135deg, {PALETTE['gradient_start']}22, {PALETTE['gradient_end']}22);
        border          : 1px solid {PALETTE['card_border']};
        border-radius   : 16px;
        padding         : 2rem 2.5rem;
        margin-bottom   : 1.5rem;
        position        : relative;
        overflow        : hidden;
    }}
    .hero-header::before {{
        content         : '';
        position        : absolute;
        top             : -50%;
        left            : -10%;
        width           : 300px;
        height          : 300px;
        background      : radial-gradient({PALETTE['primary']}33, transparent 70%);
        border-radius   : 50%;
    }}
    .hero-title {{
        font-size       : 2.2rem;
        font-weight     : 800;
        background      : linear-gradient(90deg, {PALETTE['gradient_start']}, {PALETTE['gradient_end']});
        -webkit-background-clip : text;
        -webkit-text-fill-color : transparent;
        margin          : 0;
        line-height     : 1.2;
    }}
    .hero-subtitle {{
        font-size       : 1rem;
        color           : var(--text-muted);
        margin-top      : 0.5rem;
    }}

    /* ── Metric Cards ── */
    .metric-card {{
        background      : var(--card-bg);
        border          : 1px solid var(--border);
        border-radius   : 12px;
        padding         : 1.25rem 1.5rem;
        text-align      : center;
        transition      : transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .metric-card:hover {{
        transform       : translateY(-3px);
        box-shadow      : 0 8px 24px rgba(108,99,255,0.18);
    }}
    .metric-label {{
        font-size       : 0.78rem;
        font-weight     : 600;
        text-transform  : uppercase;
        letter-spacing  : 0.08em;
        color           : var(--text-muted);
        margin-bottom   : 0.4rem;
    }}
    .metric-value {{
        font-size       : 2rem;
        font-weight     : 800;
        line-height     : 1;
    }}
    .metric-delta {{
        font-size       : 0.8rem;
        margin-top      : 0.3rem;
        color           : var(--text-muted);
    }}

    /* ── Risk Badge ── */
    .risk-badge {{
        display         : inline-flex;
        align-items     : center;
        gap             : 0.4rem;
        padding         : 0.35rem 0.9rem;
        border-radius   : 999px;
        font-weight     : 700;
        font-size       : 0.9rem;
        letter-spacing  : 0.04em;
    }}
    .risk-low    {{ background: {PALETTE['success']}22; color: {PALETTE['success']}; border: 1px solid {PALETTE['success']}55; }}
    .risk-medium {{ background: {PALETTE['warning']}22; color: {PALETTE['warning']}; border: 1px solid {PALETTE['warning']}55; }}
    .risk-high   {{ background: {PALETTE['danger']}22;  color: {PALETTE['danger']};  border: 1px solid {PALETTE['danger']}55; }}

    /* ── Result Card ── */
    .result-card {{
        background      : var(--card-bg);
        border          : 1px solid var(--border);
        border-radius   : 16px;
        padding         : 2rem;
        margin-top      : 1rem;
    }}
    .result-prob {{
        font-size       : 3.5rem;
        font-weight     : 800;
        line-height     : 1;
    }}

    /* ── Section Title ── */
    .section-title {{
        font-size       : 1.1rem;
        font-weight     : 700;
        color           : var(--text);
        border-left     : 3px solid var(--primary);
        padding-left    : 0.7rem;
        margin          : 1.5rem 0 0.8rem;
    }}

    /* ── Gauge Container ── */
    .gauge-container {{
        display         : flex;
        justify-content : center;
        align-items     : center;
    }}

    /* ── Input Labels ── */
    .stSlider label, .stSelectbox label, .stNumberInput label {{
        font-size : 0.82rem !important;
        font-weight : 500 !important;
        color : var(--text-muted) !important;
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        background      : var(--card-bg);
        border-radius   : 8px;
        padding         : 4px;
        gap             : 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background      : transparent;
        border-radius   : 6px;
        color           : var(--text-muted) !important;
        font-weight     : 500;
        padding         : 0.4rem 1rem;
    }}
    .stTabs [aria-selected="true"] {{
        background      : var(--primary) !important;
        color           : white !important;
    }}

    /* ── Buttons ── */
    .stButton > button {{
        background      : linear-gradient(135deg, var(--primary), {PALETTE['secondary']});
        color           : white !important;
        border          : none !important;
        border-radius   : 8px !important;
        font-weight     : 600 !important;
        padding         : 0.55rem 2rem !important;
        transition      : opacity 0.2s, transform 0.2s;
        width           : 100%;
    }}
    .stButton > button:hover {{
        opacity         : 0.9;
        transform       : translateY(-1px);
    }}

    /* ── Data Tables ── */
    .stDataFrame {{
        border          : 1px solid var(--border) !important;
        border-radius   : 8px !important;
    }}

    /* ── Progress bar ── */
    .stProgress > div > div {{
        background      : linear-gradient(90deg, var(--primary), var(--secondary)) !important;
        border-radius   : 999px !important;
    }}

    /* ── Tooltip / Info boxes ── */
    .info-box {{
        background      : {PALETTE['primary']}15;
        border          : 1px solid {PALETTE['primary']}44;
        border-radius   : 8px;
        padding         : 0.75rem 1rem;
        font-size       : 0.87rem;
        color           : {PALETTE['text_primary']};
    }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar       {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg); }}
    ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str) -> None:
    """Render the top hero header."""
    st.markdown(f"""
    <div class="hero-header">
        <p class="hero-title">{title}</p>
        <p class="hero-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, delta: str = "",
                        color: str = PALETTE["primary"]) -> None:
    """Render a styled metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        {"<div class='metric-delta'>" + delta + "</div>" if delta else ""}
    </div>
    """, unsafe_allow_html=True)


def render_risk_badge(risk_tier: str) -> None:
    """Render a colour-coded risk badge."""
    css_class = f"risk-{risk_tier.lower()}"
    icon      = RISK_ICONS.get(risk_tier, "")
    st.markdown(f"""
    <span class="risk-badge {css_class}">{icon} {risk_tier} Risk</span>
    """, unsafe_allow_html=True)


def render_section_title(title: str) -> None:
    """Render a section header with left accent bar."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def render_info_box(text: str) -> None:
    """Render an informational callout box."""
    st.markdown(f'<div class="info-box">ℹ️ &nbsp;{text}</div>', unsafe_allow_html=True)


def render_result_card(prob: float, risk: str, prediction: int) -> None:
    """Render the full prediction result card."""
    color = RISK_COLORS.get(risk, PALETTE["primary"])
    icon  = RISK_ICONS.get(risk, "")
    label = "⚠ LIKELY TO CHURN" if prediction else "✓ LIKELY TO STAY"
    label_color = PALETTE["danger"] if prediction else PALETTE["success"]

    st.markdown(f"""
    <div class="result-card">
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem;">
            <div>
                <div class="metric-label">Churn Probability</div>
                <div class="result-prob" style="color:{color}">{prob:.1%}</div>
                <div style="margin-top:0.5rem; font-size:1.1rem; font-weight:700; color:{label_color}">
                    {label}
                </div>
            </div>
            <div style="text-align:right;">
                <div class="metric-label">Risk Tier</div>
                <span class="risk-badge risk-{risk.lower()}" style="font-size:1.1rem; padding:0.5rem 1.2rem;">
                    {icon} {risk}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _fig_defaults(fig: go.Figure, height: int = 350) -> go.Figure:
    """Apply dark-theme defaults to any Plotly figure."""
    fig.update_layout(
        height           = height,
        paper_bgcolor    = "rgba(0,0,0,0)",
        plot_bgcolor     = "rgba(0,0,0,0)",
        font             = dict(family="Inter", color=PALETTE["text_primary"], size=12),
        legend           = dict(bgcolor="rgba(0,0,0,0)", bordercolor=PALETTE["card_border"]),
        margin           = dict(t=30, b=20, l=20, r=20),
        xaxis            = dict(gridcolor=PALETTE["card_border"], linecolor=PALETTE["card_border"]),
        yaxis            = dict(gridcolor=PALETTE["card_border"], linecolor=PALETTE["card_border"]),
    )
    return fig


def gauge_chart(probability: float, risk_tier: str) -> go.Figure:
    """Animated gauge chart for churn probability."""
    color = RISK_COLORS.get(risk_tier, PALETTE["primary"])
    fig   = go.Figure(go.Indicator(
        mode   = "gauge+number+delta",
        value  = probability * 100,
        delta  = {"reference": 50, "suffix": "%",
                   "increasing": {"color": PALETTE["danger"]},
                   "decreasing": {"color": PALETTE["success"]}},
        number = {"suffix": "%", "font": {"size": 40, "color": color}},
        gauge  = {
            "axis"      : {"range": [0, 100], "tickwidth": 1,
                           "tickcolor": PALETTE["text_muted"],
                           "tickfont": {"color": PALETTE["text_muted"]}},
            "bar"       : {"color": color, "thickness": 0.3},
            "bgcolor"   : "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps"     : [
                {"range": [0, 35],  "color": "rgba(0, 227, 150, 0.13)"},
                {"range": [35, 65], "color": "rgba(255, 195, 0, 0.13)"},
                {"range": [65, 100],"color": "rgba(255, 69, 96, 0.13)"},
            ],
            "threshold" : {
                "line" : {"color": "white", "width": 3},
                "thickness": 0.85,
                "value": probability * 100,
            },
        },
        title  = {"text": "Churn Probability",
                  "font": {"size": 14, "color": PALETTE["text_muted"]}},
    ))
    fig.update_layout(
        height        = 280,
        paper_bgcolor = "rgba(0,0,0,0)",
        font          = dict(family="Inter"),
        margin        = dict(t=20, b=20, l=30, r=30),
    )
    return fig


def feature_importance_bar(feature_names: list, importances: list,
                            top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    df  = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df  = df.nlargest(top_n, "Importance").sort_values("Importance")

    colors = [
        PALETTE["primary"] if i > df["Importance"].median()
        else PALETTE["text_muted"]
        for i in df["Importance"]
    ]

    fig = go.Figure(go.Bar(
        x             = df["Importance"],
        y             = df["Feature"],
        orientation   = "h",
        marker_color  = colors,
        hovertemplate = "<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title  = dict(text=f"Top {top_n} Feature Importances",
                      font=dict(size=13, color=PALETTE["text_primary"])),
        xaxis_title = "Importance Score",
    )
    return _fig_defaults(fig, height=400)


def churn_distribution_donut(churn_count: int, stay_count: int) -> go.Figure:
    """Donut chart showing churn vs stay split."""
    fig = go.Figure(go.Pie(
        labels       = ["Stayed", "Churned"],
        values       = [stay_count, churn_count],
        hole         = 0.62,
        marker_colors= [PALETTE["success"], PALETTE["danger"]],
        textfont_size= 13,
        hovertemplate= "<b>%{label}</b><br>Count: %{value}<br>Pct: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        title       = dict(text="Dataset Churn Distribution",
                           font=dict(size=13, color=PALETTE["text_primary"])),
        showlegend  = True,
        annotations = [dict(text=f"{churn_count/(churn_count+stay_count):.1%}<br><span style='font-size:11px'>Churn</span>",
                            x=0.5, y=0.5, font_size=20, showarrow=False,
                            font_color=PALETTE["danger"])],
    )
    return _fig_defaults(fig, height=350)


def model_comparison_bar(results_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing model metrics."""
    metrics = ["ROC-AUC", "F1", "Precision", "Recall"]
    fig     = go.Figure()

    colors  = [PALETTE["primary"], PALETTE["accent"],
               PALETTE["warning"], PALETTE["secondary"]]

    for metric, color in zip(metrics, colors):
        if metric in results_df.columns:
            fig.add_trace(go.Bar(
                name          = metric,
                x             = results_df.index,
                y             = results_df[metric],
                marker_color  = color,
                hovertemplate = f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<extra></extra>",
            ))

    fig.update_layout(
        barmode     = "group",
        title       = dict(text="Model Comparison — Key Metrics",
                           font=dict(size=13, color=PALETTE["text_primary"])),
        xaxis_tickangle = -25,
        xaxis_title = "Model",
        yaxis_title = "Score",
    )
    return _fig_defaults(fig, height=400)


def risk_breakdown_bar(risk_counts: dict) -> go.Figure:
    """Bar chart of customer risk tier breakdown."""
    tiers  = ["Low", "Medium", "High"]
    counts = [risk_counts.get(t, 0) for t in tiers]
    colors = [PALETTE["success"], PALETTE["warning"], PALETTE["danger"]]

    fig = go.Figure(go.Bar(
        x             = tiers,
        y             = counts,
        marker_color  = colors,
        text          = counts,
        textposition  = "outside",
        textfont      = dict(color=PALETTE["text_primary"], size=13),
        hovertemplate = "<b>%{x} Risk</b><br>Customers: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title      = dict(text="Customer Risk Tier Distribution",
                          font=dict(size=13, color=PALETTE["text_primary"])),
        xaxis_title = "Risk Tier",
        yaxis_title = "Number of Customers",
    )
    return _fig_defaults(fig, height=320)


def correlation_heatmap(df: pd.DataFrame, cols: list) -> go.Figure:
    """Plotly heatmap of feature correlations."""
    sub  = df[cols].select_dtypes(include=np.number)
    corr = sub.corr()
    fig  = go.Figure(go.Heatmap(
        z            = corr.values,
        x            = corr.columns.tolist(),
        y            = corr.index.tolist(),
        colorscale   = "RdBu",
        zmid         = 0,
        text         = corr.round(2).values,
        texttemplate = "%{text}",
        textfont     = {"size": 9},
        hovertemplate= "x: %{x}<br>y: %{y}<br>corr: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title  = dict(text="Feature Correlation Matrix",
                      font=dict(size=13, color=PALETTE["text_primary"])),
        height = 500,
        paper_bgcolor = "rgba(0,0,0,0)",
        font   = dict(family="Inter", color=PALETTE["text_primary"]),
        margin = dict(t=40, b=20, l=20, r=20),
    )
    return fig


def probability_histogram(probabilities: np.ndarray, labels: np.ndarray) -> go.Figure:
    """Overlapping histogram of predicted churn probabilities by true class."""
    fig = go.Figure()
    for cls, label, color in [(0, "Stayed", PALETTE["success"]),
                               (1, "Churned", PALETTE["danger"])]:
        mask = labels == cls
        fig.add_trace(go.Histogram(
            x           = probabilities[mask],
            name        = label,
            opacity     = 0.65,
            nbinsx      = 40,
            marker_color= color,
            hovertemplate = f"<b>{label}</b><br>Prob: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
        ))
    fig.update_layout(
        barmode     = "overlay",
        title       = dict(text="Predicted Probability Distribution by True Class",
                           font=dict(size=13, color=PALETTE["text_primary"])),
        xaxis_title = "Predicted Churn Probability",
        yaxis_title = "Count",
    )
    return _fig_defaults(fig, height=320)


def shap_waterfall_mock(feature_names: list, shap_values: list,
                         base_value: float) -> go.Figure:
    """
    Waterfall chart showing SHAP contribution of each feature.
    If real SHAP values are not available, mock values are shown.
    """
    df_s = pd.DataFrame({"Feature": feature_names, "SHAP": shap_values})
    df_s = df_s.reindex(df_s["SHAP"].abs().sort_values(ascending=False).index).head(12)
    df_s = df_s.sort_values("SHAP")

    colors = [PALETTE["danger"] if v > 0 else PALETTE["success"] for v in df_s["SHAP"]]

    fig = go.Figure(go.Bar(
        x             = df_s["SHAP"],
        y             = df_s["Feature"],
        orientation   = "h",
        marker_color  = colors,
        hovertemplate = "<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=PALETTE["text_muted"], line_width=1)
    fig.update_layout(
        title  = dict(text="Feature Impact on Prediction (SHAP-style)",
                      font=dict(size=13, color=PALETTE["text_primary"])),
        xaxis_title = "SHAP Value (+ → increases churn risk)",
    )
    return _fig_defaults(fig, height=380)


def render_sidebar_header() -> None:
    """Render sidebar logo and navigation help."""
    st.sidebar.markdown(f"""
    <div style="text-align:center; padding:1rem 0 1.5rem;">
        <div style="font-size:2.5rem;">🔮</div>
        <div style="font-size:1.1rem; font-weight:800;
                    background:linear-gradient(90deg,{PALETTE['primary']},{PALETTE['secondary']});
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            ChurnLens AI
        </div>
        <div style="font-size:0.75rem; color:{PALETTE['text_muted']}; margin-top:0.2rem;">
            Powered by Stacking Ensemble
        </div>
    </div>
    <hr style="border-color:{PALETTE['card_border']}; margin:0 0 1rem;">
    """, unsafe_allow_html=True)


def render_sidebar_stats(roc_auc: float, f1: float, train_size: int) -> None:
    """Display model performance stats in the sidebar."""
    st.sidebar.markdown("**🏅 Model Performance**")
    st.sidebar.metric("ROC-AUC", f"{roc_auc:.4f}", help="Area under the ROC curve on held-out test set")
    st.sidebar.metric("F1 Score", f"{f1:.4f}",    help="Harmonic mean of precision and recall")
    st.sidebar.metric("Training Samples", f"{train_size:,}")
    st.sidebar.markdown(f"""
    <hr style="border-color:{PALETTE['card_border']}">
    <div style="font-size:0.75rem; color:{PALETTE['text_muted']};">
        Model: Stacking Ensemble<br>
        Base: RF · XGB · LGBM · GB<br>
        Meta: Logistic Regression
    </div>
    """, unsafe_allow_html=True)