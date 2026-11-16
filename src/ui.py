"""UI styling, glassmorphic design system, and custom Plotly visual theme.

Provides centralized CSS injection, responsive KPI cards, animated hero badges,
custom Plotly dark-mode templates, and telecommunications architecture flow graphics.
"""

import textwrap
from typing import Any, Dict, List, Optional
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.config import COLORS


def apply_custom_css() -> None:
    """Inject polished 'futuristic telecom control room' CSS styling."""
    css = textwrap.dedent(f"""
    <style>
    /* Google Fonts with safe offline fallbacks */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {{
        --font-heading: 'Space Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --color-cyan: {COLORS['cyan']};
        --color-violet: {COLORS['violet']};
        --color-emerald: {COLORS['emerald']};
        --color-amber: {COLORS['amber']};
        --color-rose: {COLORS['rose']};
        --color-bg-card: {COLORS['bg_card']};
        --border-subtle: {COLORS['border_subtle']};
    }}

    /* Global Body and Background Gradient */
    .stApp {{
        background: linear-gradient(135deg, #0B1020 0%, #070A14 100%) !important;
        font-family: var(--font-body) !important;
        color: #F8FAFC !important;
    }}

    /* Hide Streamlit Header & Footer clutter */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* Headings */
    h1, h2, h3, h4, h5, h6 {{
        font-family: var(--font-heading) !important;
        letter-spacing: -0.02em;
        color: #F8FAFC;
    }}

    /* Futuristic Animated Hero Title */
    .hero-title {{
        font-family: var(--font-heading);
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #22D3EE 0%, #A78BFA 50%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.35rem;
        line-height: 1.2;
    }}

    .hero-subtitle {{
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.2rem;
        font-weight: 400;
    }}

    /* Badges / Status Pills */
    .badge-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-bottom: 1.5rem;
    }}

    .badge-pill {{
        display: inline-flex;
        align-items: center;
        padding: 0.28rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        border: 1px solid;
    }}

    .badge-cyan {{
        background: rgba(34, 211, 238, 0.12);
        color: #22D3EE;
        border-color: rgba(34, 211, 238, 0.35);
        box-shadow: 0 0 12px rgba(34, 211, 238, 0.15);
    }}

    .badge-violet {{
        background: rgba(139, 92, 246, 0.12);
        color: #C4B5FD;
        border-color: rgba(139, 92, 246, 0.35);
    }}

    .badge-emerald {{
        background: rgba(52, 211, 153, 0.12);
        color: #34D399;
        border-color: rgba(52, 211, 153, 0.35);
    }}

    .badge-amber {{
        background: rgba(245, 158, 11, 0.12);
        color: #FBBF24;
        border-color: rgba(245, 158, 11, 0.35);
    }}

    /* Glassmorphism Cards */
    .glass-card {{
        background: rgba(19, 27, 46, 0.65);
        border: 1px solid rgba(56, 189, 248, 0.14);
        border-radius: 16px;
        padding: 1.35rem 1.4rem;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.4);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        margin-bottom: 1.1rem;
    }}

    .glass-card:hover {{
        transform: translateY(-2px);
        border-color: rgba(34, 211, 238, 0.32);
        box-shadow: 0 14px 36px rgba(34, 211, 238, 0.12);
    }}

    /* KPI Cards */
    .kpi-card {{
        background: rgba(19, 27, 46, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.12);
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(12px);
        margin-bottom: 0.8rem;
    }}

    .kpi-accent-bar {{
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
    }}

    .kpi-title {{
        font-size: 0.82rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }}

    .kpi-value {{
        font-family: var(--font-heading);
        font-size: 1.75rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
    }}

    .kpi-subtext {{
        font-size: 0.78rem;
        color: #64748B;
    }}

    /* Notice box */
    .notice-box {{
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }}

    /* Sidebar Custom Styling */
    section[data-testid="stSidebar"] {{
        background: #090E1A !important;
        border-right: 1px solid rgba(56, 189, 248, 0.1) !important;
    }}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1.5rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }}

    .stTabs [data-baseweb="tab"] {{
        font-family: var(--font-heading);
        font-weight: 600;
        color: #94A3B8;
        padding: 0.6rem 0.2rem;
    }}

    .stTabs [aria-selected="true"] {{
        color: #22D3EE !important;
        border-bottom-color: #22D3EE !important;
    }}

    /* Button Styling */
    .stButton>button {{
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.4rem !important;
        font-weight: 600 !important;
        font-family: var(--font-heading) !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25) !important;
        transition: all 0.2s ease !important;
    }}

    .stButton>button:hover {{
        background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%) !important;
        border-color: #38BDF8 !important;
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.35) !important;
        transform: translateY(-1px) !important;
    }}
    </style>
    """).strip()
    st.markdown(css, unsafe_allow_html=True)


def format_plotly_chart(
    fig: go.Figure,
    title: Optional[str] = None,
    height: int = 420,
    show_legend: bool = True,
) -> go.Figure:
    """Apply the custom Dark Telecom Control Room template to a Plotly figure."""
    if title and show_legend:
        # Separate title (top) and legend (bottom) to prevent overlapping in narrow/multi-column layouts
        legend_layout = dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        )
        margin_t = 48
        margin_b = 52
    elif show_legend:
        legend_layout = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        )
        margin_t = 30
        margin_b = 35
    else:
        legend_layout = dict(visible=False)
        margin_t = 45 if title else 25
        margin_b = 35

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11, 16, 32, 0.4)",
        margin=dict(l=35, r=25, t=margin_t, b=margin_b),
        height=height,
        font=dict(family="Inter, -apple-system, sans-serif", color="#E2E8F0", size=12),
        hoverlabel=dict(
            bgcolor="#131B2E",
            bordercolor="#38BDF8",
            font=dict(family="Inter, sans-serif", color="#F8FAFC", size=12),
        ),
        legend=legend_layout,
    )

    if title:
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(family="Space Grotesk, sans-serif", size=14, color="#F8FAFC"),
                x=0.01,
                y=0.98,
                xanchor="left",
                yanchor="top",
            )
        )

    fig.update_xaxes(
        gridcolor="rgba(148, 163, 184, 0.08)",
        linecolor="rgba(148, 163, 184, 0.18)",
        zerolinecolor="rgba(148, 163, 184, 0.12)",
    )
    fig.update_yaxes(
        gridcolor="rgba(148, 163, 184, 0.08)",
        linecolor="rgba(148, 163, 184, 0.18)",
        zerolinecolor="rgba(148, 163, 184, 0.12)",
    )

    return fig


def render_hero(
    title: str,
    subtitle: str,
    badges: Optional[List[Dict[str, str]]] = None,
) -> None:
    """Render consistent animated hero header."""
    badge_html = ""
    if badges:
        badge_html = '<div class="badge-container">'
        for b in badges:
            cls = b.get("class", "badge-cyan")
            label = b.get("label", "")
            badge_html += f'<span class="badge-pill {cls}">{label}</span>'
        badge_html += "</div>"

    html = textwrap.dedent(f"""
        <div style="margin-bottom: 1.5rem;">
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
            {badge_html}
        </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)


def render_kpi(
    col: Any,
    title: str,
    value: str,
    subtext: str = "",
    accent_color: str = "#22D3EE",
) -> None:
    """Render a KPI metric tile with a colored accent top bar."""
    html = textwrap.dedent(f"""
        <div class="kpi-card">
            <div class="kpi-accent-bar" style="background: {accent_color};"></div>
            <div class="kpi-title">{title}</div>
            <div class="kpi-value" style="color: {accent_color};">{value}</div>
            <div class="kpi-subtext">{subtext}</div>
        </div>
    """).strip()
    col.markdown(html, unsafe_allow_html=True)


def render_architecture_diagram() -> None:
    """Render a clean SVG/HTML telecommunications architecture flow graphic."""
    svg_html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body {
  margin: 0;
  padding: 0;
  background: transparent;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
.card-wrap {
  background: rgba(19, 27, 46, 0.85);
  border: 1px solid rgba(56, 189, 248, 0.22);
  border-radius: 14px;
  padding: 1rem 1.2rem;
  box-sizing: border-box;
}
.title {
  font-size: 0.85rem;
  font-weight: 700;
  color: #22D3EE;
  margin-bottom: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
</style>
</head>
<body>
<div class="card-wrap">
  <div class="title">System Architecture &amp; Optimization Dataflow</div>
  <svg viewBox="0 0 920 120" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" style="max-height: 120px; display: block;">
    <defs>
      <linearGradient id="g-cyan" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#22D3EE" stop-opacity="0.25"/>
        <stop offset="100%" stop-color="#0284C7" stop-opacity="0.45"/>
      </linearGradient>
      <linearGradient id="g-violet" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#A78BFA" stop-opacity="0.25"/>
        <stop offset="100%" stop-color="#7C3AED" stop-opacity="0.45"/>
      </linearGradient>
      <linearGradient id="g-emerald" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#34D399" stop-opacity="0.25"/>
        <stop offset="100%" stop-color="#059669" stop-opacity="0.45"/>
      </linearGradient>
      <marker id="arr" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1 L 8 5 L 0 9 z" fill="#38BDF8"/>
      </marker>
    </defs>

    <!-- Node 1: Base Station Telemetry -->
    <rect x="5" y="10" width="155" height="95" rx="10" fill="url(#g-cyan)" stroke="#38BDF8" stroke-width="1.5"/>
    <text x="82" y="42" fill="#F8FAFC" font-size="13" font-weight="bold" text-anchor="middle">5G gNodeB Cells</text>
    <text x="82" y="64" fill="#94A3B8" font-size="10.5" text-anchor="middle">PRB Load • Traffic GB</text>
    <text x="82" y="82" fill="#94A3B8" font-size="10.5" text-anchor="middle">Users • Carriers • Temp</text>

    <line x1="165" y1="58" x2="195" y2="58" stroke="#38BDF8" stroke-width="2" marker-end="url(#arr)"/>

    <!-- Node 2: Preprocessing & Lags -->
    <rect x="200" y="10" width="155" height="95" rx="10" fill="url(#g-cyan)" stroke="#38BDF8" stroke-width="1.5"/>
    <text x="277" y="42" fill="#F8FAFC" font-size="13" font-weight="bold" text-anchor="middle">Causal Pipeline</text>
    <text x="277" y="64" fill="#94A3B8" font-size="10.5" text-anchor="middle">Time-based Imputation</text>
    <text x="277" y="82" fill="#94A3B8" font-size="10.5" text-anchor="middle">Lags (1h..168h) • Shifted</text>

    <line x1="360" y1="58" x2="390" y2="58" stroke="#38BDF8" stroke-width="2" marker-end="url(#arr)"/>

    <!-- Node 3: ML Engine -->
    <rect x="395" y="10" width="150" height="95" rx="10" fill="url(#g-violet)" stroke="#A78BFA" stroke-width="1.5"/>
    <text x="470" y="42" fill="#F8FAFC" font-size="13" font-weight="bold" text-anchor="middle">ML Regressors</text>
    <text x="470" y="64" fill="#DDD6FE" font-size="10.5" text-anchor="middle">Ridge • Random Forest</text>
    <text x="470" y="82" fill="#DDD6FE" font-size="10.5" text-anchor="middle">XGBoost (R² &gt; 0.95)</text>

    <line x1="550" y1="58" x2="580" y2="58" stroke="#A78BFA" stroke-width="2" marker-end="url(#arr)"/>

    <!-- Node 4: Forecasting & Peak Alerting -->
    <rect x="585" y="10" width="150" height="95" rx="10" fill="url(#g-violet)" stroke="#A78BFA" stroke-width="1.5"/>
    <text x="660" y="42" fill="#F8FAFC" font-size="13" font-weight="bold" text-anchor="middle">Recursive Forecast</text>
    <text x="660" y="64" fill="#DDD6FE" font-size="10.5" text-anchor="middle">24h / 48h / 168h Ahead</text>
    <text x="660" y="82" fill="#DDD6FE" font-size="10.5" text-anchor="middle">90th %ile Peak Windows</text>

    <line x1="740" y1="58" x2="770" y2="58" stroke="#34D399" stroke-width="2" marker-end="url(#arr)"/>

    <!-- Node 5: Green Telecom Savings -->
    <rect x="775" y="10" width="140" height="95" rx="10" fill="url(#g-emerald)" stroke="#34D399" stroke-width="1.5"/>
    <text x="845" y="42" fill="#34D399" font-size="13" font-weight="bold" text-anchor="middle">Energy Saver</text>
    <text x="845" y="64" fill="#A7F3D0" font-size="10.5" text-anchor="middle">Off-Peak Sleep Mode</text>
    <text x="845" y="82" fill="#A7F3D0" font-size="10.5" text-anchor="middle">Cost &amp; CO₂ Reductions</text>
  </svg>
</div>
</body>
</html>"""
    components.html(svg_html, height=185)
