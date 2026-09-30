"""Shared visual identity: one color/icon per category, used consistently
across every chart and table in the app, plus the dark-theme CSS.

Palette is the validated categorical set (CVD-safe, adjacent-pair checked).
Only 8 hues clear the safety gates, so the two lowest-signal categories
(Education, Others) get neutral grays instead of a 9th/10th cycled hue.
"""

CATEGORY_COLORS: dict[str, str] = {
    "Food": "#3987e5",
    "Transport": "#d95926",
    "Groceries": "#199e70",
    "Shopping": "#c98500",
    "Entertainment": "#d55181",
    "Bills & Utilities": "#3aa33a",
    "Health": "#9085e9",
    "Travel": "#e66767",
    "Education": "#a6a49c",
    "Others": "#6f6e69",
}

CATEGORY_ICONS: dict[str, str] = {
    "Food": "🍔",
    "Transport": "🚗",
    "Groceries": "🛒",
    "Shopping": "🛍️",
    "Entertainment": "🎬",
    "Bills & Utilities": "💡",
    "Health": "🏥",
    "Travel": "✈️",
    "Education": "🎓",
    "Others": "📦",
}

STATUS_COLORS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#e34948",
}

SEQUENTIAL_BLUES = [
    "#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf",
    "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4",
]

SURFACE = "#1a1a19"
PAGE = "#0d0d0d"
PRIMARY_INK = "#ffffff"
SECONDARY_INK = "#c3c2b7"
MUTED_INK = "#898781"
GRIDLINE = "#2c2c2a"
BORDER = "rgba(255,255,255,0.10)"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(color=SECONDARY_INK, family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
    title_font=dict(color=PRIMARY_INK, size=16),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=SECONDARY_INK)),
    margin=dict(t=48, b=32, l=8, r=8),
    xaxis=dict(gridcolor=GRIDLINE, linecolor=GRIDLINE, zerolinecolor=GRIDLINE),
    yaxis=dict(gridcolor=GRIDLINE, linecolor=GRIDLINE, zerolinecolor=GRIDLINE),
)

CUSTOM_CSS = f"""
<style>
.stApp {{
    background-color: {PAGE};
}}
[data-testid="stSidebar"] {{
    background-color: {SURFACE};
    border-right: 1px solid {BORDER};
}}
h1, h2, h3, h4, p, span, label, .stMarkdown {{
    color: {SECONDARY_INK};
}}
h1, h2, h3 {{
    color: {PRIMARY_INK} !important;
}}
.kpi-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 18px 20px;
    height: 100%;
}}
.kpi-label {{
    font-size: 0.78rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: {MUTED_INK};
    margin-bottom: 6px;
}}
.kpi-value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: {PRIMARY_INK};
    line-height: 1.2;
}}
.kpi-sub {{
    font-size: 0.82rem;
    color: {SECONDARY_INK};
    margin-top: 4px;
}}
.badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}}
.badge-good {{ background: rgba(12,163,12,0.18); color: {STATUS_COLORS['good']}; }}
.badge-warning {{ background: rgba(250,178,25,0.18); color: {STATUS_COLORS['warning']}; }}
.badge-critical {{ background: rgba(227,73,72,0.18); color: {STATUS_COLORS['critical']}; }}
.banner {{
    padding: 14px 18px;
    border-radius: 10px;
    font-size: 0.95rem;
    border: 1px solid {BORDER};
    margin: 4px 0 4px 0;
}}
.banner-good {{ background: rgba(12,163,12,0.10); color: {STATUS_COLORS['good']}; }}
.banner-warning {{ background: rgba(250,178,25,0.10); color: {STATUS_COLORS['warning']}; }}
.banner-critical {{ background: rgba(227,73,72,0.10); color: {STATUS_COLORS['critical']}; }}
.banner-info {{ background: rgba(195,194,183,0.08); color: {SECONDARY_INK}; }}
.section-divider {{
    border-top: 1px solid {BORDER};
    margin: 1.2rem 0;
}}
div[data-testid="stMetricValue"] {{
    color: {PRIMARY_INK};
}}
.stTabs [data-baseweb="tab"] {{
    color: {SECONDARY_INK};
}}
.stTabs [aria-selected="true"] {{
    color: {PRIMARY_INK} !important;
}}
</style>
"""


def category_color(category: str) -> str:
    return CATEGORY_COLORS.get(category, MUTED_INK)


def category_label(category: str) -> str:
    icon = CATEGORY_ICONS.get(category, "")
    return f"{icon} {category}".strip()


def themed(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def kpi_card_html(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"{sub_html}"
        f"</div>"
    )


def banner_html(status: str, text: str) -> str:
    cls = {"ok": "good", "warning": "warning", "over": "critical"}.get(status, "info")
    return f'<div class="banner banner-{cls}">{text}</div>'
