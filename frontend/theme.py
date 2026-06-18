"""Charte graphique partagee (editorial sombre) : couleurs, polices, helpers UI.

Palette et typographie alignees sur .streamlit/config.toml. inject() doit etre
appelee une fois dans app.py (le point d'entree multi-pages) pour s'appliquer
a toutes les pages.
"""

from __future__ import annotations

import altair as alt
import streamlit as st

BACKGROUND = "#0F1419"
SURFACE = "#1A2129"
SURFACE_BORDER = "#2A323C"
TEXT_PRIMARY = "#E8EAED"
TEXT_SECONDARY = "#9AA4AF"
ACCENT = "#E63946"

LABEL_COLORS = {
    "World": "#4D9FEC",
    "Sports": "#3DDC97",
    "Business": "#F4A93C",
    "Sci/Tech": "#C792EA",
}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, sans-serif;
}}

h1, h2, h3 {{
    font-family: 'Playfair Display', Georgia, serif !important;
    letter-spacing: -0.01em;
}}

.app-kicker {{
    color: {ACCENT};
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.75rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}}

.app-subtitle {{
    color: {TEXT_SECONDARY};
    font-size: 1.05rem;
    max-width: 50rem;
    line-height: 1.5;
}}

.app-divider {{
    height: 1px;
    background: linear-gradient(90deg, {ACCENT}, transparent);
    border: none;
    margin: 0.75rem 0 1.5rem 0;
}}

.app-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.7rem 1.2rem;
    border-radius: 0.6rem;
    font-size: 1.4rem;
    font-weight: 700;
    font-family: 'Playfair Display', Georgia, serif;
    border: 1px solid;
}}

.app-note {{
    background: {SURFACE};
    border-left: 3px solid {ACCENT};
    border-radius: 0.4rem;
    padding: 0.85rem 1.1rem;
    color: {TEXT_PRIMARY};
    font-size: 0.95rem;
    margin-bottom: 0.6rem;
}}
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def kicker(text: str) -> None:
    st.markdown(f'<div class="app-kicker">{text}</div>', unsafe_allow_html=True)


def subtitle(text: str) -> None:
    st.markdown(f'<p class="app-subtitle">{text}</p>', unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<hr class="app-divider" />', unsafe_allow_html=True)


def badge(text: str, color: str) -> None:
    st.markdown(
        f'<div class="app-badge" style="color:{color};border-color:{color}66;'
        f'background:{color}1A;">{text}</div>',
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f'<div class="app-note">{text}</div>', unsafe_allow_html=True)


def _editorial_dark() -> dict:
    return {
        "config": {
            "background": SURFACE,
            "title": {"color": TEXT_PRIMARY, "font": "Inter", "fontSize": 14},
            "axis": {
                "labelColor": TEXT_SECONDARY,
                "titleColor": TEXT_SECONDARY,
                "gridColor": SURFACE_BORDER,
                "domainColor": SURFACE_BORDER,
                "tickColor": SURFACE_BORDER,
                "labelLimit": 200,
            },
            "legend": {
                "labelColor": TEXT_SECONDARY,
                "titleColor": TEXT_SECONDARY,
            },
            "view": {"stroke": "transparent"},
            "padding": {"left": 10, "right": 10, "top": 10, "bottom": 10},
            "range": {
                "category": list(LABEL_COLORS.values()),
                "heatmap": ["#1A2129", ACCENT],
                "ramp": ["#1A2129", ACCENT],
            },
        }
    }


def register_altair_theme() -> None:
    alt.themes.register("editorial_dark", _editorial_dark)
    alt.themes.enable("editorial_dark")
