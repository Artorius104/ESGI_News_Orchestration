"""Charte graphique partagee (SaaS moderne, degrade violet) : couleurs,
polices, helpers UI.

Palette et typographie alignees sur .streamlit/config.toml. inject() doit
etre appelee une fois dans app.py (le point d'entree multi-pages) pour
s'appliquer a toutes les pages.
"""

from __future__ import annotations

import altair as alt
import streamlit as st

BACKGROUND = "#07051A"
SURFACE = "#120D2B"
SURFACE_BORDER = "#272050"
TEXT_PRIMARY = "#F1EFFF"
TEXT_SECONDARY = "#948DBD"
ACCENT = "#7C3AED"
ACCENT_2 = "#312E81"
GRADIENT = "linear-gradient(135deg, #6D28D9 0%, #4338CA 55%, #1E1B4B 100%)"

# Une couleur dediee par classe (pas de degrade partage entre elles).
LABEL_COLORS = {
    "World": "#2563EB",
    "Sports": "#16A34A",
    "Business": "#D97706",
    "Sci/Tech": "#7C3AED",
}

# Une couleur dediee par metrique (distincte des couleurs de classe ci-dessus).
METRIC_COLORS = {
    "cv_score": "#38BDF8",
    "f1": "#34D399",
    "roc_auc": "#A78BFA",
    "precision": "#38BDF8",
    "recall": "#34D399",
    "f1_score": "#A78BFA",
}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, sans-serif;
}}

h1, h2, h3 {{
    font-family: 'Space Grotesk', system-ui, sans-serif !important;
    letter-spacing: -0.02em;
    font-weight: 700 !important;
}}

[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(ellipse 60rem 40rem at 15% -10%, rgba(109,40,217,0.25), transparent 60%),
        radial-gradient(ellipse 50rem 35rem at 110% 10%, rgba(67,56,202,0.20), transparent 55%),
        {BACKGROUND};
}}

.app-logo {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.5rem;
}}

.app-logo-mark {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.1rem;
    height: 2.1rem;
    border-radius: 0.6rem;
    background: {GRADIENT};
    color: white;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    box-shadow: 0 0 1.2rem rgba(124, 58, 237, 0.55);
}}

.app-logo-text {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.01em;
}}

.app-logo-text span {{
    background: {GRADIENT};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.app-credit {{
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.78rem;
    color: {TEXT_SECONDARY};
    margin: -0.3rem 0 0.6rem 0.1rem;
}}

.app-kicker {{
    display: inline-block;
    color: #C4B5FD;
    background: rgba(124, 58, 237, 0.12);
    border: 1px solid rgba(124, 58, 237, 0.35);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    margin-bottom: 0.6rem;
}}

.app-hero-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.8rem;
    line-height: 1.1;
    letter-spacing: -0.03em;
    color: {TEXT_PRIMARY};
    margin: 0.2rem 0 0.8rem 0;
}}

.app-hero-title .gradient {{
    background: {GRADIENT};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.app-subtitle {{
    color: {TEXT_SECONDARY};
    font-size: 1.05rem;
    max-width: 46rem;
    line-height: 1.6;
}}

.app-divider {{
    height: 1px;
    background: linear-gradient(90deg, {ACCENT}, {ACCENT_2}, transparent);
    border: none;
    margin: 1.25rem 0 1.75rem 0;
}}

.app-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.7rem 1.2rem;
    border-radius: 0.8rem;
    font-size: 1.4rem;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
    border: 1px solid;
}}

.app-note {{
    background: {SURFACE};
    border-left: 3px solid {ACCENT};
    border-radius: 0.6rem;
    padding: 0.85rem 1.1rem;
    color: {TEXT_PRIMARY};
    font-size: 0.95rem;
    margin-bottom: 0.6rem;
}}

.app-feature-card {{
    background: linear-gradient(160deg, rgba(124,58,237,0.10), rgba(23,16,41,0.4));
    border: 1px solid {SURFACE_BORDER};
    border-radius: 1rem;
    padding: 1.4rem 1.3rem;
    height: 100%;
}}

.app-feature-icon {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.6rem;
    height: 2.6rem;
    border-radius: 0.7rem;
    background: {GRADIENT};
    font-size: 1.3rem;
    margin-bottom: 0.8rem;
}}

.app-feature-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: {TEXT_PRIMARY};
    margin-bottom: 0.35rem;
}}

.app-feature-text {{
    color: {TEXT_SECONDARY};
    font-size: 0.9rem;
    line-height: 1.5;
}}

div[data-testid="stButton"] button,
div[data-testid="stLinkButton"] a,
div[data-testid="stFormSubmitButton"] button {{
    background: {GRADIENT} !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 0.4rem 1.4rem rgba(124, 58, 237, 0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}

div[data-testid="stButton"] button:hover,
div[data-testid="stLinkButton"] a:hover,
div[data-testid="stFormSubmitButton"] button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 0.6rem 1.8rem rgba(124, 58, 237, 0.5);
    color: white !important;
}}

[data-testid="stMetricValue"] {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
}}
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def logo(name: str = "AG News", tagline: str = "Intelligence") -> None:
    initials = "".join(w[0] for w in name.split()[:2]).upper()
    st.markdown(
        f'<div class="app-logo">'
        f'<span class="app-logo-mark">{initials}</span>'
        f'<span class="app-logo-text">{name} <span>{tagline}</span></span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def credit(name: str) -> None:
    st.markdown(f'<div class="app-credit">{name.upper()}</div>', unsafe_allow_html=True)


def hero_title(plain: str, gradient: str) -> None:
    st.markdown(
        f'<div class="app-hero-title">{plain} <span class="gradient">{gradient}</span></div>',
        unsafe_allow_html=True,
    )


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


def colored_button_css(slug: str, color: str) -> None:
    """Recolore un bouton precis (a placer dans un st.container(key=slug)).

    Le selecteur doit etre plus specifique que la regle globale des boutons
    (meme specificite + !important des deux cotes => sans ca, l'ordre
    d'injection des <style> par Streamlit n'est pas garanti et le fond
    global peut gagner alors que l'ombre scopee, elle, s'applique).
    """
    st.markdown(
        f"<style>"
        f".st-key-{slug} div[data-testid='stButton'] button {{"
        f"background: linear-gradient(135deg, {color}, {color}AA) !important;"
        f"box-shadow: 0 0.3rem 1rem {color}55 !important;"
        f"}}"
        f"</style>",
        unsafe_allow_html=True,
    )


def feature_card(icon: str, title: str, text: str) -> None:
    st.markdown(
        f'<div class="app-feature-card">'
        f'<div class="app-feature-icon">{icon}</div>'
        f'<div class="app-feature-title">{title}</div>'
        f'<div class="app-feature-text">{text}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _saas_gradient_dark() -> dict:
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
                "heatmap": [SURFACE, ACCENT],
                "ramp": [SURFACE, ACCENT],
            },
        }
    }


def register_altair_theme() -> None:
    alt.themes.register("saas_gradient_dark", _saas_gradient_dark)
    alt.themes.enable("saas_gradient_dark")
