from __future__ import annotations

import streamlit as st

from api_client import get_api_url, get_shap_summary_bytes
from theme import divider, kicker, subtitle

kicker("SOUS LE CAPOT")
st.title("Explicabilite (SHAP)")
subtitle(
    "Impact de chaque feature sur la prediction du modele retenu, calcule avec SHAP "
    "(SHapley Additive exPlanations) sur un echantillon du jeu de test."
)
divider()

api_url = get_api_url()
try:
    image_bytes = get_shap_summary_bytes(api_url)
except Exception as exc:
    st.warning(f"Graphique SHAP indisponible : {exc}")
else:
    with st.container(border=True):
        st.image(image_bytes, width="stretch")

    st.caption(
        "Chaque point represente un article. Sa couleur indique la valeur de la feature pour "
        "cet article (rose = elevee, bleu = faible) ; sa position horizontale indique si cette "
        "valeur pousse la prediction vers la classe consideree, ou l'en eloigne."
    )
