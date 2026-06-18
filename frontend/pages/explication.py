from __future__ import annotations

import streamlit as st

from api_client import get_api_url, get_shap_summary_bytes

st.title("💡 Explicabilite (SHAP)")
st.write(
    "Ce graphique montre l'impact de chaque feature sur la prediction du modele retenu, "
    "calcule avec SHAP (SHapley Additive exPlanations) sur un echantillon du jeu de test. "
    "Chaque point represente un article ; sa couleur indique la valeur de la feature pour "
    "cet article (rose = elevee, bleu = faible), et sa position horizontale indique si cette "
    "valeur pousse la prediction vers la classe consideree ou non."
)

api_url = get_api_url()
try:
    image_bytes = get_shap_summary_bytes(api_url)
except Exception as exc:
    st.warning(f"Graphique SHAP indisponible : {exc}")
else:
    st.image(image_bytes, width="stretch")
