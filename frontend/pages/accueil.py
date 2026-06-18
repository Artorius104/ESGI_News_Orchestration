from __future__ import annotations

import streamlit as st

from api_client import get_api_url, get_dataset_info

st.title("📰 Classification d'articles de presse — AG News")
st.caption("Projet fil rouge - Orchestration Machine Learning (ESGI)")

st.subheader("Problematique")
st.write(
    "Les redactions et agregateurs d'actualites recoivent des milliers d'articles par jour. "
    "Ce projet entraine et expose un modele qui classe automatiquement chaque article dans "
    "l'une de 4 categories a partir de son titre et de sa description, sans intervention "
    "humaine."
)

st.subheader("Objectif")
st.write(
    "Construire un modele de classification **multi-classe** (4 classes), l'exposer via une "
    "API FastAPI (`/predict`), et fournir ce frontend pour le tester et l'explorer "
    "interactivement (page **Analyse**), comparer les familles de modeles entraines "
    "(page **Modeles**) et comprendre les variables qui pesent le plus sur la prediction "
    "(page **Explication**)."
)

st.subheader("Le jeu de donnees — AG News")

api_url = get_api_url()
try:
    info = get_dataset_info(api_url)
except Exception as exc:
    st.warning(f"Impossible de recuperer les statistiques du dataset depuis l'API : {exc}")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Articles (train)", f"{info['n_train']:,}".replace(",", " "))
    col2.metric("Articles (test)", f"{info['n_test']:,}".replace(",", " "))
    col3.metric("Features", len(info["features"]))

    st.markdown("**Repartition des classes (train)**")
    st.bar_chart(info["class_distribution"])

    with st.expander("Voir les 10 features utilisees par le modele"):
        st.write(info["features"])
