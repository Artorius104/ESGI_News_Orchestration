"""Frontend Streamlit : demonstrateur de classification AG News.

Point d'entree multi-pages (Accueil / Analyse / Modeles / Explication), qui
appelle l'API FastAPI (src/mlproject/api.py).

Lancement local : streamlit run frontend/app.py
L'URL de l'API par defaut est lue depuis la variable d'environnement API_URL
(utile en docker compose, ou l'API est joignable via le nom de service `api`).
"""

from __future__ import annotations

import os

import streamlit as st

from theme import credit, inject, logo, register_altair_theme

st.set_page_config(page_title="AG News Intelligence", layout="wide", page_icon="📰")
inject()
register_altair_theme()

MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://127.0.0.1:5000")
AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://127.0.0.1:8080")

with st.sidebar:
    logo()
    credit("Divin Badiabo")
    st.markdown("---")
    st.markdown("**Liens utiles**")
    st.link_button("📈 Ouvrir MLflow", MLFLOW_URL, width="stretch")
    st.link_button("🗓️ Ouvrir Airflow", AIRFLOW_URL, width="stretch")

pages = [
    st.Page("pages/accueil.py", title="Accueil", icon="🏠", default=True),
    st.Page("pages/analyse.py", title="Analyse", icon="🔍"),
    st.Page("pages/modeles.py", title="Modeles", icon="📊"),
    st.Page("pages/explication.py", title="Explication", icon="💡"),
]

navigation = st.navigation(pages, position="top")
navigation.run()
