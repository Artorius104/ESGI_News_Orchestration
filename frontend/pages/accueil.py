from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from api_client import get_api_url, get_dataset_info
from theme import LABEL_COLORS, divider, kicker, note, subtitle

kicker("PROJET FIL ROUGE · ESGI · ORCHESTRATION ML")
st.title("Classification d'articles de presse — AG News")
subtitle(
    "Un modele de machine learning classe automatiquement chaque article dans l'une de "
    "4 rubriques — <b>World</b>, <b>Sports</b>, <b>Business</b>, <b>Sci/Tech</b> — a partir "
    "de son titre et de sa description, sans intervention humaine."
)
divider()

api_url = get_api_url()
try:
    info = get_dataset_info(api_url)
except Exception as exc:
    st.warning(f"Impossible de recuperer les statistiques du dataset depuis l'API : {exc}")
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Articles d'entrainement", f"{info['n_train']:,}".replace(",", " "))
    col2.metric("Articles de test", f"{info['n_test']:,}".replace(",", " "))
    col3.metric("Categories", len(info["class_distribution"]))
    col4.metric("Features par article", len(info["features"]))

    st.write("")
    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("Repartition des classes")
            dist = info["class_distribution"]
            df_dist = pd.DataFrame({"Classe": list(dist.keys()), "Articles": list(dist.values())})
            chart = (
                alt.Chart(df_dist)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Articles:Q", title="Nombre d'articles"),
                    y=alt.Y("Classe:N", sort="-x", title=None),
                    color=alt.Color(
                        "Classe:N",
                        scale=alt.Scale(
                            domain=list(LABEL_COLORS.keys()), range=list(LABEL_COLORS.values())
                        ),
                        legend=None,
                    ),
                    tooltip=["Classe", "Articles"],
                )
                .properties(height=220)
            )
            st.altair_chart(chart, width="stretch")

            counts = list(dist.values())
            balanced = max(counts) - min(counts) <= 0.02 * (sum(counts) / len(counts))
            if balanced:
                note(
                    "📐 Dataset parfaitement equilibre : chaque classe represente environ "
                    f"{100 / len(counts):.0f}% du jeu d'entrainement — pas de biais de classe "
                    "majoritaire a corriger."
                )

    with right:
        with st.container(border=True):
            st.subheader("Problematique")
            st.write(
                "Les redactions et agregateurs d'actualites recoivent des milliers "
                "d'articles par jour. Trier manuellement chaque depeche est couteux et "
                "ne passe pas a l'echelle."
            )
        with st.container(border=True):
            st.subheader("Objectif")
            st.write(
                "Entrainer un modele de classification **multi-classe**, l'exposer via "
                "une API FastAPI, et fournir ce tableau de bord pour le tester "
                "(**Analyse**), comparer les modeles entraines (**Modeles**) et comprendre "
                "ses decisions (**Explication**)."
            )

    with st.expander("Voir les 10 features utilisees par le modele"):
        st.write(info["features"])
