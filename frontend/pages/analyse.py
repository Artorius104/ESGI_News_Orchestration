from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from api_client import check_health, get_api_url, predict
from examples import EXAMPLES
from features import extract_features
from theme import LABEL_COLORS, badge, colored_button_css, divider, kicker, note, subtitle

kicker("ATELIER DE CLASSIFICATION")
st.title("Analyse d'un article")
subtitle("Colle un titre et une description, ou pars d'un exemple, pour interroger le modele.")
divider()

api_url = st.text_input("URL de l'API", value=get_api_url(), key="api_url")

healthy, detail = check_health(api_url)
if healthy:
    st.success(f"API connectee — GET {api_url}/health → {{'status': '{detail}'}}")
else:
    st.error(f"API injoignable sur {api_url} : {detail}")

if "title_input" not in st.session_state:
    st.session_state.title_input = EXAMPLES["World"]["title"]
    st.session_state.desc_input = EXAMPLES["World"]["description"]

st.subheader("1. Choisis un exemple ou colle ton propre article")
example_cols = st.columns(4)
for col, label in zip(example_cols, EXAMPLES, strict=True):
    slug = "ex-" + label.lower().replace("/", "")
    with col:
        colored_button_css(slug, LABEL_COLORS[label])
        with st.container(key=slug):
            if st.button(f"Exemple {label}", width="stretch"):
                st.session_state.title_input = EXAMPLES[label]["title"]
                st.session_state.desc_input = EXAMPLES[label]["description"]

title = st.text_input("Titre", key="title_input")
description = st.text_area("Description", key="desc_input", height=120)

st.subheader("2. Prediction")

if st.button("🚀 Analyser et predire", type="primary", width="stretch"):
    if not healthy:
        st.error("Impossible de predire : l'API n'est pas joignable.")
    else:
        payload = extract_features(title, description)
        try:
            result = predict(api_url, payload)
        except Exception as exc:
            st.error(f"Echec de la prediction : {exc}")
        else:
            label = result["label"]
            color = LABEL_COLORS.get(label, "#888888")

            st.write("")
            badge(f"📰 {label}", color)
            st.write("")

            col1, col2 = st.columns(2)
            col1.metric("Classe predite", label)
            col2.metric("Probabilite", f"{result['probability']:.1%}")

            probs = sorted(
                zip(EXAMPLES.keys(), result["probabilities"], strict=True),
                key=lambda kv: kv[1],
                reverse=True,
            )
            top_label, top_prob = probs[0]
            runner_label, runner_prob = probs[1]
            gap = top_prob - runner_prob
            if gap > 0.3:
                note(
                    f"✅ Le modele est **tres confiant** : {top_label} devance {runner_label} "
                    f"de {gap:.0%} de probabilite."
                )
            elif gap > 0.1:
                note(
                    f"🔹 Confiance moderee : {top_label} l'emporte sur {runner_label}, "
                    f"mais l'ecart reste resserre ({gap:.0%})."
                )
            else:
                note(
                    f"⚠️ Le modele **hesite** entre {top_label} et {runner_label} "
                    f"(ecart de seulement {gap:.0%})."
                )

            with st.container(border=True):
                st.markdown("**Probabilites par classe**")
                df_probs = pd.DataFrame(
                    {
                        "Classe": list(EXAMPLES.keys()),
                        "Probabilite": result["probabilities"],
                    }
                )
                chart = (
                    alt.Chart(df_probs)
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X("Probabilite:Q", title="Probabilite", axis=alt.Axis(format="%")),
                        y=alt.Y("Classe:N", sort="-x", title=None),
                        color=alt.Color(
                            "Classe:N",
                            scale=alt.Scale(
                                domain=list(LABEL_COLORS.keys()),
                                range=list(LABEL_COLORS.values()),
                            ),
                            legend=None,
                        ),
                        tooltip=[
                            "Classe",
                            alt.Tooltip("Probabilite:Q", format=".1%"),
                        ],
                    )
                    .properties(height=200)
                )
                st.altair_chart(chart, width="stretch")

            with st.expander("🛠️ Reponse brute de l'API — POST /predict"):
                st.json(result)
            with st.expander("🧮 Features envoyees a l'API (calculees a partir du texte)"):
                st.json(payload)
