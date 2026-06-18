from __future__ import annotations

import streamlit as st

from api_client import check_health, get_api_url, predict
from examples import EXAMPLES
from features import extract_features

LABEL_COLORS = {
    "World": "#1f77b4",
    "Sports": "#2ca02c",
    "Business": "#9467bd",
    "Sci/Tech": "#ff7f0e",
}

st.title("🔍 Analyse d'un article")

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
    if col.button(f"Exemple {label}", width="stretch"):
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
            st.markdown(
                f"<div style='background-color:{color}22;border-left:6px solid {color};"
                f"padding:1rem;border-radius:0.5rem;'>"
                f"<span style='font-size:1.5rem;font-weight:bold;color:{color};'>"
                f"📰 Classe predite : {label}</span></div>",
                unsafe_allow_html=True,
            )

            st.write("")
            col1, col2 = st.columns(2)
            col1.metric("Classe predite", label)
            col2.metric("Probabilite", f"{result['probability']:.1%}")

            st.markdown("**Confiance du modele**")
            st.progress(result["probability"])

            st.markdown("**Probabilites par classe**")
            st.bar_chart(dict(zip(EXAMPLES.keys(), result["probabilities"], strict=True)))

            with st.expander("🛠️ Reponse brute de l'API — POST /predict"):
                st.json(result)
            with st.expander("🧮 Features envoyees a l'API (calculees a partir du texte)"):
                st.json(payload)
