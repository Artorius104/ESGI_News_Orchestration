from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from api_client import get_api_url, get_confusion_matrix, get_models
from theme import divider, kicker, note, subtitle

kicker("BANC D'ESSAI")
st.title("Comparatif des modeles")
subtitle(
    "Resultats de la recherche d'hyperparametres (GridSearchCV) sur RandomForest, "
    "XGBoost et LightGBM."
)
divider()

api_url = get_api_url()

try:
    models = get_models(api_url)
except Exception as exc:
    st.warning(f"Comparatif indisponible : {exc}")
    models = None

if models:
    df = pd.DataFrame(models).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    runner_up = df.iloc[1] if len(df) > 1 else None

    col1, col2, col3 = st.columns(3)
    col1.metric("Meilleur modele", best["name"])
    col2.metric("ROC AUC", f"{best['roc_auc']:.3f}")
    col3.metric("F1-score", f"{best['f1']:.3f}")

    if runner_up is not None:
        gap = best["roc_auc"] - runner_up["roc_auc"]
        note(
            f"🏆 <b>{best['name']}</b> devance <b>{runner_up['name']}</b> de "
            f"{gap:.2%} de ROC AUC ({best['roc_auc']:.4f} contre {runner_up['roc_auc']:.4f})."
        )

    with st.container(border=True):
        st.subheader("Scores par modele")
        df_long = df.melt(
            id_vars="name",
            value_vars=["roc_auc", "f1", "cv_score"],
            var_name="Metrique",
            value_name="Score",
        )
        chart = (
            alt.Chart(df_long)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("name:N", title=None),
                color=alt.Color("Metrique:N", legend=alt.Legend(orient="bottom", title=None)),
                yOffset="Metrique:N",
                tooltip=["name", "Metrique", alt.Tooltip("Score:Q", format=".3f")],
            )
            .properties(height=220)
        )
        st.altair_chart(chart, width="stretch")

    with st.expander("Meilleurs hyperparametres par modele"):
        for entry in models:
            st.markdown(f"**{entry['name']}**")
            st.json(entry["best_params"])

try:
    cm = get_confusion_matrix(api_url)
except Exception as exc:
    st.warning(f"Matrice de confusion indisponible : {exc}")
    cm = None

if cm:
    divider()
    st.subheader(f"Matrice de confusion — {cm['model_name']}")

    labels = cm["labels"]
    matrix = cm["matrix"]
    rows = [
        {"Reel": labels[i], "Predit": labels[j], "Count": matrix[i][j]}
        for i in range(len(labels))
        for j in range(len(labels))
    ]
    df_cm = pd.DataFrame(rows)
    max_count = df_cm["Count"].max()

    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(border=True):
            heatmap = (
                alt.Chart(df_cm)
                .mark_rect()
                .encode(
                    x=alt.X("Predit:N", sort=labels, title="Classe predite"),
                    y=alt.Y("Reel:N", sort=labels, title="Classe reelle"),
                    color=alt.Color("Count:Q", scale=alt.Scale(scheme="reds"), title="Articles"),
                    tooltip=["Reel", "Predit", "Count"],
                )
            )
            text = (
                alt.Chart(df_cm)
                .mark_text(baseline="middle", fontWeight="bold")
                .encode(
                    x=alt.X("Predit:N", sort=labels),
                    y=alt.Y("Reel:N", sort=labels),
                    text="Count:Q",
                    color=alt.condition(
                        alt.datum.Count > max_count / 2, alt.value("white"), alt.value("#E8EAED")
                    ),
                )
            )
            st.altair_chart((heatmap + text).properties(height=320), width="stretch")

    with right:
        with st.container(border=True):
            st.markdown("**Precision / Rappel / F1 par classe**")
            df_pc = (
                pd.DataFrame(cm["per_class"]).T.reset_index().rename(columns={"index": "Classe"})
            )
            df_pc_long = df_pc.melt(
                id_vars="Classe",
                value_vars=["precision", "recall", "f1_score"],
                var_name="Metrique",
                value_name="Score",
            )
            chart_pc = (
                alt.Chart(df_pc_long)
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("Classe:N", title=None),
                    color=alt.Color("Metrique:N", legend=alt.Legend(orient="bottom", title=None)),
                    yOffset="Metrique:N",
                    tooltip=["Classe", "Metrique", alt.Tooltip("Score:Q", format=".3f")],
                )
                .properties(height=320)
            )
            st.altair_chart(chart_pc, width="stretch")

    # Observation auto-generee : paire de classes la plus confondue (hors diagonale)
    off_diag = df_cm[df_cm["Reel"] != df_cm["Predit"]]
    worst_confusion = off_diag.loc[off_diag["Count"].idxmax()]

    per_class_f1 = {k: v["f1_score"] for k, v in cm["per_class"].items()}
    strongest = max(per_class_f1, key=lambda k: per_class_f1[k])
    weakest = min(per_class_f1, key=lambda k: per_class_f1[k])

    note(
        f"🔀 Confusion la plus frequente : des articles <b>{worst_confusion['Reel']}</b> "
        f"classes a tort en <b>{worst_confusion['Predit']}</b> "
        f"({int(worst_confusion['Count'])} cas)."
    )
    note(
        f"📈 Classe la plus fiable : <b>{strongest}</b> (F1={per_class_f1[strongest]:.2f}) — "
        f"la plus difficile : <b>{weakest}</b> (F1={per_class_f1[weakest]:.2f})."
    )
