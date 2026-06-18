from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import get_api_url, get_models

st.title("📊 Comparatif des modeles")
st.caption(
    "Resultats de la recherche d'hyperparametres (GridSearchCV) sur RandomForest, "
    "XGBoost et LightGBM (src/mlproject/train_models.py)."
)

api_url = get_api_url()
try:
    models = get_models(api_url)
except Exception as exc:
    st.warning(f"Comparatif indisponible : {exc}")
else:
    df = pd.DataFrame(models).sort_values("roc_auc", ascending=False)
    best = df.iloc[0]
    st.success(
        f"Meilleur modele retenu pour la production : **{best['name']}** "
        f"(roc_auc={best['roc_auc']:.3f})"
    )

    st.subheader("Scores")
    st.bar_chart(df.set_index("name")[["roc_auc", "f1", "cv_score"]])

    st.subheader("Detail")
    st.dataframe(
        df[["name", "roc_auc", "f1", "cv_score", "is_best"]].rename(
            columns={
                "name": "Modele",
                "roc_auc": "ROC AUC",
                "f1": "F1",
                "cv_score": "CV score",
                "is_best": "Retenu",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    with st.expander("Meilleurs hyperparametres par modele"):
        for entry in models:
            st.markdown(f"**{entry['name']}**")
            st.json(entry["best_params"])
