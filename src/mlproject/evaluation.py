from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import shap
from matplotlib.figure import Figure
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _build_shap_figure(pipeline: Pipeline, x_test, name: str, max_samples: int) -> Figure | None:
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["clf"]

    transformed = preprocessor.transform(x_test)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()
    sample = transformed[:max_samples]

    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(sample)
    except Exception:
        logger.warning("SHAP TreeExplainer indisponible pour %s, artefact ignore", name)
        return None

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
    fig = plt.gcf()
    fig.suptitle(f"Importance des variables (SHAP) : {name}")
    return fig


def log_shap_summary(pipeline: Pipeline, x_test, name: str, max_samples: int = 200) -> None:
    fig = _build_shap_figure(pipeline, x_test, name, max_samples)
    if fig is None:
        return
    mlflow.log_figure(fig, "shap_summary.png")
    plt.close(fig)


def save_shap_summary(
    pipeline: Pipeline, x_test, name: str, output_path: Path, max_samples: int = 200
) -> None:
    """Sauvegarde le graphique SHAP sur disque, sans dependance a un run MLflow actif."""
    fig = _build_shap_figure(pipeline, x_test, name, max_samples)
    if fig is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
