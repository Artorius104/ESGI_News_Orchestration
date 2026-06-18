"""Configuration partagée du suivi MLflow (S5-8, S5-9).

Centralise tracking URI + expérience pour éviter la duplication dans chaque
script, et ajoute la traçabilité des données (dataset lineage).
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import mlflow.data
import pandas as pd

from mlproject.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_EXPERIMENT_DESCRIPTION,
    MLFLOW_EXPERIMENT_TAGS,
    MLFLOW_TRACKING_URI,
    TARGET,
    TRAIN_PATH,
)

logger = logging.getLogger(__name__)


def setup_experiment() -> None:
    """Configurer le tracking MLflow et les métadonnées de l'expérience (S5-8).

    Idempotent : re-appelable sans erreur.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT)
    client = mlflow.MlflowClient()
    if MLFLOW_EXPERIMENT_DESCRIPTION:
        client.set_experiment_tag(
            experiment.experiment_id,
            "mlflow.note.content",
            MLFLOW_EXPERIMENT_DESCRIPTION,
        )
    for key, value in MLFLOW_EXPERIMENT_TAGS.items():
        client.set_experiment_tag(experiment.experiment_id, key, value)
    logger.info("MLflow experiment : %s (%s)", MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI)


def log_dataset(
    df: pd.DataFrame, context: str, name: str = "dataset", source: Path | str | None = None
) -> None:
    """Logger un dataset MLflow dans le run courant (S5-9).

    Parameters
    ----------
    df : pd.DataFrame
        Données à référencer (features + cible).
    context : str
        Rôle du dataset : "training" ou "evaluation".
    name : str
        Nom logique du dataset.
    source : Path or str, optional
        Chemin source du fichier CSV. Par défaut TRAIN_PATH.
    """
    src = str(source) if source is not None else str(TRAIN_PATH)
    dataset = mlflow.data.from_pandas(df, source=src, targets=TARGET, name=name)  # type: ignore[attr-defined]
    mlflow.log_input(dataset, context=context)
