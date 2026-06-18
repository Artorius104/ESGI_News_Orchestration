from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from mlproject.config import MODEL_DIR, MODEL_NAME, TRAIN_PATH
from mlproject.data import load_data, split
from mlproject.features import build_preprocessor
from mlproject.tracking import log_dataset, setup_experiment


def train(c: float = 1.0, max_iter: int = 1000) -> dict[str, float]:
    train_df, test_df = load_data()
    X_train, X_test, y_train, y_test = split(train_df, test_df)

    model = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("clf", LogisticRegression(C=c, max_iter=max_iter)),
        ]
    )

    setup_experiment()

    with mlflow.start_run(run_name=f"logreg-c{c}"):
        log_dataset(train_df, context="training", name="train", source=TRAIN_PATH)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        metrics = {
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "roc_auc": roc_auc_score(y_test, y_prob, multi_class="ovr"),
        }

        mlflow.log_params({"c": c, "max_iter": max_iter, "model": "logreg"})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name=MODEL_NAME)

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=["World", "Sports", "Business", "Sci/Tech"],
            ax=ax,
        )
        fig.tight_layout()
        fig.savefig("confusion.png")
        plt.close(fig)
        mlflow.log_artifact("confusion.png")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        print(f"f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
