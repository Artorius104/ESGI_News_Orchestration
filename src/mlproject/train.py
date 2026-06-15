import argparse
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from mlproject.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    SCORING,
)
from mlproject.data import load_split
from mlproject.features import build_preprocessor


def train(c: float = 1.0, max_iter: int = 1000) -> dict[str, float]:
    X_train, X_test, y_train, y_test = load_split()

    model = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", LogisticRegression(C=c, max_iter=max_iter)),
    ])

    # TODO (S5-1, S5-2)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # TODO (S5-3)
    with mlflow.start_run(run_name=f"logreg-c{c}"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        metrics = {
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "roc_auc": roc_auc_score(y_test, y_prob, multi_class="ovr"),
        }

        # TODO (S5-4, S5-5)
        mlflow.log_params({"c": c, "max_iter": max_iter, "model": "logreg"})
        mlflow.log_metrics(metrics)

        # TODO (S5-6)
        mlflow.sklearn.log_model(model, name=MODEL_NAME)

        # TODO (S5-7) — matrice de confusion
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=["World", "Sports", "Business", "Sci/Tech"],
            ax=ax,
        )
        fig.tight_layout()
        fig.savefig("confusion.png")
        plt.close(fig)
        mlflow.log_artifact("confusion.png")

        print(f"f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter)
