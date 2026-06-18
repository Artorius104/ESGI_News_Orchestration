from mlflow.models import MetricThreshold

from mlproject.config import EVAL_F1_MIN, EVAL_ROC_AUC_MIN
from mlproject.evaluate import build_thresholds


def test_build_thresholds_keys():
    thresholds = build_thresholds()
    assert "roc_auc" in thresholds
    assert "f1_score" in thresholds


def test_build_thresholds_types():
    thresholds = build_thresholds()
    assert all(isinstance(v, MetricThreshold) for v in thresholds.values())


def test_build_thresholds_values():
    thresholds = build_thresholds()
    assert thresholds["roc_auc"].threshold == EVAL_ROC_AUC_MIN
    assert thresholds["f1_score"].threshold == EVAL_F1_MIN
    assert thresholds["roc_auc"].greater_is_better is True
    assert thresholds["f1_score"].greater_is_better is True
