from mlproject.config import (
    CATEGORICAL_FEATURES,
    EVAL_F1_MIN,
    EVAL_ROC_AUC_MIN,
    MLFLOW_EXPERIMENT_TAGS,
    MODEL_DIR,
    N_CLASSES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    SCORING,
    TARGET,
    TRAIN_PATH,
    TEST_PATH,
    _parse_tags,
)


def test_target():
    assert TARGET == "label"


def test_numeric_features():
    assert len(NUMERIC_FEATURES) == 10
    assert all(isinstance(f, str) for f in NUMERIC_FEATURES)


def test_categorical_features_empty():
    assert CATEGORICAL_FEATURES == []


def test_multiclass_constants():
    assert N_CLASSES == 4
    assert SCORING == "roc_auc_ovr"
    assert RANDOM_STATE == 42


def test_paths_are_absolute():
    assert TRAIN_PATH.is_absolute()
    assert TEST_PATH.is_absolute()
    assert MODEL_DIR.is_absolute()


def test_eval_thresholds_positive():
    assert 0 < EVAL_ROC_AUC_MIN <= 1
    assert 0 < EVAL_F1_MIN <= 1


def test_parse_tags_valid():
    result = _parse_tags("course=mlops,dataset=ag-news")
    assert result == {"course": "mlops", "dataset": "ag-news"}


def test_parse_tags_empty():
    assert _parse_tags("") == {}


def test_experiment_tags_dict():
    assert isinstance(MLFLOW_EXPERIMENT_TAGS, dict)
