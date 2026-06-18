import pandas as pd

from mlproject.config import NUMERIC_FEATURES, TARGET
from mlproject.data import split


def _make_dfs(n_train: int = 80, n_test: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = {f: range(n_train) for f in NUMERIC_FEATURES}
    cols[TARGET] = [i % 4 for i in range(n_train)]
    train_df = pd.DataFrame(cols)

    cols_t = {f: range(n_test) for f in NUMERIC_FEATURES}
    cols_t[TARGET] = [i % 4 for i in range(n_test)]
    test_df = pd.DataFrame(cols_t)
    return train_df, test_df


def test_split_shapes():
    train_df, test_df = _make_dfs()
    X_train, X_test, y_train, y_test = split(train_df, test_df)
    assert X_train.shape == (80, len(NUMERIC_FEATURES))
    assert X_test.shape == (20, len(NUMERIC_FEATURES))
    assert y_train.shape == (80,)
    assert y_test.shape == (20,)


def test_split_no_target_in_features():
    train_df, test_df = _make_dfs()
    X_train, X_test, _, _ = split(train_df, test_df)
    assert TARGET not in X_train.columns
    assert TARGET not in X_test.columns


def test_split_target_values():
    train_df, test_df = _make_dfs()
    _, _, y_train, y_test = split(train_df, test_df)
    assert set(y_train.unique()).issubset({0, 1, 2, 3})
    assert set(y_test.unique()).issubset({0, 1, 2, 3})
