import pandas as pd
from sklearn.model_selection import train_test_split

from mlproject.config import CATEGORICAL_FEATURES, DATA_PATH, NUMERIC_FEATURES, TARGET


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


def load_split(
    test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = load_data()
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
