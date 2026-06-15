from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlproject.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    transformers: list = [("num", StandardScaler(), NUMERIC_FEATURES)]
    if CATEGORICAL_FEATURES:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES))
    return ColumnTransformer(transformers=transformers)
