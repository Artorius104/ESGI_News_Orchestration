from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# TODO (S0-1)
DATA_PATH = ROOT / "data" / "dataset.csv"

# TODO (S0-2)
TARGET = "label"

# TODO (S0-3)
NUMERIC_FEATURES = [
    "title_word_count",
    "title_char_count",
    "desc_word_count",
    "desc_char_count",
    "desc_avg_word_len",
    "title_avg_word_len",
    "has_reuters",
    "has_ap",
    "digit_ratio_desc",
    "upper_ratio_title",
]

# TODO (S0-4)
CATEGORICAL_FEATURES = []

MLFLOW_EXPERIMENT = "ag-news-classification"
MODEL_NAME = "ag-news-classifier"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# Multiclasse 4 classes : World=0, Sports=1, Business=2, Sci/Tech=3
N_CLASSES = 4
SCORING = "roc_auc_ovr"
