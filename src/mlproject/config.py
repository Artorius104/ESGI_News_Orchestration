from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

TRAIN_PATH = ROOT / "data" / "train_features.csv"
TEST_PATH  = ROOT / "data" / "test_features.csv"
MODEL_DIR  = ROOT / "models"

TARGET = "label"

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

CATEGORICAL_FEATURES: list[str] = []

RANDOM_STATE = 42

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "ag-news-classification")
MODEL_NAME = os.getenv("MODEL_NAME", "ag-news-classifier")

N_CLASSES = 4
SCORING = "roc_auc_ovr"
