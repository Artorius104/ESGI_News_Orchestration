"""Prepare AG News → dataset.csv avec features numériques et labels 0-3.

Classes AG News : 1=World, 2=Sports, 3=Business, 4=Sci/Tech
Remapping      : {1→0, 2→1, 3→2, 4→3}
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT              = Path(__file__).resolve().parents[1]
TRAIN_PATH        = ROOT / "data" / "train.csv"
TEST_PATH         = ROOT / "data" / "test.csv"
TRAIN_FEATURES    = ROOT / "data" / "train_features.csv"
TEST_FEATURES     = ROOT / "data" / "test_features.csv"

LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3}


def _avg_word_len(series: pd.Series) -> pd.Series:
    def _awl(text: str) -> float:
        words = text.split()
        return sum(len(w) for w in words) / len(words) if words else 0.0
    return series.map(_awl)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    title = df["Title"].fillna("")
    desc  = df["Description"].fillna("")

    return pd.DataFrame({
        "label":              df["Class Index"].map(LABEL_MAP),
        "title_word_count":   title.str.split().str.len(),
        "title_char_count":   title.str.len(),
        "desc_word_count":    desc.str.split().str.len(),
        "desc_char_count":    desc.str.len(),
        "desc_avg_word_len":  _avg_word_len(desc),
        "title_avg_word_len": _avg_word_len(title),
        "has_reuters":        desc.str.contains("Reuters", case=False).astype(int),
        "has_ap":             desc.str.contains(r"\bAP\b").astype(int),
        "digit_ratio_desc":   desc.apply(lambda t: sum(c.isdigit() for c in t) / max(len(t), 1)),
        "upper_ratio_title":  title.apply(lambda t: sum(c.isupper() for c in t) / max(len(t), 1)),
    })


def main() -> None:
    log.info("Chargement des données brutes...")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw  = pd.read_csv(TEST_PATH)
    log.info("train : %d lignes | test : %d lignes", len(train_raw), len(test_raw))

    train_out = extract_features(train_raw)
    test_out  = extract_features(test_raw)

    log.info("Distribution des labels (train) :\n%s", train_out["label"].value_counts().sort_index().to_string())

    TRAIN_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(TRAIN_FEATURES, index=False)
    test_out.to_csv(TEST_FEATURES, index=False)
    log.info("Sauvegardé → %s  (%d lignes)", TRAIN_FEATURES, len(train_out))
    log.info("Sauvegardé → %s  (%d lignes)", TEST_FEATURES, len(test_out))


if __name__ == "__main__":
    main()
