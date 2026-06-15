"""Configuration centrale du projet de classification.

C'est le SEUL fichier a adapter pour brancher votre propre jeu de donnees :
data.py, features.py et les scripts d'entrainement lisent toutes leurs
colonnes via ces constantes. Voir tp/TP_S0_projet_personnel.md.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# (S0-1) dataset généré par scripts/prepare_data.py
DATA_PATH = ROOT / "data" / "dataset.csv"
MODEL_DIR = ROOT / "models"

# (S0-2) cible multiclasse : 0=World, 1=Sports, 2=Business, 3=Sci/Tech
TARGET = "label"

# (S0-3) features numériques extraites du texte (titre + description)
NUMERIC_FEATURES: list[str] = [
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

# (S0-4) pas de colonnes catégorielles dans ce dataset
CATEGORICAL_FEATURES: list[str] = []

RANDOM_STATE = 42

# Surcouche via variables d'environnement (principe 12-factor)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "ag-news-multiclass")
MODEL_NAME = os.getenv("MODEL_NAME", "ag-news-classifier")
