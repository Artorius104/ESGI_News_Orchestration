"""Extraction de features a partir d'un titre/description bruts.

Reprend la meme logique que scripts/prepare_data.py (utilisee pour construire
data/train_features.csv et data/test_features.csv), mais pour un seul article
au lieu d'un DataFrame entier : permet a l'utilisateur de coller du texte
plutot que de saisir directement les 10 valeurs numeriques.
"""

from __future__ import annotations

import re


def _avg_word_len(text: str) -> float:
    words = text.split()
    return sum(len(w) for w in words) / len(words) if words else 0.0


def extract_features(title: str, description: str) -> dict:
    title = title or ""
    description = description or ""

    return {
        "title_word_count": len(title.split()),
        "title_char_count": len(title),
        "desc_word_count": len(description.split()),
        "desc_char_count": len(description),
        "desc_avg_word_len": round(_avg_word_len(description), 4),
        "title_avg_word_len": round(_avg_word_len(title), 4),
        "has_reuters": int("reuters" in description.lower()),
        "has_ap": int(bool(re.search(r"\bAP\b", description))),
        "digit_ratio_desc": round(
            sum(c.isdigit() for c in description) / max(len(description), 1), 4
        ),
        "upper_ratio_title": round(sum(c.isupper() for c in title) / max(len(title), 1), 4),
    }
