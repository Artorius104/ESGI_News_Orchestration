"""API d'inference pour la classification AG News (FastAPI) — S12.

Lancement :
    uvicorn mlproject.api:app --reload
    make api
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlproject.config import MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_path = MODEL_DIR / "model.joblib"
    if not model_path.exists():
        raise RuntimeError(f"Modele introuvable : {model_path}. Lancez d'abord make train.")
    ml["model"] = joblib.load(model_path)
    logger.info("Modele charge depuis %s", model_path)
    yield
    ml.clear()


app = FastAPI(title="AG News Classification API", version="0.1.0", lifespan=lifespan)


class Features(BaseModel):
    title_word_count: int = Field(..., ge=0)
    title_char_count: int = Field(..., ge=0)
    desc_word_count: int = Field(..., ge=0)
    desc_char_count: int = Field(..., ge=0)
    desc_avg_word_len: float = Field(..., ge=0.0)
    title_avg_word_len: float = Field(..., ge=0.0)
    has_reuters: int = Field(..., ge=0, le=1)
    has_ap: int = Field(..., ge=0, le=1)
    digit_ratio_desc: float = Field(..., ge=0.0, le=1.0)
    upper_ratio_title: float = Field(..., ge=0.0, le=1.0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title_word_count": 8,
                    "title_char_count": 45,
                    "desc_word_count": 32,
                    "desc_char_count": 210,
                    "desc_avg_word_len": 5.2,
                    "title_avg_word_len": 4.8,
                    "has_reuters": 1,
                    "has_ap": 0,
                    "digit_ratio_desc": 0.03,
                    "upper_ratio_title": 0.12,
                }
            ]
        }
    }


class PredictionOut(BaseModel):
    prediction: int
    label: str
    probability: float
    probabilities: list[float]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    row = pd.DataFrame([features.model_dump()])
    proba_array = model.predict_proba(row)[0]
    prediction = int(proba_array.argmax())
    return PredictionOut(
        prediction=prediction,
        label=LABEL_NAMES[prediction],
        probability=round(float(proba_array.max()), 4),
        probabilities=[round(float(p), 4) for p in proba_array],
    )


@app.get("/model-info")
def model_info() -> dict:
    return {"version": os.environ.get("MODEL_VERSION", "unknown")}
