"""Client HTTP partage vers l'API FastAPI (src/mlproject/api.py).

Toutes les pages du frontend passent par ces fonctions plutot que d'appeler
httpx directement : ca centralise l'URL de l'API et la gestion d'erreurs.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

DEFAULT_API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")


def get_api_url() -> str:
    return st.session_state.get("api_url", DEFAULT_API_URL)


def check_health(api_url: str) -> tuple[bool, str]:
    try:
        response = httpx.get(f"{api_url}/health", timeout=5.0)
        response.raise_for_status()
        return True, response.json().get("status", "ok")
    except httpx.HTTPError as exc:
        return False, str(exc)


def predict(api_url: str, payload: dict) -> dict:
    response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
    response.raise_for_status()
    return response.json()


def get_models(api_url: str) -> list[dict]:
    response = httpx.get(f"{api_url}/models", timeout=10.0)
    response.raise_for_status()
    return response.json()


def get_dataset_info(api_url: str) -> dict:
    response = httpx.get(f"{api_url}/dataset-info", timeout=10.0)
    response.raise_for_status()
    return response.json()


def get_shap_summary_bytes(api_url: str) -> bytes:
    response = httpx.get(f"{api_url}/shap-summary", timeout=10.0)
    response.raise_for_status()
    return response.content
