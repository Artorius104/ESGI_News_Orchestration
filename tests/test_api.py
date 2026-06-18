import pytest
from fastapi.testclient import TestClient

from mlproject.api import app

VALID_PAYLOAD = {
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


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "version" in response.json()


def test_predict_valid_payload(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in range(4)
    assert body["label"] in ["World", "Sports", "Business", "Sci/Tech"]
    assert 0.0 <= body["probability"] <= 1.0
    assert len(body["probabilities"]) == 4
    assert pytest.approx(sum(body["probabilities"]), abs=1e-2) == 1.0


def test_predict_missing_field(client):
    payload = dict(VALID_PAYLOAD)
    del payload["title_word_count"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_ratio(client):
    payload = dict(VALID_PAYLOAD)
    payload["digit_ratio_desc"] = 1.5
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_count(client):
    payload = dict(VALID_PAYLOAD)
    payload["title_word_count"] = -1
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
