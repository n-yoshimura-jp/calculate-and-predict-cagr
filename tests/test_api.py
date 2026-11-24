import math

import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor

from cagr_predictor import api


@pytest.fixture
def client(monkeypatch, sample_prices):
    """ネットワークと学習済みファイルを使わないテスト用クライアント。"""
    model = DummyRegressor(strategy="constant", constant=0.1).fit([[0]], [0.1])
    monkeypatch.setattr(api, "load_models", lambda: {"lightgbm": model})
    monkeypatch.setattr(api, "get_nasdaq100_symbols", lambda: ["AAPL", "MSFT"])
    monkeypatch.setattr(api, "download_price", lambda symbol, period: sample_prices)
    with TestClient(api.app) as client:
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models_loaded": ["lightgbm"]}


def test_symbols(client):
    assert client.get("/symbols").json() == {"count": 2, "symbols": ["AAPL", "MSFT"]}


def test_cagr(client):
    response = client.get("/cagr/AAPL")
    assert response.status_code == 200
    assert response.json()["symbol"] == "AAPL"


def test_cagr_last_years(client):
    body = client.get("/cagr/AAPL", params={"last_years": 1}).json()
    assert body["years"] == pytest.approx(1.0, abs=0.01)


def test_cagr_rejects_invalid_last_years(client):
    assert client.get("/cagr/AAPL", params={"last_years": 0}).status_code == 422


def test_annual_growth(client):
    response = client.get("/annual-growth/AAPL")
    assert response.status_code == 200
    rows = response.json()["annual_growth"]
    assert [row["year"] for row in rows] == [2020, 2021, 2022, 2023]
    assert rows[0]["year_on_year_return"] is None  # 最初の年は前年比が無い
    assert rows[1]["year_on_year_return"] is not None


def test_predict(client, sample_prices):
    response = client.get("/predict/aapl", params={"model": "lightgbm"})
    assert response.status_code == 200
    body = response.json()
    last_close = sample_prices["Close"].iloc[-1]
    assert body["symbol"] == "AAPL"
    assert body["model"] == "lightgbm"
    assert body["horizon_days"] == 252
    assert body["predicted_return"] == pytest.approx(0.1)
    assert body["predicted_growth"] == pytest.approx(math.exp(0.1) - 1)
    assert body["predicted_price"] == pytest.approx(last_close * math.exp(0.1))


def test_predict_rejects_invalid_symbol(client):
    assert client.get("/predict/INVALID!!").status_code == 422


def test_predict_rejects_symbol_outside_nasdaq100(client):
    assert client.get("/predict/TSLAX").status_code == 404


def test_predict_rejects_unknown_model(client):
    assert client.get("/predict/AAPL", params={"model": "xgboost"}).status_code == 422


def test_predict_without_trained_model(client):
    assert client.get("/predict/AAPL", params={"model": "ridge"}).status_code == 503
