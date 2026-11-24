"""CAGR・年次成長率の計算と、1年後の株価予測の API。

起動: uv run fastapi dev src/cagr_predictor/api.py
"""

import logging
from contextlib import asynccontextmanager
from datetime import date
from enum import StrEnum
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel

from cagr_predictor.cagr import annual_growth, calc_cagr
from cagr_predictor.data import download_price, get_nasdaq100_symbols
from cagr_predictor.predict import load_metrics, load_models, predict_price

logger = logging.getLogger(__name__)


class ModelName(StrEnum):
    baseline = "baseline"
    ridge = "ridge"
    random_forest = "random_forest"
    lightgbm = "lightgbm"


SymbolParam = Annotated[
    str,
    Path(pattern=r"^[A-Za-z.\-]{1,10}$", description="ティッカー（例: AAPL）", examples=["AAPL"]),
]

ml_models = {}
symbols: set[str] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 起動時にモデルと銘柄リストを1回だけ読み込む
    ml_models.update(load_models())
    symbols.update(get_nasdaq100_symbols())
    yield
    ml_models.clear()
    symbols.clear()


app = FastAPI(
    title="NASDAQ-100 CAGR & Price Prediction API",
    description="教育・研究目的のデモです。投資助言ではありません。",
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]


class SymbolsResponse(BaseModel):
    count: int
    symbols: list[str]


class CagrResponse(BaseModel):
    symbol: str
    start_date: date
    end_date: date
    years: float
    start_price: float
    end_price: float
    cagr: float


class AnnualGrowth(BaseModel):
    year: int
    first_close: float
    last_close: float
    within_year_return: float
    year_on_year_return: float | None


class AnnualGrowthResponse(BaseModel):
    symbol: str
    annual_growth: list[AnnualGrowth]


class PredictionResponse(BaseModel):
    symbol: str
    model: str
    as_of: date
    last_close: float
    horizon_days: int
    predicted_return: float
    predicted_growth: float
    predicted_price: float


class ModelMetrics(BaseModel):
    mae: float
    rmse: float
    direction_accuracy: float


class MetricsResponse(BaseModel):
    horizon_days: int
    train_period: list[str]
    test_period: list[str]
    models: dict[str, ModelMetrics]


def fetch_prices(symbol: str, period: str) -> tuple[str, pd.DataFrame]:
    """NASDAQ-100 の銘柄だけ株価を取得する。取得できなければ HTTP エラーにする。"""
    symbol = symbol.upper()
    if symbol not in symbols:
        raise HTTPException(status_code=404, detail=f"{symbol} は NASDAQ-100 の銘柄ではありません")
    try:
        prices = download_price(symbol, period=period)
    except Exception:
        logger.exception("株価の取得に失敗しました: %s", symbol)
        raise HTTPException(status_code=502, detail="株価データの取得に失敗しました")
    if prices.empty:
        raise HTTPException(status_code=404, detail=f"{symbol} の株価データがありません")
    return symbol, prices


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok", models_loaded=sorted(ml_models))


@app.get("/symbols")
def list_symbols() -> SymbolsResponse:
    return SymbolsResponse(count=len(symbols), symbols=sorted(symbols))


@app.get("/cagr/{symbol}")
def get_cagr(
    symbol: SymbolParam,
    last_years: Annotated[
        int | None, Query(ge=1, le=50, description="直近 N 年で計算する（省略時は全期間）")
    ] = None,
) -> CagrResponse:
    symbol, prices = fetch_prices(symbol, period="max")
    try:
        result = calc_cagr(prices["Close"], last_years=last_years)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return CagrResponse(symbol=symbol, **result)


@app.get("/annual-growth/{symbol}")
def get_annual_growth(symbol: SymbolParam) -> AnnualGrowthResponse:
    symbol, prices = fetch_prices(symbol, period="max")
    table = annual_growth(prices["Close"]).reset_index()
    # 最初の年の前年比は NaN なので、JSON では null にする
    rows = table.astype(object).where(table.notna(), None).to_dict(orient="records")
    return AnnualGrowthResponse(symbol=symbol, annual_growth=rows)


@app.get("/predict/{symbol}")
def predict(
    symbol: SymbolParam,
    model: Annotated[ModelName, Query(description="使用するモデル")] = ModelName.lightgbm,
) -> PredictionResponse:
    if model not in ml_models:
        raise HTTPException(
            status_code=503, detail="モデルが未学習です。`uv run cagr-train` を実行してください"
        )
    # 特徴量に直近1年分（253営業日）の株価が必要なので2年分を取得する
    symbol, prices = fetch_prices(symbol, period="2y")
    try:
        result = predict_price(prices, ml_models[model])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return PredictionResponse(symbol=symbol, model=model.value, **result)


@app.get("/models/metrics")
def get_metrics() -> MetricsResponse:
    metrics = load_metrics()
    if metrics is None:
        raise HTTPException(
            status_code=503, detail="評価結果がありません。`uv run cagr-train` を実行してください"
        )
    return MetricsResponse(**metrics)
