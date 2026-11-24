"""学習済みモデルを読み込み、約1年後（HORIZON 営業日後）の価格を予測する。

注意: joblib（pickle）は読み込み時に任意のコードを実行できるため、
自分で `uv run cagr-train` やノートブックで作ったファイルだけを読み込むこと。
"""

import json
import math

import joblib
import pandas as pd

from cagr_predictor.features import FEATURE_COLUMNS, HORIZON, MIN_HISTORY, make_features
from cagr_predictor.train import MODELS_DIR, create_models

MODEL_NAMES = list(create_models())


def load_models() -> dict:
    """models/ にある学習済みモデルを読み込む。無いモデルは含めない。"""
    models = {}
    for name in MODEL_NAMES:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
    return models


def load_metrics() -> dict | None:
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def predict_price(prices: pd.DataFrame, model, horizon: int = HORIZON) -> dict:
    """直近の株価（Close, Volume）から horizon 営業日後の価格を予測する。

    Returns:
        predicted_return: 対数リターンの予測値
        predicted_growth: 予測成長率（小数。0.1 = +10%）= exp(predicted_return) - 1
        predicted_price: 予測価格 = last_close × exp(predicted_return)
    """
    if len(prices) < MIN_HISTORY:
        raise ValueError(f"予測には{MIN_HISTORY}営業日以上の株価が必要です")

    latest = make_features(prices)[FEATURE_COLUMNS].iloc[[-1]]
    if latest.isna().any().any():
        raise ValueError("直近の特徴量に欠損があるため予測できません")

    predicted_return = float(model.predict(latest)[0])
    last_close = float(prices["Close"].iloc[-1])
    return {
        "as_of": prices.index[-1].date(),
        "last_close": last_close,
        "horizon_days": horizon,
        "predicted_return": predicted_return,
        "predicted_growth": math.exp(predicted_return) - 1,
        "predicted_price": last_close * math.exp(predicted_return),
    }
