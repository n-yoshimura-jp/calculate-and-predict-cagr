"""モデルを学習・評価して models/ に保存する。

実行: uv run cagr-train
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cagr_predictor.data import download_prices, get_nasdaq100_symbols
from cagr_predictor.features import FEATURE_COLUMNS, HORIZON, build_dataset

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
TEST_RATIO = 0.2


def create_models() -> dict:
    """比較するモデル。baseline は「学習期間の平均リターン」を常に予測する。"""
    return {
        "baseline": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=200, n_jobs=-1, random_state=42
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=200,
            importance_type="gain",
            random_state=42,
            verbose=-1,
        ),
    }


def split_by_date(data: pd.DataFrame, horizon: int = HORIZON, test_ratio: float = TEST_RATIO):
    """日付で train / test に分ける。

    目的変数は horizon 日先まで見ているので、train の最後の horizon 日分を捨てて
    test 期間の価格が train の答えに入らないようにする。
    """
    dates = data.index.unique().sort_values()
    split_index = int(len(dates) * (1 - test_ratio))
    if split_index - horizon - 1 < 0:
        raise ValueError("データの期間が短すぎて train / test に分けられません")
    test_start = dates[split_index]
    train_end = dates[split_index - horizon - 1]

    train = data[data.index <= train_end]
    test = data[data.index >= test_start]
    return train, test


def evaluate(y_true, y_pred) -> dict:
    """対数リターンの誤差（MAE / RMSE）と、上がる・下がるの的中率。"""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "direction_accuracy": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }


def train_and_evaluate(prices: dict[str, pd.DataFrame]) -> tuple[dict, dict, pd.DataFrame]:
    """全モデルを train 期間で学習し、test 期間で評価する。

    Returns:
        models: {モデル名: 学習済みモデル}
        metrics: 期間・件数・モデルごとの評価指標
        test_result: test 期間の実際のリターン（target）と各モデルの予測値
    """
    data = build_dataset(prices, horizon=HORIZON)
    train, test = split_by_date(data)
    X_train, y_train = train[FEATURE_COLUMNS], train["target"]
    X_test, y_test = test[FEATURE_COLUMNS], test["target"]

    models = create_models()
    test_result = test[["symbol", "target"]].copy()
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        test_result[name] = model.predict(X_test)
        results[name] = evaluate(y_test, test_result[name])

    metrics = {
        "horizon_days": HORIZON,
        "features": FEATURE_COLUMNS,
        "n_symbols": len(prices),
        "train_period": [str(train.index.min().date()), str(train.index.max().date())],
        "test_period": [str(test.index.min().date()), str(test.index.max().date())],
        "n_train": len(train),
        "n_test": len(test),
        "models": results,
    }
    return models, metrics, test_result


def save_models(models: dict, metrics: dict) -> None:
    """学習済みモデルと評価結果を models/ に保存する。"""
    MODELS_DIR.mkdir(exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


def main() -> None:
    symbols = get_nasdaq100_symbols()
    print(f"{len(symbols)} 銘柄の株価（全期間）をダウンロードしています...")
    prices = download_prices(symbols, period="max")
    print(f"{len(prices)} 銘柄のデータを取得しました")

    models, metrics, _ = train_and_evaluate(prices)
    print(f"train: {metrics['train_period'][0]} 〜 {metrics['train_period'][1]} ({metrics['n_train']:,} 行)")
    print(f"test : {metrics['test_period'][0]} 〜 {metrics['test_period'][1]} ({metrics['n_test']:,} 行)")
    for name, result in metrics["models"].items():
        print(f"{name:>14}: {result}")

    save_models(models, metrics)
    print(f"モデルと評価結果を {MODELS_DIR} に保存しました")


if __name__ == "__main__":
    main()
