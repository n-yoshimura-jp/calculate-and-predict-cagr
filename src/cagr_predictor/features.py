"""特徴量と目的変数を作る。

特徴量は「その日までのデータ」だけで計算する（未来の値を使わない）。
目的変数は「HORIZON 営業日後（約1年後）までの対数リターン」。
"""

import numpy as np
import pandas as pd

HORIZON = 252  # 何営業日後を予測するか（1年 ≒ 252営業日）

FEATURE_COLUMNS = [
    "return_20d",
    "return_60d",
    "return_120d",
    "return_252d",
    "volatility_60d",
    "volatility_252d",
    "close_to_ma50",
    "close_to_ma200",
    "drawdown_252d",
    "volume_ratio",
]

# 特徴量を1行作るのに必要な最低日数（252日リターン = 253日分の価格）
MIN_HISTORY = 253


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """1銘柄の日足（Close, Volume）から特徴量を作る。"""
    close = df["Close"]
    volume = df["Volume"]
    daily_return = close.pct_change()

    features = pd.DataFrame(
        {
            # 過去のリターン（モメンタム）
            "return_20d": close.pct_change(20),
            "return_60d": close.pct_change(60),
            "return_120d": close.pct_change(120),
            "return_252d": close.pct_change(252),
            # 日次リターンの標準偏差（値動きの大きさ）
            "volatility_60d": daily_return.rolling(60).std(),
            "volatility_252d": daily_return.rolling(252).std(),
            # 移動平均からの乖離率
            "close_to_ma50": close / close.rolling(50).mean() - 1,
            "close_to_ma200": close / close.rolling(200).mean() - 1,
            # 直近1年の高値からの下落率
            "drawdown_252d": close / close.rolling(252).max() - 1,
            # 直近20日の平均出来高 ÷ 直近1年の平均出来高
            "volume_ratio": volume.rolling(20).mean() / volume.rolling(252).mean() - 1,
        },
        index=df.index,
    )
    # 出来高0の日などで割り算が inf になった場合は欠損扱いにする
    return features.replace([np.inf, -np.inf], np.nan)


def make_target(close: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """horizon 営業日後までの対数リターン log(close[t + horizon] / close[t])。"""
    return np.log(close.shift(-horizon) / close)


def build_dataset(prices: dict[str, pd.DataFrame], horizon: int = HORIZON) -> pd.DataFrame:
    """全銘柄の特徴量と目的変数を縦に結合した学習用データを作る。

    列: symbol, FEATURE_COLUMNS..., target。index は日付。
    """
    frames = []
    for symbol, df in prices.items():
        data = make_features(df)
        data["target"] = make_target(df["Close"], horizon)
        data.insert(0, "symbol", symbol)
        frames.append(data.dropna())

    if not frames:
        raise ValueError("学習データが作れませんでした（価格データが空です）")
    return pd.concat(frames).sort_index()
