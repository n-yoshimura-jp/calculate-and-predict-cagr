"""NASDAQ-100 の銘柄リストと株価データを取得する。"""

from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# 構成銘柄のスナップショット（README に取得日を記載）
SYMBOLS_CSV = DATA_DIR / "nasdaq100.csv"


def get_nasdaq100_symbols() -> list[str]:
    """同梱の CSV から NASDAQ-100 の構成銘柄を読み込む。"""
    return pd.read_csv(SYMBOLS_CSV)["symbol"].tolist()


def download_prices(symbols: list[str], period: str = "10y") -> dict[str, pd.DataFrame]:
    """複数銘柄の日足を取得し、{銘柄: DataFrame[Close, Volume]} の形で返す。

    データが取れなかった銘柄は結果に含めない。
    """
    raw = yf.download(
        symbols,
        period=period,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
    )
    if raw is None or raw.empty:
        return {}

    prices = {}
    for symbol in symbols:
        if symbol not in raw.columns.get_level_values(0):
            continue
        df = raw[symbol][["Close", "Volume"]].dropna()
        if not df.empty:
            prices[symbol] = df
    return prices


def download_price(symbol: str, period: str = "max") -> pd.DataFrame:
    """1銘柄の日足を取得する。データが無い場合は空の DataFrame を返す。"""
    return download_prices([symbol], period=period).get(symbol, pd.DataFrame())
