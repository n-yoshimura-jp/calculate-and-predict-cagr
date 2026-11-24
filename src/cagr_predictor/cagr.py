"""CAGR（年平均成長率）と年次の騰落率を計算する。"""

import pandas as pd

DAYS_PER_YEAR = 365.25


def calc_cagr(close: pd.Series, last_years: int | None = None) -> dict:
    """終値の系列から CAGR を計算する。

    Args:
        close: 日付を index に持つ終値の系列（古い順）。
        last_years: 直近 N 年だけで計算する場合に指定する。None なら全期間。

    Returns:
        開始日・終了日・年数・開始/終了価格・CAGR（小数。0.2 = 20%）。
    """
    close = close.dropna()
    if last_years is not None and not close.empty:
        close = close[close.index >= close.index[-1] - pd.DateOffset(years=last_years)]
    if len(close) < 2:
        raise ValueError("CAGR の計算には2日分以上の価格が必要です")

    start_date = close.index[0]
    end_date = close.index[-1]
    years = (end_date - start_date).days / DAYS_PER_YEAR
    if years <= 0:
        raise ValueError("期間が0日のため CAGR を計算できません")

    start_price = float(close.iloc[0])
    end_price = float(close.iloc[-1])
    cagr = (end_price / start_price) ** (1 / years) - 1

    return {
        "start_date": start_date.date(),
        "end_date": end_date.date(),
        "years": round(years, 2),
        "start_price": start_price,
        "end_price": end_price,
        "cagr": cagr,
    }


def annual_growth(close: pd.Series) -> pd.DataFrame:
    """年ごとの騰落率を計算する。

    列:
        first_close: その年の最初の取引日の終値
        last_close: その年の最後の取引日の終値
        within_year_return: その年の初値 → 年末の騰落率
        year_on_year_return: 前年末 → 当年末の騰落率（最初の年は NaN）
    """
    close = close.dropna().sort_index()
    if close.empty:
        raise ValueError("価格データが空です")

    by_year = close.groupby(close.index.year)
    first = by_year.first()
    last = by_year.last()

    table = pd.DataFrame(
        {
            "first_close": first,
            "last_close": last,
            "within_year_return": last / first - 1,
            "year_on_year_return": last.pct_change(),
        }
    )
    table.index.name = "year"
    return table


def yoy_table(closes: dict[str, pd.Series]) -> pd.DataFrame:
    """複数銘柄の前年比騰落率を「年 × 銘柄」の表にする。"""
    table = pd.DataFrame(
        {symbol: annual_growth(close)["year_on_year_return"] for symbol, close in closes.items()}
    )
    table.index.name = "year"
    return table.sort_index()
