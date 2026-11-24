import numpy as np
import pandas as pd
import pytest

from cagr_predictor.cagr import annual_growth, calc_cagr, yoy_table


def test_calc_cagr_doubles_in_one_year():
    close = pd.Series([100.0, 200.0], index=pd.to_datetime(["2020-01-01", "2021-01-01"]))
    assert calc_cagr(close)["cagr"] == pytest.approx(1.0, abs=0.01)


def test_calc_cagr_ten_percent_per_year():
    close = pd.Series([100.0, 100 * 1.1**4], index=pd.to_datetime(["2020-01-01", "2024-01-01"]))
    assert calc_cagr(close)["cagr"] == pytest.approx(0.1, abs=0.001)


def test_calc_cagr_last_years():
    # 最初の2年で2倍、その後の2年は横ばい → 直近2年の CAGR は 0%
    close = pd.Series(
        [100.0, 200.0, 200.0], index=pd.to_datetime(["2020-01-01", "2022-01-01", "2024-01-01"])
    )
    result = calc_cagr(close, last_years=2)
    assert result["start_date"] == pd.Timestamp("2022-01-01").date()
    assert result["cagr"] == pytest.approx(0.0)


def test_calc_cagr_needs_two_points():
    with pytest.raises(ValueError):
        calc_cagr(pd.Series([100.0], index=pd.to_datetime(["2020-01-01"])))


def test_annual_growth():
    close = pd.Series(
        [100.0, 120.0, 110.0, 132.0],
        index=pd.to_datetime(["2020-01-02", "2020-12-31", "2021-01-04", "2021-12-31"]),
    )
    table = annual_growth(close)
    assert list(table.index) == [2020, 2021]
    assert table.loc[2020, "within_year_return"] == pytest.approx(0.2)
    assert table.loc[2021, "within_year_return"] == pytest.approx(0.2)
    assert np.isnan(table.loc[2020, "year_on_year_return"])  # 前年が無い
    assert table.loc[2021, "year_on_year_return"] == pytest.approx(0.1)  # 120 → 132


def test_yoy_table_aligns_years(sample_prices):
    close = sample_prices["Close"]
    table = yoy_table({"AAA": close, "BBB": close[close.index.year >= 2021]})
    assert list(table.columns) == ["AAA", "BBB"]
    assert list(table.index) == [2020, 2021, 2022, 2023]
    assert np.isnan(table.loc[2021, "BBB"])  # BBB は2021年が最初の年
