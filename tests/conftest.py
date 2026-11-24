import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """ランダムウォークの合成株価（約3年分 = 800営業日）。"""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=800)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(dates))))
    volume = rng.integers(1_000_000, 2_000_000, len(dates)).astype(float)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=dates)


@pytest.fixture
def long_prices() -> pd.DataFrame:
    """学習・評価のテスト用に長くした合成株価（約8年分 = 2,000営業日）。"""
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2015-01-01", periods=2000)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(dates))))
    volume = rng.integers(1_000_000, 2_000_000, len(dates)).astype(float)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=dates)
