import numpy as np

from cagr_predictor.features import (
    FEATURE_COLUMNS,
    HORIZON,
    MIN_HISTORY,
    build_dataset,
    make_features,
    make_target,
)


def test_features_do_not_use_future_data(sample_prices):
    """t より後の価格を変えても、t 時点の特徴量は変わらない。"""
    t = 400
    original = make_features(sample_prices).iloc[t]

    changed = sample_prices.copy()
    changed.iloc[t + 1 :] *= 3  # 未来の価格だけを大きく変える
    after = make_features(changed).iloc[t]

    assert np.allclose(original.values, after.values)


def test_features_available_after_min_history(sample_prices):
    features = make_features(sample_prices)
    assert features.iloc[MIN_HISTORY - 2].isna().any()
    assert features.iloc[MIN_HISTORY - 1].notna().all()


def test_target_is_future_log_return(sample_prices):
    target = make_target(sample_prices["Close"])
    close = sample_prices["Close"]
    assert np.isclose(target.iloc[0], np.log(close.iloc[HORIZON] / close.iloc[0]))
    assert target.iloc[-HORIZON:].isna().all()  # 最後の1年分は答えがまだ無い


def test_build_dataset_has_no_missing_values(sample_prices):
    data = build_dataset({"AAA": sample_prices, "BBB": sample_prices * 2})
    assert list(data.columns) == ["symbol", *FEATURE_COLUMNS, "target"]
    assert not data.isna().any().any()
    assert set(data["symbol"]) == {"AAA", "BBB"}
