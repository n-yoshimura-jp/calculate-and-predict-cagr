import math

import pytest

from cagr_predictor.features import HORIZON, build_dataset
from cagr_predictor.predict import predict_price
from cagr_predictor.train import split_by_date, train_and_evaluate


def test_split_by_date_leaves_gap(long_prices):
    data = build_dataset({"AAA": long_prices})
    train, test = split_by_date(data)
    dates = data.index.unique().sort_values()
    # train の最後と test の最初の間に horizon 日以上のすき間がある
    gap = dates.get_loc(test.index.min()) - dates.get_loc(train.index.max())
    assert gap > HORIZON


def test_split_by_date_rejects_short_data(sample_prices):
    data = build_dataset({"AAA": sample_prices})
    with pytest.raises(ValueError):
        split_by_date(data)


def test_train_and_predict(long_prices):
    models, metrics, test_result = train_and_evaluate({"AAA": long_prices, "BBB": long_prices * 2})
    assert set(models) == {"baseline", "ridge", "random_forest", "lightgbm"}
    assert metrics["horizon_days"] == HORIZON
    assert set(test_result.columns) == {"symbol", "target", *models}

    result = predict_price(long_prices, models["lightgbm"])
    assert result["predicted_growth"] == pytest.approx(math.exp(result["predicted_return"]) - 1)
    assert result["predicted_price"] == pytest.approx(
        result["last_close"] * math.exp(result["predicted_return"])
    )
