import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path

rootdir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(rootdir))
import src.markets.ppocat1 as markets


@pytest.fixture
def df() -> pd.DataFrame:
    feature_cols = ["f1", "f2"]
    cols = ["price", "max_price", "min_price", "buy_price", "sell_price"]
    values = np.random.randn(10, len(feature_cols) + len(cols))
    df = pd.DataFrame({c: values[:, i] for i, c in enumerate(feature_cols + cols)})
    return df


def test_sample1(df: pd.DataFrame):
    market = markets.random_market(
        df=df,
        features=["f1", "f2"],
        num_steps=5,
        action_params={"NUM_DISCRETE": 5, "MAX_SPREAD": 0.01},
        n_lag=3,
    )
    assert market.num_steps == 5

    for i in range(5):
        _, is_end = market.step(action=22)
        if i != 4:
            assert not is_end
        else:
            assert is_end


def test_sample2(df: pd.DataFrame):
    market = markets.random_market(
        df=df,
        features=["f1", "f2"],
        num_steps=5,
        action_params={"NUM_DISCRETE": 5, "MAX_SPREAD": 0.01},
        n_lag=3,
    )
    assert market.num_steps == 5

    for i in range(market.num_steps):
        _, is_end = market.step(action=22)
        if i != 4:
            assert not is_end
        else:
            assert is_end
