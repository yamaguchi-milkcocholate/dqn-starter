import pytest
import sys
import pandas as pd
import numpy as np
from pathlib import Path

rootdir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(rootdir))
import src.markets.ppobaseline3sharpratio as markets


@pytest.fixture
def market():
    df = pd.DataFrame(
        {
            "price": 1 + np.arange(10) * 0.0001,
            "max_price": 1 + np.arange(10) * 0.0001,
            "min_price": 1 + np.arange(10) * 0.0001,
            "buy_price": 1 + np.arange(10) * 0.0001,
            "sell_price": 1 + np.arange(10) * 0.0001,
            "f": 1 + np.arange(10) * 0.0001,
        }
    )
    return markets.Market(
        df=df,
        features=["f"],
        action_params={"NUM_DISCRETE": 1, "MAX_SPREAD": 0.001},
        n_lag=1,
        is_single_transaction=False,
    )


def test_hold(market: markets.Market):
    for _ in range(market.num_steps):
        obs = market.state()
        assert obs[-1] == float(1)
        sharp_ratio, done = market.step(action=market.action_parser.hold_index)
        if done:
            assert sharp_ratio == -1
        else:
            assert sharp_ratio == 0


def test_fill(market: markets.Market):
    _ = market.state()
    sharp_ratio, done = market.step(action=0)
    assert market.fb
    assert sharp_ratio < 0

    _ = market.state()
    sharp_ratio, done = market.step(action=3)
    assert not market.fb
    assert not market.fs
    assert not done
    assert sharp_ratio < 0

    for _ in range(market.num_steps - 2):
        _ = market.state()
        sharp_ratio, done = market.step(action=market.action_parser.hold_index)
        assert not market.fb
        assert not market.fs
        if done:
            assert sharp_ratio == 0
        else:
            assert sharp_ratio == 0
