import pytest
import sys
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

rootdir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(rootdir))
import src.markets.ppobaseline3 as markets
import src.models.ppobaseline3 as models


@pytest.fixture
def df() -> pd.DataFrame:
    feature_cols = ["f1", "f2"]
    cols = ["price", "max_price", "min_price", "buy_price", "sell_price"]
    values = np.random.randn(50, len(feature_cols) + len(cols)) ** 2
    df = pd.DataFrame({c: values[:, i] for i, c in enumerate(feature_cols + cols)})
    return df


def curdir() -> Path:
    return Path(__file__).parent


class Test_create_model関数はMLPのPPOモデルを返す:
    def teardown_method(self, method):
        shutil.rmtree(curdir() / "tmp")

    def test_環境がVecEnvになっている(self, df: pd.DataFrame):
        env = markets.MarketEnv(
            df=df,
            features=["f1", "f2"],
            num_steps=5,
            action_params={"NUM_DISCRETE": 5, "MAX_SPREAD": 0.01},
            n_lag=3,
        )

        model = models.create_model(savedir=curdir() / "tmp", env=env)

        assert isinstance(model.env, VecEnv)
