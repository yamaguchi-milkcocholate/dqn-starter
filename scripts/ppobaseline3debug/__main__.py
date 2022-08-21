import sys
import json
from pathlib import Path
from typing import *

rootdir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(rootdir))

import src.models.ppobaseline3 as models
import src.markets.ppobaseline3 as markets
import src.utils.data as data
import src.utils.plot as plot


def main():
    exptdir = Path(__file__).resolve().parent
    savedir = exptdir / "out"
    savedir.mkdir(exist_ok=True, parents=True)

    config = json.load(open(exptdir / "config.json", "r"))
    ppo_params = config["ppo_params"]
    train_params = config["train_params"]
    action_params = config["action_params"]

    df, features = data.load_bybit_data(
        num_devide=train_params["NUM_DEVIDE"], lags=train_params["LAGS"]
    )

    df_train, df_eval = (
        df.loc[df["fold"] != (train_params["NUM_DEVIDE"] - 1)].reset_index(drop=True),
        df.loc[df["fold"] == (train_params["NUM_DEVIDE"] - 1)].reset_index(drop=True),
    )

    train_env = markets.MarketEnv(
        df=df_train,
        features=features,
        num_steps=train_params["NUM_TAIN_ENV_STEPS"],
        action_params=action_params,
        n_lag=train_params["N_LAG"],
        market_cls=markets.DummyMarket,
        is_single_transaction=True,
    )
    eval_env = markets.MarketEnv(
        df=df_eval,
        features=features,
        num_steps=train_params["NUM_EVAL_ENV_STEPS"],
        action_params=action_params,
        n_lag=train_params["N_LAG"],
        market_cls=markets.DummyMarket,
        is_single_transaction=False,
    )
    model = models.create_model(
        savedir=savedir / "log", env=train_env, ppo_params=ppo_params
    )
    model.learn(
        total_timesteps=train_params["NUM_STEPS"],
        eval_env=eval_env,
        eval_freq=train_params["EVAL_FREQ"],
    )
    model.save(savedir / "model")

    plot.plot_from_baseline3(fromdir=savedir / "log", todir=savedir)


if __name__ == "__main__":
    main()
