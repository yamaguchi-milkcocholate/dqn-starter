import sys
import json
from pathlib import Path
from typing import *

rootdir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(rootdir))

import src.models.lagfeature as models
import src.markets.lagfeature as markets
import src.utils.data as data
import src.utils.plot as plot
import src.utils.seed as seed


def main(fold: int):
    seed.seed_everything(seed=43)
    exptdir = Path(__file__).resolve().parent
    savedir = exptdir / "out" / f"fold{fold}"
    savedir.mkdir(exist_ok=True, parents=True)

    config = json.load(open(exptdir / "config.json", "r"))
    ppo_params = config["ppo_params"]
    train_params = config["train_params"]
    action_params = config["action_params"]

    df, features = data.load_bybit_data(
        num_divide=train_params["NUM_DEVIDE"],
        interval=train_params["MINUTES"],
    )

    df_train1, df_train2 = (
        df.loc[df["fold"] < (fold - 1)].reset_index(drop=True),
        df.loc[df["fold"] > (fold - 1)].reset_index(drop=True),
    )
    df_eval = df.loc[df["fold"] == (fold - 1)].reset_index(drop=True)

    train_env = markets.DualMarketEnv(
        df1=df_train1,
        df2=df_train2,
        features=features,
        num_steps=train_params["NUM_TAIN_ENV_STEPS"],
        action_params=action_params,
        n_lag=train_params["N_LAG"],
        market_cls=markets.Market,
        is_single_transaction=False,
    )
    eval_env = markets.MarketEnv(
        df=df_eval,
        features=features,
        num_steps=None,
        action_params=action_params,
        n_lag=train_params["N_LAG"],
        market_cls=markets.Market,
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

    eval_market = markets.Market(
        df=df_eval,
        features=features,
        action_params=action_params,
        n_lag=train_params["N_LAG"],
        is_single_transaction=False,
    )
    plot.eval_and_plot(model=model, market=eval_market, savedir=savedir)


if __name__ == "__main__":
    main(fold=int(sys.argv[1]))
