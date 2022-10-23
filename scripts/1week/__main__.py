import sys
import json
import gc
from pathlib import Path
from typing import *

rootdir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(rootdir))

import src.models.lagfeature as models
import src.markets.lagfeaturesharpratio as markets
import src.utils.data as data
import src.utils.plot as plot
import src.utils.seed as seed


def main():
    seed.seed_everything(seed=43)
    exptdir = Path(__file__).resolve().parent

    config = json.load(open(exptdir / "config.json", "r"))
    ppo_params = config["ppo_params"]
    train_params = config["train_params"]
    period_params = config["period"]

    df, features = data.load_bybit_data(
        num_divide=1,
        interval=train_params["MINUTES"],
        use_cache=False,
        ta_config_file="config.json",
    )

    for period_i in range(period_params["NUM_TRIALS"]):
        savedir = exptdir / "out" / f"trial{period_i + 1}"
        savedir.mkdir(exist_ok=True, parents=True)

        df_train, df_eval = data.arange_1week(
            df=df,
            evalday=period_i + period_params["START"],
            interval=train_params["MINUTES"],
        )

        train_env = markets.MarketEnv(
            df=df_train,
            features=features,
            num_steps=288,
            n_lag=train_params["N_LAG"],
            market_cls=markets.Market,
            is_single_transaction=True,
        )
        eval_env = markets.MarketEnv(
            df=df_eval,
            features=features,
            num_steps=None,
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
            eval_log_path=str(savedir),
        )
        model.save(savedir / "final_model")

        plot.plot_from_baseline3(fromdir=savedir / "log", todir=savedir)

        eval_market = markets.Market(
            df=df_eval,
            features=features,
            n_lag=train_params["N_LAG"],
            is_single_transaction=False,
        )
        plot.eval_and_plot(model=model, market=eval_market, savedir=savedir)

        del model, train_env, eval_env, df_train, df_eval
        gc.collect()


if __name__ == "__main__":
    main()
