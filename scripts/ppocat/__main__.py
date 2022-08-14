import numpy as np
import torch
import multiprocessing
import gc
import sys
import json
from pathlib import Path
from typing import *
from collections import deque
from tqdm import tqdm

rootdir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(rootdir))

import src.models.ppocat as models
import src.markets.ppocat as markets
import src.utils.data as data
import src.utils.plot as plot


def _run_episode(
    agent: models.PPO, market: markets.Market, device: torch.device
) -> Tuple[models.ReplayMemory, deque]:
    memory = models.ReplayMemory()
    rewards = deque()
    for _ in range(market.num_steps):
        state = torch.tensor(
            market.state()[np.newaxis, :, :], device=device, dtype=torch.float32
        )

        action, action_logprob = agent.select_action(state=state)

        reward, is_end = market.step(action=action.detach().item())
        rewards.append(reward)
        reward = torch.tensor([reward], device=device, dtype=torch.float32)

        memory.push(
            state=state,
            action=action,
            reward=reward,
            logprob=action_logprob,
            is_end=is_end,
        )

        if is_end:
            break
    return memory, np.mean(rewards)


def run_episode(args):
    return _run_episode(**args)


def main():
    exptdir = Path(__file__).resolve().parent
    savedir = exptdir / "out"
    savedir.mkdir(exist_ok=True, parents=True)

    config = json.load(open(exptdir / "config.json", "r"))
    ppo_params = config["ppo_params"]
    train_params = config["train_params"]
    action_params = config["action_params"]

    device = torch.device("cpu")

    df, features = data.load_bybit_data(
        num_devide=train_params["NUM_DEVIDE"], lags=train_params["LAGS"]
    )
    n_actions = (action_params["NUM_DISCRETE"] * 2 + 1) * 2 + 2
    state_dim = 9 + len(features)

    df_train, df_eval = (
        df.loc[df["fold"] != (train_params["NUM_DEVIDE"] - 1)].reset_index(drop=True),
        df.loc[df["fold"] == (train_params["NUM_DEVIDE"] - 1)].reset_index(drop=True),
    )

    agent = models.PPO(
        state_dim=state_dim,
        lag_dim=train_params["N_LAG"],
        action_dim=n_actions,
        device=device,
        params=ppo_params,
    )

    num_async = multiprocessing.cpu_count() - 1
    num_async = 1
    pool = multiprocessing.Pool(num_async)

    log = list()
    best_reward = -np.inf
    episode_durations = []
    for i_episode in tqdm(range(train_params["NUM_EPISODES"])):
        params = list()
        for _ in range(num_async):
            params.append(
                {
                    "agent": agent,
                    "market": markets.random_market(
                        df=df_train,
                        features=features,
                        num_steps=train_params["NUM_STEPS"],
                        action_params=action_params,
                        n_lag=train_params["N_LAG"],
                    ),
                    "device": device,
                }
            )
        result = pool.map(run_episode, params)
        memories, rewards = [r[0] for r in result], [r[1] for r in result]

        episode_durations += rewards
        train_reward = np.mean(rewards)
        agent.merge_memories(memories=memories)

        del memories, rewards
        gc.collect()

        if (i_episode + 1) % train_params["UPDATE_INTERVAL"] == 0:
            agent.update()

        plot.plot(
            x=episode_durations,
            filepath=savedir / "episode_durations.png",
            xlabel="episode",
            ylabel="avg reward",
        )

        if (i_episode + 1) % train_params["EVAL_LOG_INTERVAL"] == 0:
            market = markets.random_market(
                df=df_eval,
                features=features,
                num_steps=train_params["NUM_STEPS"],
                action_params=action_params,
                n_lag=train_params["N_LAG"],
            )

            rewards = list()
            for _ in range(market.num_steps):
                state = torch.tensor(
                    market.state()[np.newaxis, :, :], device=device, dtype=torch.float32
                )

                action = agent.select_deterministical_action(state=state)

                reward, _ = market.step(action=action.detach().item())
                rewards.append(reward)

            eval_returns = market.get_return()["rtn"].values
            plot.plot(
                x=eval_returns,
                filepath=savedir / f"eval_returns_{i_episode + 1}.png",
                xlabel="steps",
                ylabel="return",
            )
            plot.plot(
                x=rewards,
                filepath=savedir / f"eval_reward_{i_episode + 1}.png",
                xlabel="steps",
                ylabel="reward",
            )
            market.get_return().to_csv(
                savedir / f"eval_log_{i_episode + 1}.csv", index=False
            )

            eval_reward = np.mean(rewards)
            if eval_reward > best_reward:
                plot.plot(
                    x=rewards,
                    filepath=savedir / "eval_reward_best.png",
                    xlabel="steps",
                    ylabel="reward",
                )
                plot.plot(
                    x=eval_returns,
                    filepath=savedir / "eval_returns_best.png",
                    xlabel="steps",
                    ylabel="return",
                )
                agent.save_model(filepath=savedir / "model_best.pt")
                best_reward = eval_reward

            del market
            gc.collect()
        else:
            eval_returns = [np.nan]
            eval_reward = np.nan

        log.append(
            {
                "episode": i_episode + 1,
                "train_reward": train_reward,
                "train_return": None,
                "eval_reward": eval_reward,
                "eval_return": eval_returns[-1],
            }
        )
        gc.collect()

    del agent, pool
    gc.collect()


if __name__ == "__main__":
    main()
