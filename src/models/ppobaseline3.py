from pathlib import Path
from typing import *

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.logger import configure

import src.markets.ppobaseline3 as markets


def create_model(
    savedir: Path, env: markets.MarketEnv, ppo_params: Dict[str, Any] = {}
) -> PPO:
    logger = configure(str(savedir), ["stdout", "csv"])
    model = PPO(policy=ActorCriticPolicy, env=env, verbose=1, **ppo_params)
    model.set_logger(logger)

    return model
