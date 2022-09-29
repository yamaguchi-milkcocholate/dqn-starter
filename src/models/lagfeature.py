from pathlib import Path
from typing import *

import gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.logger import configure

import src.markets.ppobaseline3 as markets


class FeatureAxisCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, emb_dim: int):
        super(FeatureAxisCombinedExtractor, self).__init__(
            observation_space, features_dim=1
        )
        self._check_state_dim(observation_space=observation_space)

        extractors = {
            key: nn.Sequential(nn.Linear(self.state_dim, emb_dim), nn.ReLU())
            for key in observation_space.spaces.keys()
        }
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = emb_dim * len(observation_space)

    def forward(self, observations) -> torch.Tensor:
        embeds = []
        for key, extractor in self.extractors.items():
            embeds.append(extractor(observations[key]))
        return torch.cat(embeds, dim=1)

    def _check_state_dim(self, observation_space: gym.spaces.Dict) -> None:
        state_dims = []
        for key, subspace in observation_space.spaces.items():
            if not key.startswith("state_"):
                raise Exception("ラグ特徴量の状態を示さないkey")
            state_dims.append(subspace.shape[0])

        if max(state_dims) != min(state_dims):
            raise Exception("特徴量数が時系列で異なる")

        self.state_dim = state_dims[0]


class TimeAxisCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(TimeAxisCombinedExtractor, self).__init__(
            observation_space, features_dim=1
        )
        self._check_state_dim(observation_space=observation_space)

        self.convs = nn.Sequential(
            *[
                nn.Conv1d(self.state_dim, 32, 2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
                nn.Dropout(0.2),
                nn.Conv1d(32, 64, 2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
                nn.Dropout(0.2),
                nn.Conv1d(64, 128, 2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
                nn.Dropout(0.2),
            ]
        )
        self.fc = nn.Linear(128, 64)
        self._features_dim = 64

    def forward(self, observations) -> torch.Tensor:
        X = torch.cat([v.unsqueeze(0) for v in observations.values()], 0)
        X = torch.permute(X, (1, 2, 0))
        X = self.convs(X)
        X = torch.mean(X, dim=2)
        X = self.fc(X)
        return X

    def _check_state_dim(self, observation_space: gym.spaces.Dict) -> None:
        state_dims = []
        for key, subspace in observation_space.spaces.items():
            if not key.startswith("state_"):
                raise Exception("ラグ特徴量の状態を示さないkey")
            state_dims.append(subspace.shape[0])

        if max(state_dims) != min(state_dims):
            raise Exception("特徴量数が時系列で異なる")

        self.state_dim = state_dims[0]


def create_model(
    savedir: Path, env: markets.MarketEnv, ppo_params: Dict[str, Any] = {}
) -> PPO:

    if "policy_kwargs" not in ppo_params.keys():
        raise Exception("policy_kwargsが設定されていない")
    else:
        pkwargs = ppo_params["policy_kwargs"]

        if "features_extractor_class" not in pkwargs.keys():
            pkwargs["features_extractor_class"] = FeatureAxisCombinedExtractor
            pkwargs["features_extractor_kwargs"] = {"emb_dim": 16}
        else:
            pkwargs["features_extractor_class"] = globals()[
                pkwargs["features_extractor_class"]
            ]

        ppo_params["policy_kwargs"] = pkwargs

    print(ppo_params)

    logger = configure(str(savedir), ["stdout", "csv"])
    model = PPO(policy=ActorCriticPolicy, env=env, verbose=1, **ppo_params)
    model.set_logger(logger)
    return model
