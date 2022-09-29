import torch
import torch.nn as nn
import pytest
import gym
import sys
import numpy as np
from pathlib import Path

rootdir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(rootdir))
from src.models.lagfeature import TimeAxisCombinedExtractor


@pytest.fixture
def dict_space() -> gym.spaces.Dict:
    dict_space = {}
    for i in range(5):
        dict_space[f"state_{i}"] = gym.spaces.Box(
            low=np.full(3, -1).astype(np.float32),
            high=np.full(3, 1).astype(np.float32),
        )
    return gym.spaces.Dict(dict_space)


@pytest.fixture
def observations() -> dict[str, torch.Tensor]:
    obs = {}
    for i in range(5):
        obs[f"state_{i}"] = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    return obs


def test_sample(dict_space, observations):
    extr = TimeAxisCombinedExtractor(observation_space=dict_space)
    y = extr(observations)
    print(y.shape)
    assert y.shape == (2, 64)
