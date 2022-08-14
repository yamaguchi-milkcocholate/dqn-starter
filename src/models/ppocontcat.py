import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import *
from collections import namedtuple, deque

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "logprob", "is_end")
)


class ReplayMemory(object):
    def __init__(self):
        self.memory = deque()

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        logprob: torch.Tensor,
        is_end: bool,
    ) -> None:
        """Save a transition"""
        self.memory.append(Transition(state, action, reward, logprob, is_end))

    def extend(self, memory: "ReplayMemory") -> None:
        self.memory.extend(memory.memory)

    def __len__(self) -> int:
        return len(self.memory)

    def clear(self) -> None:
        self.memory = deque()

    def get_batches(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = Transition(*zip(*self.memory))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        logprob_batch = torch.cat(batch.logprob)
        is_end_batch = np.vstack(batch.is_end)

        return state_batch, action_batch, reward_batch, logprob_batch, is_end_batch


class Actor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, clips: List[Dict[str, Any]]):
        super(Actor, self).__init__()
        self.clips = clips

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
        )
        self.transforms = {
            "softmax": nn.Softmax(dim=1),
            "tanh": nn.Tanh(),
        }

    def forward(self, x):
        x = self.net(x)
        for clip in self.clips:
            x[:, clip["dims"]] = clip["coef"] * self.transforms[clip["name"]](
                x[:, clip["dims"]]
            )
        return x


class ActorCritic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, device: torch.device, spread: float
    ):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mean_dim = action_dim
        self.cov_dim = action_dim
        # self.cov_dim = (action_dim + 1) * action_dim // 2

        self.device = device

        self.actor = Actor(
            input_dim=state_dim,
            output_dim=self.mean_dim + self.cov_dim,
            clips=[
                {"dims": [0], "coef": spread, "name": "tanh"},
                # {"dims": [1, 2, 3], "coef": 1, "name": "softmax"},
                {"dims": [4], "coef": spread / 2, "name": "tanh"},
            ],
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.softplus = nn.Softplus()

    def forward(self):
        raise NotImplementedError()

    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self._get_normal_dist(state=state)

        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach()

    def deterministical_act(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embed = self.actor(state)
        _mean = embed[:, : self.action_dim]
        return _mean.detach()

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._get_normal_dist(state=state)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

    def _get_normal_dist(
        self, state: torch.Tensor
    ) -> torch.distributions.MultivariateNormal:
        embed = self.actor(state)

        _mean = embed[:, : self.action_dim]
        _cov_as_vec = embed[0, self.action_dim :]
        _cov = torch.diag(_cov_as_vec)
        #         _cov = torch.zeros((self.action_dim, self.action_dim), dtype=torch.float32)

        #         tmp = 0
        #         for i in range(self.action_dim):
        #             for j in range(i, self.action_dim):
        #                 if i == j:
        #                     _cov[i, j] = self.softplus(_cov_as_vec[tmp + j])
        #                 else:
        #                     _cov[i, j] = self.softplus(_cov_as_vec[tmp + j])
        #             tmp += (self.action_dim - i - 1)
        _cov = torch.mm(_cov, torch.t(_cov))
        return torch.distributions.MultivariateNormal(loc=_mean, covariance_matrix=_cov)


class PPO(object):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        params: Dict[str, Any],
    ):
        self.device = device

        self.gamma = params["GAMMA"]
        self.eps_clip = params["EPS_CLIP"]
        self.K_epochs = params["K_EPOCHS"]

        self.memory = ReplayMemory()

        self.policy = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            spread=params["SPREAD"],
        ).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": params["LR_ACTOR"]},
                {"params": self.policy.critic.parameters(), "lr": params["LR_CRITIC"]},
            ]
        )

        self.policy_old = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            spread=params["SPREAD"],
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        with torch.no_grad():
            action, action_logprob = self.policy_old.act(state=state)
            return action, action_logprob

    def select_deterministical_action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action = self.policy_old.deterministical_act(state=state)
            return action

    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        logprob: torch.Tensor,
        is_end: bool,
    ) -> None:
        self.memory.push(
            state=state, action=action, reward=reward, logprob=logprob, is_end=is_end
        )

    def merge_memories(self, memories: List[ReplayMemory]) -> None:
        self.memory = ReplayMemory()
        for _mem in memories:
            self.memory.extend(_mem)

    def update(self):
        (
            state_batch,
            action_batch,
            reward_batch,
            logprob_batch,
            is_end_batch,
        ) = self.memory.get_batches()

        old_rewards = self._calc_cumsum_discount_rewards(
            reward_batch=reward_batch, is_end_batch=is_end_batch
        )

        old_states = state_batch.detach().to(self.device)
        old_actions = action_batch.detach().to(self.device)
        old_logprobs = logprob_batch.detach().to(self.device)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                state=old_states, action=old_actions
            )
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = old_rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, old_rewards)
                - 0.01 * dist_entropy
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.memory.clear()

    def _calc_cumsum_discount_rewards(
        self, reward_batch: torch.Tensor, is_end_batch: np.ndarray
    ) -> torch.Tensor:
        reward_batch = reward_batch.squeeze()
        is_end_batch = is_end_batch.squeeze()
        cumsum_discounted_rewards = deque()
        _discounted_reward = 0
        for reward, is_end in zip(reversed(reward_batch), reversed(is_end_batch)):
            if is_end:
                _discounted_reward = 0
            _discounted_reward = reward + self.gamma * _discounted_reward
            cumsum_discounted_rewards.appendleft(_discounted_reward)
        cumsum_discounted_rewards = torch.tensor(
            cumsum_discounted_rewards, dtype=torch.float32
        ).to(self.device)

        def normalize(x: torch.Tensor) -> torch.Tensor:
            return (x - x.mean()) / (x.std() + 1e-7)

        return normalize(x=cumsum_discounted_rewards)

    def save_model(self, filepath: Path) -> None:
        torch.save(self.policy_old.state_dict(), filepath)
