import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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


class PPODataset(Dataset):
    def __init__(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        logprob: torch.Tensor,
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.logprob = logprob

    def __len__(self):
        return len(self.reward)

    def __getitem__(self, idx):
        return self.state[idx], self.action[idx], self.reward[idx], self.logprob[idx]


class Actor(nn.Module):
    def __init__(self, input_dim: int, channel_dim: int, output_dim: int):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv1d(channel_dim, 16, kernel_size=1)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=1)
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim * 16, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tanh(self.conv1(x))
        x = self.tanh(self.conv2(x))
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class Critic(nn.Module):
    def __init__(self, input_dim: int, channel_dim: int):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv1d(channel_dim, 16, kernel_size=1)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=1)
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim * 16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tanh(self.conv1(x))
        x = self.tanh(self.conv2(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        lag_dim: int,
        action_dim: int,
        device: torch.device,
    ):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.lag_dim = lag_dim
        self.action_dim = action_dim

        self.device = device

        self.actor = Actor(
            input_dim=state_dim, channel_dim=lag_dim, output_dim=action_dim
        )
        self.critic = Critic(input_dim=state_dim, channel_dim=lag_dim)

    def forward(self):
        raise NotImplementedError()

    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)

        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach()

    def deterministical_act(self, state: torch.Tensor) -> torch.Tensor:
        action_probs = self.actor(state)
        return action_probs.argmax().detach()

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO(object):
    def __init__(
        self,
        state_dim: int,
        lag_dim: int,
        action_dim: int,
        device: torch.device,
        params: Dict[str, Any],
    ):
        self.device = device

        self.gamma = params["GAMMA"]
        self.eps_clip = params["EPS_CLIP"]
        self.K_epochs = params["K_EPOCHS"]
        self.batch_size = params["BATCH_SIZE"]

        self.memory = ReplayMemory()

        self.policy = ActorCritic(
            state_dim=state_dim,
            lag_dim=lag_dim,
            action_dim=action_dim,
            device=device,
        ).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": params["LR_ACTOR"]},
                {"params": self.policy.critic.parameters(), "lr": params["LR_CRITIC"]},
            ]
        )

        self.policy_old = ActorCritic(
            state_dim=state_dim,
            lag_dim=lag_dim,
            action_dim=action_dim,
            device=device,
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

        dataloader = DataLoader(
            PPODataset(
                state=state_batch.detach().to(self.device),
                action=action_batch.detach().to(self.device),
                reward=self._calc_cumsum_discount_rewards(
                    reward_batch=reward_batch, is_end_batch=is_end_batch
                ),
                logprob=logprob_batch.detach().to(self.device),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        for _ in range(self.K_epochs):
            for old_states, old_actions, old_rewards, old_logprobs in dataloader:
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    state=old_states, action=old_actions
                )
                state_values = torch.squeeze(state_values)

                ratios = torch.exp(logprobs - old_logprobs.detach())

                advantages = old_rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages
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
