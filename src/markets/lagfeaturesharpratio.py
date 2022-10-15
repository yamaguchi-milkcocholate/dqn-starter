import numpy as np
import pandas as pd
import gym
from typing import *
from collections import deque

ACTION_DIM = 3
TRADE_STATE_DIM = 5


class Market(object):
    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        n_lag: int,
        is_single_transaction: bool = True,
    ):
        self.action_parser = ActionParser()

        self.market_states_cols = features
        self.market_states = df[self.market_states_cols].values
        self.prices = df["close"].values
        self.raw_df = df[["timestamp", "close", "high", "low", "volume"]]

        self.ob, self.os = False, False
        self.fb, self.fs = False, False
        self.lb, self.ls = None, None
        self.step_from_fb, self.step_from_fs = 0, 0

        self.sum_rtn = 0
        self.rtns = list()
        self.cur_rtn = 0
        self.position_side = None
        self.first_fill = False
        self.last_fill = False

        self.is_transaction_end = False
        self.is_single_transaction = is_single_transaction
        self.num_transaction_done = 0

        self.i = n_lag

        self.n_lag = n_lag
        self.trader_state_que = deque(
            [np.zeros(self.trader_state_dim) for _ in range(n_lag)], maxlen=n_lag
        )

    @property
    def num_steps(self) -> int:
        return self.prices.shape[0] - 1 - self.n_lag

    def step(self, action: int) -> Tuple[float, bool]:
        action, spread = self.action_parser.find_action(action=action)
        price = self.prices[self.i]

        if action == "Buy":
            if self.fb:
                pass
            else:
                if self.os:
                    self.unset_order_sell()
                self.set_order_buy(price=price * (1 - spread))
        elif action == "Sell":
            if self.fs:
                pass
            else:
                if self.ob:
                    self.unset_order_buy()
                self.set_order_sell(price=price * (1 + spread))
        elif action == "Hold":
            pass

        self.first_fill, self.last_fill = False, False

        if self.ob:
            self.fb = True
            self.ob = False
            if self.position_side is None:
                self.position_side = "Buy"
                self.first_fill = True
            else:
                self.last_fill = True

        if self.os:
            self.fs = True
            self.os = False
            if self.position_side is None:
                self.position_side = "Sell"
                self.first_fill = True
            else:
                self.last_fill = True

        if self.fb:
            self.step_from_fb += 1

        if self.fs:
            self.step_from_fs += 1

        if (self.lb is not None) and (self.ls is not None):
            self.cur_rtn = self.ls / self.lb - 1
        elif self.lb is not None:
            self.cur_rtn = price / self.lb - 1
        elif self.ls is not None:
            self.cur_rtn = self.ls / price - 1
        else:
            self.cur_rtn = 0
        sharp_ratio = self.calc_sharp_ratio()

        if self.fb and self.fs:
            self.step_from_fb, self.step_from_fs = 0, 0
            self.sum_rtn += self.ls / self.lb - 1
            self.cur_rtn = 0
            self.fb, self.fs = False, False
            self.lb, self.ls = None, None
            self.position_side = None
            self.num_transaction_done += 1
            if self.is_single_transaction:
                self.is_transaction_end = True

        if self.is_transaction_end:
            if self.cur_rtn > 0:
                terminal_reward = 1
            else:
                terminal_reward = -1
        else:
            terminal_reward = 0

        self.rtns.append(
            {
                "i": self.i,
                "rtn": self.sum_rtn,
                "cur_rtn": self.cur_rtn,
                "ob": self.ob,
                "os": self.os,
                "fb": self.fb,
                "fs": self.fs,
                "act": action,
                "spread": spread,
            }
        )
        self.i += 1
        if self.i >= (self.num_steps + self.n_lag):
            self.is_transaction_end = True
            terminal_reward = -1

        self.trader_state_que.append(np.array(self.trader_state))
        return sharp_ratio + terminal_reward, self.is_transaction_end

    def calc_sharp_ratio(self) -> float:
        if self.position_side == "Buy":
            if self.first_fill:
                logdiff = np.log(self.prices[self.i + 1] / self.lb)
            elif self.last_fill:
                logdiff = np.log(self.ls / self.prices[self.i])
            else:
                logdiff = np.log(self.prices[self.i + 1] / self.prices[self.i])
        elif self.position_side == "Sell":
            if self.first_fill:
                logdiff = np.log(self.ls / self.prices[self.i + 1])
            elif self.last_fill:
                logdiff = np.log(self.prices[self.i] / self.lb)
            else:
                logdiff = np.log(self.prices[self.i] / self.prices[self.i + 1])
        else:
            logdiff = np.log(1)
        return logdiff

    @property
    def trader_state(self) -> List[float]:
        return [
            int(self.fb),
            int(self.fs),
            self.cur_rtn,
            np.log1p(self.step_from_fb),
            np.log1p(self.step_from_fs),
        ]

    @property
    def trader_state_dim(self) -> int:
        return len(self.trader_state)

    def set_order_buy(self, price):
        self.ob = True
        self.lb = price
        self.step_from_ob = 0

    def set_order_sell(self, price):
        self.os = True
        self.ls = price
        self.step_from_os = 0

    def unset_order_buy(self):
        self.ob = False
        self.lb = None
        self.step_from_ob = 0

    def unset_order_sell(self):
        self.os = False
        self.ls = None
        self.step_from_os = 0

    def state(self) -> np.ndarray:
        trade_state = np.array(self.trader_state_que)
        market_state = self.market_states[self.i - self.n_lag + 1 : self.i + 1]
        state_dict = {}
        for i, (_trade, _market) in enumerate(
            zip(trade_state[::-1], market_state[::-1])
        ):
            state_dict[f"state_{i}"] = np.clip(np.hstack([_trade, _market]), -1, 1)
        return state_dict

    def get_return(self) -> pd.DataFrame:
        return pd.DataFrame(self.rtns)


class DummyMarket(Market):
    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        n_lag: int,
        is_single_transaction: bool = True,
    ):
        super().__init__(
            df=df,
            features=features,
            n_lag=n_lag,
            is_single_transaction=is_single_transaction,
        )

    def step(self, action: int) -> Tuple[float, bool]:
        reward, is_end = super().step(action=action)
        if self.action_parser.hold_index == action:
            reward = 100
        else:
            reward = -100
        return reward, is_end


def random_market(
    df: pd.DataFrame,
    features: List[str],
    n_lag: int,
    num_steps: Optional[int] = None,
    market: Optional["Market"] = None,
    is_single_transaction: bool = True,
):
    if market is None:
        market = Market

    if num_steps is not None:
        idx = np.random.randint(n_lag, df.shape[0] - 2 - num_steps + 1)
        _df = df.iloc[idx - n_lag : idx + num_steps + 1].reset_index(drop=True)
    else:
        _df = df

    return market(
        df=_df,
        features=features,
        n_lag=n_lag,
        is_single_transaction=is_single_transaction,
    )


class ActionParser(object):
    def __init__(self) -> None:
        self.cat_action_dict = self._arange_actions()

    def find_action(self, action: int) -> Tuple[str, float]:
        return self.cat_action_dict[action]

    def _arange_actions(self) -> Dict[int, Tuple[str, float]]:
        cat_action_dict = {
            0: ("Buy", 0.0),
            1: ("Sell", 0.0),
            2: ("Hold", None),
        }
        return cat_action_dict

    @property
    def hold_index(self) -> int:
        return (self.num_discrete * 2 + 1) * 2

    @property
    def cancel_index(self) -> int:
        return (self.num_discrete * 2 + 1) * 2 + 1


class MarketEnv(gym.Env):
    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        n_lag: int,
        num_steps: Optional[int] = None,
        market_cls: Optional["Market"] = None,
        is_single_transaction: bool = True,
    ):
        self.df = df
        self.features = features
        self.num_steps = num_steps
        self.n_lag = n_lag

        self.action_dim = ACTION_DIM
        self.action_space = gym.spaces.Discrete(self.action_dim)

        self.state_dim = len(features) + TRADE_STATE_DIM
        dict_space = {}
        for i in range(n_lag):
            dict_space[f"state_{i}"] = gym.spaces.Box(
                low=np.full(self.state_dim, -1).astype(np.float32),
                high=np.full(self.state_dim, 1).astype(np.float32),
            )
        self.observation_space = gym.spaces.Dict(dict_space)

        self.market = None
        self.market_cls = market_cls
        self.is_single_transaction = is_single_transaction

    def reset(self) -> np.ndarray:
        self.market = random_market(
            df=self.df,
            features=self.features,
            num_steps=self.num_steps,
            n_lag=self.n_lag,
            market=self.market_cls,
            is_single_transaction=self.is_single_transaction,
        )

        observation = self.market.state()
        return observation

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        reward, done = self.market.step(action=action_index)
        observation = self.market.state()
        return observation, reward, done, {}

    def render(self):
        raise NotImplementedError()


class DualMarketEnv(gym.Env):
    def __init__(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        features: List[str],
        n_lag: int,
        num_steps: Optional[int] = None,
        market_cls: Optional["Market"] = None,
        is_single_transaction: bool = True,
    ):
        if df1.shape[0] != 0:
            self.env1 = MarketEnv(
                df=df1,
                features=features,
                n_lag=n_lag,
                num_steps=num_steps,
                market_cls=market_cls,
                is_single_transaction=is_single_transaction,
            )
        else:
            self.env1 = None

        if df2.shape[0] != 0:
            self.env2 = MarketEnv(
                df=df2,
                features=features,
                n_lag=n_lag,
                num_steps=num_steps,
                market_cls=market_cls,
                is_single_transaction=is_single_transaction,
            )
        else:
            self.env2 = None

        self.prob_to_use_1 = df1.shape[0] / (df1.shape[0] + df2.shape[0])
        self.reset()

    @property
    def state_dim(self) -> int:
        return self.env_to_use.state_dim

    @property
    def action_dim(self) -> int:
        return self.env_to_use.action_dim

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self.env_to_use.action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self.env_to_use.observation_space

    def reset(self) -> np.ndarray:
        if np.random.rand() < self.prob_to_use_1:
            self.env_to_use = self.env1
        else:
            self.env_to_use = self.env2
        return self.env_to_use.reset()

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        return self.env_to_use.step(action_index=action_index)

    def render(self):
        raise NotImplementedError()
