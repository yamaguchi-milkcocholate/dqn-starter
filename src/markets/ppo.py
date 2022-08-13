import numpy as np
import pandas as pd
from typing import *
from collections import deque


class Market(object):
    def __init__(self, df: pd.DataFrame, features: List[str], is_eval: bool = False):
        self.market_states_cols = features
        self.market_states = df[self.market_states_cols].values
        self.is_eval = is_eval
        self.prices = df[
            ["price", "max_price", "min_price", "buy_price", "sell_price"]
        ].values
        self.i = 0

        self.ob, self.os = False, False
        self.fb, self.fs = False, False
        self.lb, self.ls = None, None
        self.step_from_ob, self.step_from_os = 0, 0
        self.step_from_fb, self.step_from_fs = 0, 0

        self.sum_rtn = 0
        self.rtns = list()
        self.cur_rtn_sum = 0
        self.prices_when_fill = deque()
        self.position_side = None

        self.is_transaction_end = False

    @property
    def num_steps(self) -> int:
        return self.prices.shape[0] - 2

    def step(self, logits: np.ndarray) -> Tuple[float, bool]:
        def sample_categorical(logits: np.ndarray) -> str:
            logits = logits - logits.max()
            probs = np.exp(logits) / np.exp(logits).sum()
            return np.random.choice(["Hold", "Buy", "Sell", "Cancel"], p=probs)

        spread = logits[0]
        if self.is_eval:
            action = ["Hold", "Buy", "Sell", "Cancel"][logits[1:].argmax()]
        else:
            action = sample_categorical(logits=logits[1:])

        price, _, _ = self.prices[self.i, [0, 3, 4]]
        sthprice, bthprice = self.prices[self.i + 1, [1, 2]]

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
        elif action == "Cancel":
            if self.ob:
                self.unset_order_buy()
            elif self.os:
                self.unset_order_sell()

        if self.ob:
            self.step_from_ob += 1
            if self.lb > bthprice:
                self.fb = True
                self.ob = False
                self.step_from_ob = 0
                if self.position_side is None:
                    self.position_side = "Buy"

        if self.os:
            self.step_from_os += 1
            if self.ls < sthprice:
                self.fs = True
                self.os = False
                self.step_from_os = 0
                if self.position_side is None:
                    self.position_side = "Sell"

        if self.fb:
            self.step_from_fb += 1

        if self.fs:
            self.step_from_fs += 1

        if (self.lb is not None) and (self.ls is not None):
            cur_rtn = self.ls / self.lb - 1
        elif self.lb is not None:
            cur_rtn = price / self.lb - 1
        elif self.ls is not None:
            cur_rtn = self.ls / price - 1
        else:
            cur_rtn = 0
        self.cur_rtn_sum += cur_rtn
        sharp_ratio = self.calc_sharp_ratio()

        if self.fb and self.fs:
            self.step_from_fb, self.step_from_fs = 0, 0
            rtn = self.ls / self.lb - 1
            self.fb, self.fs = False, False
            self.sum_rtn += rtn
            self.cur_rtn_sum = 0
            self.prices_when_fill = deque()
            self.position_side = None
            self.is_transaction_end = True
        else:
            rtn = 0

        self.rtns.append(
            {
                "i": self.i,
                "rtn": self.sum_rtn,
                "cur_rtn": cur_rtn,
                "ob": self.ob,
                "os": self.os,
                "fb": self.fb,
                "fs": self.fs,
                "act": action,
                "spread": spread,
            }
        )
        self.i += 1
        return sharp_ratio, self.is_transaction_end

    def calc_sharp_ratio(self) -> float:
        if self.position_side == "Buy":
            return np.log(self.prices[self.i, 0] / self.prices[self.i - 1, 0])
        elif self.position_side == "Sell":
            return np.log(self.prices[self.i - 1, 0] / self.prices[self.i, 0])
        else:
            return np.log(1)

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
        if (self.lb is not None) and (self.ls is not None):
            cur_rtn = self.ls / self.lb - 1
        elif self.lb is not None:
            cur_rtn = self.prices[self.i, 0] / self.lb - 1
        elif self.ls is not None:
            cur_rtn = self.ls / self.prices[self.i, 0] - 1
        else:
            cur_rtn = 0
        trade_state = np.array(
            [
                int(self.ob),
                int(self.os),
                int(self.fb),
                int(self.fs),
                cur_rtn,
                np.log1p(self.step_from_ob),
                np.log1p(self.step_from_os),
                np.log1p(self.step_from_fb),
                np.log1p(self.step_from_fs),
            ]
        )
        market_state = self.market_states[self.i]
        return np.hstack([trade_state, market_state])

    def get_return(self) -> pd.DataFrame:
        return pd.DataFrame(self.rtns)


def random_market(
    df: pd.DataFrame, features: List[str], num_steps: int, is_eval: bool = False
):
    idx = np.random.randint(df.shape[0] - 2 - num_steps)
    return Market(
        df=df.iloc[idx : idx + num_steps].reset_index(drop=True),
        features=features,
        is_eval=is_eval,
    )
