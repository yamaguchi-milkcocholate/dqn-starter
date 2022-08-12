import numpy as np
import pandas as pd
import feather
import gc
import pickle as pkl
from pathlib import Path
from typing import *


def _load_bybit_data(rootdir: Path):
    datadir = rootdir / "data" / "bybit" / "2022-07-24"

    dfs = list()
    for i in range(1, 10):
        dfs.append(feather.read_dataframe(datadir / f"data_{i}.feather"))
    df = pd.concat(dfs).reset_index(drop=True)
    del dfs
    gc.collect()

    df = df[["open_time", "close", "high", "low"]].astype(float).astype(int)

    df.columns = ["timestamp", "price", "max_price", "min_price"]
    df[["buy_price", "sell_price"]] = df[["max_price", "min_price"]]

    return df


def add_features(df):
    df["_diff"] = df["price"].diff()
    df["spread_upper"] = df["max_price"] / df["price"] - 1
    df["spread_lower"] = df["min_price"] / df["price"] - 1

    for minutes in [1, 2]:
        (
            nm_dsharp,
            nm_p,
            nm_pcs,
            nm_area,
            nm_maxval,
            nm_minval,
            nm_maxlen,
            nm_minlen,
            nm_change,
        ) = [
            f"{nm}_{minutes}"
            for nm in [
                "dsharp",
                "_p",
                "_pcs",
                "area",
                "maxval",
                "minval",
                "maxlen",
                "minlen",
                "change",
            ]
        ]

        # 微分Sharp比
        df[nm_dsharp] = df["_diff"].rolling(minutes * 6).mean() / (
            df["_diff"].rolling(minutes * 6).std() + 1.0
        )

        df[nm_p] = find_cross_zero(x=df[nm_dsharp].values)
        df[nm_pcs] = df[nm_p].cumsum()

        _values = np.empty((df.shape[0], 4))
        for i, (price, ds, p, pcs) in enumerate(
            df[["price", nm_dsharp, nm_p, nm_pcs]].values
        ):
            sign = np.sign(ds)
            if pcs == 0:
                mt = {
                    nm_area: np.nan,
                    nm_maxval: np.nan,
                    nm_minval: np.nan,
                    nm_maxlen: np.nan,
                    nm_minlen: np.nan,
                }
            else:
                if p:
                    mt = {
                        nm_area: 0,
                        nm_maxval: -np.inf,
                        nm_minval: np.inf,
                        nm_maxlen: 0,
                        nm_minlen: 0,
                    }
                mt[nm_area] += sign * ds
                if ds > mt[nm_maxval]:
                    mt[nm_maxlen] = 0
                    mt[nm_maxval] = ds
                else:
                    mt[nm_maxlen] += 1
                if ds < mt[nm_minval]:
                    mt[nm_minlen] = 0
                    mt[nm_minval] = ds
                else:
                    mt[nm_minlen] += 1
            _values[i] = np.array(
                [
                    mt[nm_area],
                    max(sign * mt[nm_maxval], sign * mt[nm_minval]),
                    mt[nm_maxlen],
                    mt[nm_minlen],
                ]
            )
        df[[nm_area, nm_change, nm_maxlen, nm_minlen]] = _values

    df = df.dropna().reset_index(drop=True)
    return df


def find_cross_zero(x: np.ndarray) -> np.ndarray:
    x_len = x.shape[0] - 1
    y = np.zeros(x.shape[0]).astype(bool)
    for i in range(x.shape[0] - 1):
        if (x[x_len - i] > 0) and x[x_len - i - 1] <= 0:
            y[x_len - i] = True
        elif (x[x_len - i] < 0) and x[x_len - i - 1] >= 0:
            y[x_len - i] = True
        else:
            y[x_len - i] = False
    return y


def equal_divide_indice(length, num_divide):
    x = np.linspace(0, length - 1, length)
    indice = np.ones_like(x) * -1
    for i, thresh in enumerate(np.linspace(0, length, num_divide + 1)[:-1].astype(int)):
        indice[thresh:] = i
    return indice


def divide_with_pcs(df, num_divide, division):
    df["_eq_fold"] = equal_divide_indice(length=df.shape[0], num_divide=num_divide)
    df["fold"] = np.nan
    for i, (start, end) in enumerate(
        df.groupby("_eq_fold")[division].agg(["min", "max"]).values
    ):
        indice = (start < df[division]) & (df[division] <= end)
        df.loc[indice, "fold"] = i
    df["fold"] = df["fold"].fillna(method="ffill").fillna(method="bfill")
    return df


def add_lag_features(
    df: pd.DataFrame, features: List[str], lags: List[int]
) -> Tuple[pd.DataFrame, List[str]]:
    features_with_lags = [] + features
    for lag in lags:
        lag_features = [f"{f}_lag{lag}" for f in features]
        df[lag_features] = df[features].shift(lag)
        features_with_lags += lag_features
    df = df.dropna().reset_index(drop=True)
    return df, features_with_lags


def load_bybit_data() -> Tuple[pd.DataFrame, List[str]]:
    rootdir = Path(__file__).resolve().parent.parent.parent
    dfcachedir = rootdir / "data" / "cache" / "df"
    dfcachedir.mkdir(parents=True, exist_ok=True)

    dfpath = dfcachedir / "ppo_df.feather"
    featurespath = dfcachedir / "ppo_features.pkl"
    if dfpath.is_file() and featurespath.is_file():
        train = feather.read_dataframe(dfpath)
        features = pkl.load(open(featurespath, "rb"))
    else:
        df = _load_bybit_data(rootdir=rootdir)
        features = [
            "dsharp_1",
            "area_1",
            "change_1",
            "maxlen_1",
            "minlen_1",
            "dsharp_2",
            "area_2",
            "change_2",
            "maxlen_2",
            "minlen_2",
            "spread_upper",
            "spread_lower",
        ]

        dfa = add_features(df=df)
        dfa, features = add_lag_features(df=dfa, features=features, lags=[1, 2, 3])

        train = divide_with_pcs(df=dfa, num_divide=5, division="_pcs_2")
        train = train[train.columns[~train.columns.str.startswith("_")]]

        feather.write_dataframe(train, dfpath)
        pkl.dump(features, open(featurespath, "wb"))

    return train, features
