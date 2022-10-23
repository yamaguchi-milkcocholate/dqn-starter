from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
import feather
import gc
import pickle as pkl
import json
from pprint import pprint
from pathlib import Path
from typing import *
from sklearn.preprocessing import RobustScaler
from ta import add_all_ta_features
import src.utils.tainvoke as tainvoke

STATE_RANGE_MAX, STATE_RANGE_MIN = 10, -10


def _load_bybit_data(rootdir: Path, interval: str):
    datadir = rootdir / "data" / "bybit" / interval

    dfs = list()
    for i in range(1, 10):
        dfs.append(feather.read_dataframe(datadir / f"data_{i}.feather"))
    df = pd.concat(dfs).reset_index(drop=True)
    del dfs
    gc.collect()

    df = df[["open_time", "close", "high", "low", "volume"]].astype(float).astype(int)

    df.columns = ["timestamp", "price", "max_price", "min_price", "volume"]
    df[["buy_price", "sell_price"]] = df[["max_price", "min_price"]]

    return df


def add_features(_df):
    df = _df.copy()
    df["spread_upper"] = df["max_price"] / df["price"] - 1
    df["spread_lower"] = df["min_price"] / df["price"] - 1

    df = add_all_ta_features(
        df,
        open="price_1",
        high="max_price",
        low="min_price",
        close="price",
        volume="volume",
        fillna=True,
    )
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


def load_bybit_data(
    num_divide: int,
    interval: str,
    use_cache: bool = True,
    ta_config_file: str = "config.json",
) -> Tuple[pd.DataFrame, List[str]]:
    if interval not in ("1min", "5min"):
        raise Exception()

    rootdir = Path(__file__).resolve().parent.parent.parent
    tadir = rootdir / "data" / "ta"
    dfcachedir = rootdir / "data" / "cache" / "df"
    dfcachedir.mkdir(parents=True, exist_ok=True)

    dfpath = dfcachedir / f"ppo_df_{interval}.feather"
    featurespath = dfcachedir / f"ppo_features_{interval}.csv"
    scalerpath = dfcachedir / f"ppo_scaler_{interval}.pkl"
    if dfpath.is_file() and featurespath.is_file() and use_cache:
        dfa = feather.read_dataframe(dfpath)
        df_features = pd.read_csv(featurespath)
        features = df_features["feature_name"].values.tolist()
    else:
        df = _load_bybit_data(rootdir=rootdir, interval=interval)
        df["open"] = df["price"].shift(1)
        df = df.dropna().reset_index(drop=True)
        df = df.rename(
            columns={"price": "close", "max_price": "high", "min_price": "low"}
        )

        ta_config = json.load(open(tadir / ta_config_file, "r"))
        dfa = add_ta_features(df=df, ta_config=ta_config)
        features = [
            col for col in set(dfa.columns) - set(df.columns) if not col.startswith("_")
        ]
        df_features = pd.DataFrame(features, columns=["feature_name"])

        scaler = RobustScaler(quantile_range=(5, 95))
        dfa[features] = scaler.fit_transform(dfa[features])
        dfa[features] = np.clip(dfa[features], STATE_RANGE_MIN, STATE_RANGE_MAX)

        dfa["fold"] = equal_divide_indice(length=dfa.shape[0], num_divide=num_divide)
        dfa = dfa[dfa.columns[~dfa.columns.str.startswith("_")]]

        feather.write_dataframe(dfa, dfpath)
        df_features.to_csv(featurespath, index=False)
        pkl.dump(scaler, open(scalerpath, "wb"))

    pprint(features)

    return dfa, features


def arange_1week(
    df: pd.DataFrame, evalday: int, interval: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if interval == "1min":
        numrows = 288 * 1
    elif interval == "5min":
        numrows = 288 * 5
    else:
        raise ValueError()

    df_train, df_eval = (
        df.iloc[-(numrows * (evalday + 8)) : -(numrows * evalday)].reset_index(
            drop=True
        ),
        df.iloc[-(numrows * evalday) :].reset_index(drop=True),
    )
    return df_train, df_eval


def make_args(df: pd.DataFrame, ohlcv: str) -> Dict[str, pd.DataFrame]:
    args = {}
    for k in ohlcv:
        if k == "o":
            args["open"] = df["open"]
        elif k == "h":
            args["high"] = df["high"]
        elif k == "l":
            args["low"] = df["low"]
        elif k == "c":
            args["close"] = df["close"]
        elif k == "v":
            args["volume"] = df["volume"]
    return args


def add_ta_features(df: pd.DataFrame, ta_config: Dict[str, Any]) -> pd.DataFrame:
    values = {}

    for cls_name, cls_config in ta_config.items():
        print(cls_name, "・・・")

        params, method, ohlcv = (
            cls_config["params"],
            cls_config["method"],
            cls_config["ohlcv"],
        )
        args = make_args(df=df, ohlcv=ohlcv)

        for i, _param in enumerate(params):
            ta_instance = getattr(tainvoke, cls_name)(**args, **_param)

            for method_name, prefix in method.items():
                print(" " * 4, f"- {prefix} {i + 1}  ", end="")
                _values = getattr(ta_instance, method_name)()
                values[f"{cls_name}_{prefix}_{i + 1}"] = _values

                if np.isinf(_values).any():
                    print("Warning [infinity]")
                else:
                    print("Done")
        print(" Done.")

    dfa = pd.DataFrame(values)
    dfa = dfa.replace([np.inf, -np.inf], np.nan)
    dfa = dfa.fillna(dfa.median())
    dfa = pd.concat([df, dfa], axis=1).reset_index(drop=True)
    return dfa
