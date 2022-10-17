import sys
import json
import torch
from pathlib import Path
from typing import *

rootdir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(rootdir))


import src.utils.data as data


def main():
    exptdir = Path(__file__).resolve().parent

    config = json.load(open(exptdir / "config.json", "r"))
    train_params = config["train_params"]

    df, features = data.load_bybit_data(
        num_divide=train_params["NUM_DEVIDE"],
        interval=train_params["MINUTES"],
        use_cache=False,
        ta_config_file="config_mine.json",
    )


if __name__ == "__main__":
    main()
