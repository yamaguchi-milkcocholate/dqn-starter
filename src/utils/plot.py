from pathlib import Path
from matplotlib import pyplot as plt
from typing import *


def plot(
    x: List[float],
    filepath: Path,
    xlabel: str = "",
    ylabel: str = "",
    hist: bool = False,
):
    if hist:
        fig, axes = plt.subplots(1, 2, figsize=(24, 8))
        axes[0].plot(x)
        axes[0].set_ylabel(ylabel)
        axes[0].set_xlabel(xlabel)

        axes[1].hist(x, bins=100)
        axes[0].set_xlabel(ylabel)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(x)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
    fig.savefig(filepath)
    plt.close(fig)
