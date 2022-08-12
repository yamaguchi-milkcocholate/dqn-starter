from pathlib import Path
from matplotlib import pyplot as plt
from typing import *


def plot(x: List[float], filepath: Path, xlabel: str = "", ylabel: str = ""):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(x)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    fig.savefig(filepath)
    plt.close(fig)
