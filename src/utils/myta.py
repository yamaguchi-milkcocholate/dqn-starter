import numpy as np
import pandas as pd


class ROC:
    def __init__(self, close: pd.Series, window: int, fillna: bool = True):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def roc(self) -> pd.Series:
        series = self._df[f"roc{self._window}"]
        return self._check_na(series=series)

    def roc_diff(self) -> pd.Series:
        series = self._df[f"roc1"] - self._df[f"roc{self._window}"]
        return self._check_na(series=series)

    def roc_range(self) -> pd.Series:
        df = self._df[self._roc_cols]
        series = (self._df.max(axis=1) - self._df.min(axis=1)) / np.abs(df).mean(axis=1)
        return self._check_na(series=series)

    def roc_mean(self) -> pd.Series:
        df = self._df[self._roc_cols]
        series = df.mean(axis=1)
        return self._check_na(series=series)

    def roc_mean_diff(self) -> pd.Series:
        df = self._df[self._roc_cols]
        series = self._df[f"roc1"] - df.mean(axis=1)
        return self._check_na(series=series)

    def _run(self):
        self._df = pd.DataFrame({"shift0": self._close})
        self._shift_cols = []
        self._roc_cols = []
        for i in range(1, self._window + 1):
            self._df.loc[:, f"shift{i}"] = self._close.shift(i).values
            self._df.loc[:, f"roc{i}"] = (
                self._close.values / self._close.shift(i).values - 1
            )
            self._shift_cols.append(f"shift{i}")
            self._roc_cols.append(f"roc{i}")

    def _check_na(self, series: pd.Series) -> pd.Series:
        if self._fillna:
            series_out = series.copy()
            series_out = series_out.replace([np.inf, -np.inf], np.nan)
            series = series_out.fillna(method="ffill").fillna(method="bfill")
        return series
