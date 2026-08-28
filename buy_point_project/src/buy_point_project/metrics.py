"""复现方案B的单股时序 Rank IC 与回归误差指标。

先沿每只股票的时间轴分别计算 Spearman、RMSE 和 MAE，再对股票等权汇总；同时
保留整个面板的 RMSE 与 MAE。原模型不计算日度横截面 Rank IC。

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _spearman(actual: pd.Series, predicted: pd.Series) -> float:
    """在样本充分且两序列可排序时计算 Spearman 系数。"""
    if len(actual) < 3 or actual.nunique() <= 1 or predicted.nunique() <= 1:
        return np.nan
    value = spearmanr(actual, predicted, nan_policy="omit").statistic
    return float(value) if np.isfinite(value) else np.nan


def prediction_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    """计算单股时序指标与面板误差。

    Args:
        frame: 含 order_book_id、target 和 value 的预测表。

    Returns:
        有效样本数、股票数、时序 Spearman 汇总、股票等权误差和面板误差。
    """
    usable = frame.dropna(subset=["target", "value"])
    if usable.empty:
        return {
            "rows": 0,
            "metric_stocks": 0,
            "ts_spearman_mean": np.nan,
            "ts_spearman_median": np.nan,
            "ts_spearman_std": np.nan,
            "stock_rmse_mean": np.nan,
            "stock_mae_mean": np.nan,
            "panel_rmse": np.nan,
            "panel_mae": np.nan,
        }
    stock_ic: dict[str, float] = {}
    stock_rmse: dict[str, float] = {}
    stock_mae: dict[str, float] = {}
    for stock_id, group in usable.groupby("order_book_id", sort=False):
        stock_key = str(stock_id)
        stock_ic[stock_key] = _spearman(group["target"], group["value"])
        stock_rmse[stock_key] = float(np.sqrt(mean_squared_error(group["target"], group["value"])))
        stock_mae[stock_key] = float(mean_absolute_error(group["target"], group["value"]))
    ic_values = pd.Series(stock_ic, dtype=float).dropna()
    return {
        "rows": len(usable),
        "metric_stocks": int(usable["order_book_id"].nunique()),
        "ts_spearman_mean": float(ic_values.mean()) if len(ic_values) else np.nan,
        "ts_spearman_median": float(ic_values.median()) if len(ic_values) else np.nan,
        "ts_spearman_std": float(ic_values.std()) if len(ic_values) > 1 else np.nan,
        "stock_rmse_mean": float(pd.Series(stock_rmse, dtype=float).mean()),
        "stock_mae_mean": float(pd.Series(stock_mae, dtype=float).mean()),
        "panel_rmse": float(np.sqrt(mean_squared_error(usable["target"], usable["value"]))),
        "panel_mae": float(mean_absolute_error(usable["target"], usable["value"])),
    }
