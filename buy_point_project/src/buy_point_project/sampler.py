"""构建训练日期、标签隔离和季度样本权重。

采样口径来自 `model.ipynb`：每季度确定性等距选日，测试日前隔离 20 个交易日，
并按时间方案选择固定回看期或季度半衰期。

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TimeSpec


def deterministic_quarter_dates(calendar: pd.DatetimeIndex, count: int) -> pd.DatetimeIndex:
    """每季度确定性等距选择指定数量的交易日。

    Args:
        calendar: 完整升序交易日历。
        count: 每季度最多选取的日期数。

    Returns:
        去重且升序的训练日期索引。
    """
    frame = pd.DataFrame({"trade_date": calendar})
    selected: list[np.datetime64] = []
    for _, group in frame.groupby(frame["trade_date"].dt.to_period("Q"), sort=True):
        dates = group["trade_date"].drop_duplicates().sort_values().to_numpy()
        if len(dates) <= count:
            selected.extend(dates)
        else:
            positions = np.linspace(0, len(dates) - 1, count, dtype=int)
            selected.extend(dates[positions])
    return pd.DatetimeIndex(selected).drop_duplicates().sort_values()


def purged_train_end(calendar: pd.DatetimeIndex, test_start: pd.Timestamp,
                     purge_days: int) -> pd.Timestamp:
    """计算标签未来窗口不进入测试期的最后训练交易日。

    Raises:
        ValueError: 测试期前的交易日不足以完成隔离。
    """
    first_test_position = int(calendar.searchsorted(test_start, side="left"))
    train_end_position = first_test_position - purge_days - 1
    if train_end_position < 0:
        raise ValueError("测试期之前没有足够的标签隔离数据")
    return calendar[train_end_position]


def time_spec_start(data_start: pd.Timestamp, reference_quarter: pd.Period,
                    spec: TimeSpec) -> pd.Timestamp:
    """根据时间方案返回候选训练集起始日。"""
    if spec.lookback_quarters is None:
        return data_start
    return max(data_start, (reference_quarter - spec.lookback_quarters).start_time.normalize())


def quarter_balanced_weights(frame: pd.DataFrame, reference_quarter: pd.Period,
                             half_life: int | None) -> np.ndarray:
    """生成季度等权或按季度半衰期衰减的样本权重。

    Returns:
        均值为 1 的逐行浮点权重。
    """
    quarters = frame["trade_date"].dt.to_period("Q")
    rows_per_quarter = quarters.value_counts()
    latest_train_quarter = reference_quarter - 1
    ages = np.fromiter(
        (latest_train_quarter.ordinal - quarter.ordinal for quarter in quarters),
        dtype=float,
        count=len(quarters),
    )
    quarter_weights = (np.ones(len(frame), dtype=float) if half_life is None else np.power(
        2.0, -ages / half_life))
    weights = quarter_weights / quarters.map(rows_per_quarter).to_numpy(dtype=float)
    return weights * (len(weights) / weights.sum())
