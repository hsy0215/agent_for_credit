"""构建严格使用 T-5 至 T-1 信息的树模型特征。

财务列先沿单股连续时序生成变化量，再统一生成五个滞后日。日期抽样必须发生在本
模块执行之后，以免把抽样后的相邻行误认为相邻交易日。

"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fundamental_columns(factor_columns: list[str]) -> list[str]:
    """返回 `mrq0_` 或 `financial_` 开头的财务因子列。"""
    return [column for column in factor_columns if column.startswith(("mrq0_", "financial_"))]


def dynamicize_fundamentals(frame: pd.DataFrame, factor_columns: list[str]) -> pd.DataFrame:
    """在连续单股时序上生成财务变化量与变化率。

    Args:
        frame: 含股票、交易日与原始因子的连续面板。
        factor_columns: 当前风格的原始因子列。

    Returns:
        新增 `__delta` 与 `__pct_change` 列的面板副本。
    """
    result = frame.sort_values(["order_book_id", "trade_date"]).copy()
    extras: dict[str, pd.Series] = {}
    stock_ids = result["order_book_id"]
    for column in fundamental_columns(factor_columns):
        values = pd.to_numeric(result[column], errors="coerce")
        previous = values.groupby(stock_ids, sort=False).shift(1)
        delta = values - previous
        extras[f"{column}__delta"] = delta
        extras[f"{column}__pct_change"] = delta / previous.abs().where(previous.abs() > 1e-12)
    if not extras:
        return result
    return pd.concat([result, pd.DataFrame(extras, index=result.index)], axis=1)


def source_columns(factor_columns: list[str]) -> list[str]:
    """返回滞后处理前的基础因子与财务衍生列名。"""
    dynamic = [
        f"{column}__{suffix}" for column in fundamental_columns(factor_columns)
        for suffix in ("delta", "pct_change")
    ]
    return [*factor_columns, *dynamic]


def add_previous_window_features(
    frame: pd.DataFrame,
    factor_columns: list[str],
    lags: tuple[int, ...],
) -> pd.DataFrame:
    """为每只股票生成严格早于目标日的 T-n 特征。

    Args:
        frame: 连续的单股日频因子面板。
        factor_columns: 当前风格原始因子列。
        lags: 正整数交易日滞后序列，例如 `(5, 4, 3, 2, 1)`。

    Returns:
        新增历史计数和全部滞后特征的面板。
    """
    result = dynamicize_fundamentals(frame, factor_columns)
    result = result.sort_values(["order_book_id", "trade_date"]).copy()
    groups = result.groupby("order_book_id", sort=False)
    result["__history_count"] = groups.cumcount()
    extras: dict[str, pd.Series] = {}
    for lag in lags:
        for column in source_columns(factor_columns):
            extras[f"{column}__tminus{lag}"] = groups[column].shift(lag)
    return pd.concat([result, pd.DataFrame(extras, index=result.index)], axis=1)


def model_columns(factor_columns: list[str], lags: tuple[int, ...]) -> list[str]:
    """按 notebook 原始顺序返回模型输入列名。"""
    return [f"{column}__tminus{lag}" for lag in lags for column in source_columns(factor_columns)]


def safe_feature_matrix(frame: pd.DataFrame, columns: list[str], clip_abs: float) -> np.ndarray:
    """将面板转为可输入 sklearn 的 float32 数组。

    Args:
        frame: 待转换数据。
        columns: 按固定顺序选取的特征列。
        clip_abs: 有限数值的绝对值裁剪上限。

    Returns:
        非数值与无穷值已置为 NaN 的二维 float32 数组。
    """
    selected = frame.reindex(columns=columns)
    try:
        values = selected.to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError):
        values = selected.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64,
                                                                         copy=True)
    values[~np.isfinite(values)] = np.nan
    np.clip(values, -clip_abs, clip_abs, out=values)
    return values.astype(np.float32, copy=False)
