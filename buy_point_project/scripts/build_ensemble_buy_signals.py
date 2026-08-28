"""从方案B的100模型预测中构建优选8模型均值买卖事件。

8个模型是根据120日90%买点的未来50日 Calmar 排序，并结合预测曲线走势复核后
固定的名单。本脚本执行严格对齐、算术平均、120日滚动百分位和阈值划分，最终按
交易日写出仅含买卖事件的四字段 Parquet，不在运行时重新选择模型。

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from buy_point_project.config import Config
from buy_point_project.io import write_parquet

SELECTED_MODEL_IDS = (
    "rf_volatility_risk_T02",
    "rf_correlation_size_exposure_T10",
    "rf_momentum_T02",
    "rf_reversal_T04",
    "rf_technical_state_T01",
    "rf_growth_expectation_T01",
    "rf_liquidity_turnover_T05",
    "rf_microstructure_orderflow_T05",
)
"""Calmar 排序与预测曲线走势共同筛选的8个方案B模型。"""

EVENT_COLUMNS = ["datetime", "order_book_id", "model_id", "signal"]


@dataclass(frozen=True)
class SelectedEnsembleArtifacts:
    """保存优选8模型组合的内部连续结果和最终四字段事件。"""

    summary: pd.DataFrame
    events: pd.DataFrame
    output_paths: tuple[Path, ...] = ()


def _validate_parameters(
    model_ids: tuple[str, ...],
    window: int,
    buy_percentile: float,
    sell_percentile: float,
) -> None:
    """校验模型名单、窗口和双向阈值。"""
    if not model_ids or any(not model_id.strip() for model_id in model_ids):
        raise ValueError("model_ids 不能为空")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("model_ids 不能重复")
    if window <= 0:
        raise ValueError("window 必须为正整数")
    if not 0.0 <= sell_percentile < buy_percentile <= 1.0:
        raise ValueError("百分位必须满足 0 <= sell_percentile < buy_percentile <= 1")


def load_selected_predictions(
    prediction_dir: Path,
    model_ids: tuple[str, ...] = SELECTED_MODEL_IDS,
    *,
    expected_model_count: int = 100,
) -> pd.DataFrame:
    """从日期分区读取100模型预测，并严格提取完整的8模型股票日面板。

    Args:
        prediction_dir: ``prediction/test`` 日期分区目录。
        model_ids: 需要组合的模型名单。
        expected_model_count: 输入预测中预期存在的模型总数。

    Returns:
        每个股票交易日都完整包含指定模型的长表。
    """
    if expected_model_count <= 0:
        raise ValueError("expected_model_count 必须为正整数")
    if not model_ids or any(not model_id.strip() for model_id in model_ids):
        raise ValueError("model_ids 不能为空")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("model_ids 不能重复")
    if len(model_ids) > expected_model_count:
        raise ValueError("优选模型数不能超过输入模型总数")
    paths = sorted(prediction_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"没有日期分区预测: {prediction_dir}")
    parts = []
    available_model_ids: set[str] = set()
    for path in paths:
        frame = pd.read_parquet(
            path,
            columns=["datetime", "order_book_id", "model_id", "value"],
        )
        frame["model_id"] = frame["model_id"].astype(str)
        available_model_ids.update(frame["model_id"].unique())
        selected = frame.loc[frame["model_id"].isin(model_ids)].copy()
        if not selected.empty:
            parts.append(selected)
    actual_model_count = len(available_model_ids)
    if actual_model_count != expected_model_count:
        raise ValueError(f"预测应包含 {expected_model_count} 个模型，实际 {actual_model_count} 个")
    missing_models = set(model_ids).difference(available_model_ids)
    if missing_models:
        raise KeyError(f"预测缺少优选模型: {sorted(missing_models)}")
    if not parts:
        raise ValueError("优选模型没有预测记录")
    predictions = pd.concat(parts, ignore_index=True)
    predictions["datetime"] = pd.to_datetime(predictions["datetime"]).dt.normalize()
    predictions["order_book_id"] = predictions["order_book_id"].astype(str)
    predictions["value"] = pd.to_numeric(predictions["value"], errors="coerce")
    if predictions["value"].isna().any():
        raise ValueError("优选模型预测包含空值或非数值")
    keys = ["datetime", "order_book_id", "model_id"]
    if predictions.duplicated(keys).any():
        raise ValueError("优选模型预测存在重复 datetime × order_book_id × model_id")
    source_counts = predictions.groupby(["datetime", "order_book_id"],
                                        sort=False)["model_id"].nunique()
    incomplete = int(source_counts.ne(len(model_ids)).sum())
    if incomplete:
        raise ValueError(f"有 {incomplete} 个股票交易日没有集齐 {len(model_ids)} 个模型")
    return predictions.sort_values(keys, kind="stable").reset_index(drop=True)


def build_selected_ensemble(
    predictions: pd.DataFrame,
    *,
    model_ids: tuple[str, ...] = SELECTED_MODEL_IDS,
    window: int = 120,
    buy_percentile: float = 0.80,
    sell_percentile: float = 0.10,
) -> SelectedEnsembleArtifacts:
    """计算优选8模型均值、滚动百分位，并展开四字段双向事件。

    百分位按预测值从小到大计算；当前预测位于最近120条的80%以上时买入，位于
    10%以下时卖出。最终事件再展开8个 ``model_id``，明确记录全部来源模型。
    """
    _validate_parameters(model_ids, window, buy_percentile, sell_percentile)
    summary = (predictions.groupby(["datetime", "order_book_id"], as_index=False,
                                   sort=False).agg(buy_prediction=("value", "mean"),
                                                   source_model_count=("model_id", "nunique")))
    summary = summary.sort_values(["order_book_id", "datetime"], kind="stable")
    summary["percent"] = summary.groupby("order_book_id", sort=False)["buy_prediction"].transform(
        lambda values: values.rolling(window, min_periods=1).rank(method="average", pct=True))
    summary["buy_signal"] = summary["percent"].ge(buy_percentile)
    summary["sell_signal"] = summary["percent"].le(sell_percentile)
    summary["datetime"] = summary["datetime"].dt.normalize().add(pd.Timedelta(hours=9, minutes=30))
    summary = summary.sort_values(["datetime", "order_book_id"],
                                  kind="stable").reset_index(drop=True)

    event_summary = summary.loc[summary["buy_signal"] | summary["sell_signal"],
                                ["datetime", "order_book_id", "buy_signal"]].copy()
    event_summary["signal"] = event_summary["buy_signal"].map({True: 1, False: 2}).astype("int8")
    events = event_summary.loc[event_summary.index.repeat(len(model_ids)),
                               ["datetime", "order_book_id", "signal"]].reset_index(drop=True)
    events.insert(2, "model_id", list(model_ids) * len(event_summary))
    events = events[EVENT_COLUMNS].sort_values(["datetime", "order_book_id", "model_id"],
                                               kind="stable").reset_index(drop=True)
    return SelectedEnsembleArtifacts(summary=summary, events=events)


def build_ensemble_buy_sell_signals(
    prediction_dir: Path,
    output_dir: Path,
    *,
    model_ids: tuple[str, ...] = SELECTED_MODEL_IDS,
    expected_model_count: int = 100,
    window: int = 120,
    buy_percentile: float = 0.80,
    sell_percentile: float = 0.10,
) -> SelectedEnsembleArtifacts:
    """读取100模型预测并按 ``YYYY-MM-DD.parquet`` 写出四字段事件。"""
    predictions = load_selected_predictions(
        prediction_dir,
        model_ids,
        expected_model_count=expected_model_count,
    )
    artifacts = build_selected_ensemble(
        predictions,
        model_ids=model_ids,
        window=window,
        buy_percentile=buy_percentile,
        sell_percentile=sell_percentile,
    )
    output_paths = []
    for trade_date in artifacts.summary["datetime"].drop_duplicates().sort_values():
        filename = pd.Timestamp(trade_date).strftime("%Y-%m-%d.parquet")
        path = output_dir / filename
        day = artifacts.events.loc[artifacts.events["datetime"].eq(trade_date)]
        write_parquet(day[EVENT_COLUMNS].reset_index(drop=True), path)
        output_paths.append(path)
    return SelectedEnsembleArtifacts(
        summary=artifacts.summary,
        events=artifacts.events,
        output_paths=tuple(output_paths),
    )


def main() -> None:
    """解析配置并生成优选8模型组合的买入与卖出信号。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--buy-percentile", type=float, default=0.80)
    parser.add_argument("--sell-percentile", type=float, default=0.10)
    parser.add_argument("--expected-model-count", type=int, default=100)
    parser.add_argument("--model-id", action="append", dest="model_ids")
    args = parser.parse_args()
    config = Config.load()
    model_ids = tuple(args.model_ids) if args.model_ids else SELECTED_MODEL_IDS
    artifacts = build_ensemble_buy_sell_signals(
        config.prediction_dir / "test",
        config.ensemble_8_models_dir,
        model_ids=model_ids,
        expected_model_count=args.expected_model_count,
        window=args.window,
        buy_percentile=args.buy_percentile,
        sell_percentile=args.sell_percentile,
    )
    print({
        "source_models": list(model_ids),
        "ensemble_rows": len(artifacts.summary),
        "signal_rows": len(artifacts.events),
        "buy_signal_rows": int(artifacts.events["signal"].eq(1).sum()),
        "sell_signal_rows": int(artifacts.events["signal"].eq(2).sum()),
        "date_partitions": len(artifacts.output_paths),
        "output": str(config.ensemble_8_models_dir),
    })


if __name__ == "__main__":
    main()
