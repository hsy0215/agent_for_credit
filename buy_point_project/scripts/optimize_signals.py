"""从连续模型预测生成逐模型及集成均值的滚动百分位买卖信号。

该文件直接放在 ``scripts`` 下，属于训练完成后的显式信号优化步骤。逐模型百分位
按照 ``model_id × order_book_id`` 计算；集成百分位先对同日同股的全部模型预测
取均值，再按照 ``order_book_id`` 计算。两种计算都只使用截至当天的历史值。

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from buy_point_project.config import Config
from buy_point_project.io import write_parquet


@dataclass(frozen=True)
class SignalArtifacts:
    """保存逐模型和集成信号的六张结果表。"""

    model_signals: pd.DataFrame
    model_buy_points: pd.DataFrame
    model_sell_points: pd.DataFrame
    ensemble_signals: pd.DataFrame
    ensemble_buy_points: pd.DataFrame
    ensemble_sell_points: pd.DataFrame


def _signal_suffix(window: int, buy_percentile: float, sell_percentile: float) -> str:
    """根据窗口和阈值生成稳定的结果文件后缀。"""
    return (f"rolling{window}_buyq{int(round(buy_percentile * 100)):02d}"
            f"_sellq{int(round(sell_percentile * 100)):02d}")


def _validate_parameters(window: int, buy_percentile: float, sell_percentile: float) -> None:
    """校验窗口和双向百分位阈值。"""
    if window <= 0:
        raise ValueError("window 必须为正整数")
    if not 0.0 <= sell_percentile < buy_percentile <= 1.0:
        raise ValueError("百分位必须满足 0 <= sell_percentile < buy_percentile <= 1")


def _load_predictions(prediction_dir: Path) -> pd.DataFrame:
    """读取所有测试日期分区并校验逐模型预测 schema。"""
    paths = sorted(prediction_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"没有日期分区预测: {prediction_dir}")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    required = {"datetime", "order_book_id", "model_id", "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"预测文件缺少字段: {sorted(missing)}")
    frame = frame[["datetime", "order_book_id", "model_id", "value"]].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    frame["order_book_id"] = frame["order_book_id"].astype(str)
    frame["model_id"] = frame["model_id"].astype(str)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if frame.duplicated(["datetime", "order_book_id", "model_id"]).any():
        raise ValueError("预测存在重复 datetime × order_book_id × model_id")
    return frame.dropna(subset=["value"])


def _add_rolling_signals(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    window: int,
    buy_percentile: float,
    sell_percentile: float,
) -> pd.DataFrame:
    """为连续预测增加滚动百分位和布尔买卖信号。"""
    ordered = frame.sort_values([*group_columns, "datetime"], kind="stable").copy()
    grouped = ordered.groupby(group_columns, sort=False)["value"]
    ordered["percent"] = grouped.transform(
        lambda values: values.rolling(window, min_periods=1).rank(method="average", pct=True))
    ordered["buy_signal"] = ordered["percent"].ge(buy_percentile)
    ordered["sell_signal"] = ordered["percent"].le(sell_percentile)
    return ordered.sort_values(["datetime", "order_book_id", *group_columns],
                               kind="stable").reset_index(drop=True)


def _event_points(signals: pd.DataFrame, signal_column: str) -> pd.DataFrame:
    """从全量信号中提取统一三字段事件表。"""
    columns = ["datetime", "order_book_id", "model_id"]
    return signals.loc[signals[signal_column], columns].reset_index(drop=True)


def optimize_signals(
    prediction_dir: Path,
    output_dir: Path,
    *,
    window: int = 120,
    buy_percentile: float = 0.80,
    sell_percentile: float = 0.10,
) -> SignalArtifacts:
    """计算并写出逐模型与集成均值的滚动百分位买卖信号。

    Args:
        prediction_dir: ``prediction/test`` 日期分区目录。
        output_dir: 信号结果目录。
        window: 每只股票使用的滚动预测观测数。
        buy_percentile: 百分位大于等于该值时产生买入信号。
        sell_percentile: 百分位小于等于该值时产生卖出信号。

    Returns:
        逐模型和集成均值对应的全量信号、买点、卖点。
    """
    _validate_parameters(window, buy_percentile, sell_percentile)
    predictions = _load_predictions(prediction_dir)
    model_signals = _add_rolling_signals(
        predictions,
        group_columns=["model_id", "order_book_id"],
        window=window,
        buy_percentile=buy_percentile,
        sell_percentile=sell_percentile,
    )[[
        "datetime",
        "order_book_id",
        "model_id",
        "value",
        "percent",
        "buy_signal",
        "sell_signal",
    ]]
    model_buy_points = _event_points(model_signals, "buy_signal")
    model_sell_points = _event_points(model_signals, "sell_signal")

    ensemble = (predictions.groupby(["datetime", "order_book_id"], as_index=False,
                                    sort=False).agg(value=("value", "mean"),
                                                    model_count=("model_id", "nunique")))
    ensemble["model_id"] = "ensemble_mean"
    ensemble_signals = _add_rolling_signals(
        ensemble,
        group_columns=["order_book_id"],
        window=window,
        buy_percentile=buy_percentile,
        sell_percentile=sell_percentile,
    )
    ensemble_signals["buy_prediction"] = ensemble_signals["value"]
    ensemble_signals = ensemble_signals[[
        "datetime",
        "order_book_id",
        "model_id",
        "model_count",
        "value",
        "buy_prediction",
        "percent",
        "buy_signal",
        "sell_signal",
    ]]
    ensemble_buy_points = _event_points(ensemble_signals, "buy_signal")
    ensemble_sell_points = _event_points(ensemble_signals, "sell_signal")

    suffix = _signal_suffix(window, buy_percentile, sell_percentile)
    write_parquet(model_signals, output_dir / f"signals_{suffix}.parquet")
    write_parquet(model_buy_points, output_dir / f"buy_points_{suffix}.parquet")
    write_parquet(model_sell_points, output_dir / f"sell_points_{suffix}.parquet")
    write_parquet(ensemble_signals, output_dir / f"ensemble_signals_{suffix}.parquet")
    write_parquet(ensemble_buy_points, output_dir / f"ensemble_buy_points_{suffix}.parquet")
    write_parquet(ensemble_sell_points, output_dir / f"ensemble_sell_points_{suffix}.parquet")
    return SignalArtifacts(
        model_signals=model_signals,
        model_buy_points=model_buy_points,
        model_sell_points=model_sell_points,
        ensemble_signals=ensemble_signals,
        ensemble_buy_points=ensemble_buy_points,
        ensemble_sell_points=ensemble_sell_points,
    )


def main() -> None:
    """解析信号参数并写出逐模型及集成买卖信号。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--buy-percentile", type=float, default=0.80)
    parser.add_argument("--sell-percentile", type=float, default=0.10)
    args = parser.parse_args()
    config = Config.load()
    artifacts = optimize_signals(
        config.prediction_dir / "test",
        config.signals_dir,
        window=args.window,
        buy_percentile=args.buy_percentile,
        sell_percentile=args.sell_percentile,
    )
    print({
        "model_signal_rows": len(artifacts.model_signals),
        "model_buy_points": len(artifacts.model_buy_points),
        "model_sell_points": len(artifacts.model_sell_points),
        "ensemble_buy_points": len(artifacts.ensemble_buy_points),
        "ensemble_sell_points": len(artifacts.ensemble_sell_points),
        "output": str(config.signals_dir),
    })


if __name__ == "__main__":
    main()
