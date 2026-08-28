"""从配置中的复权收盘价一键生成按交易日分区的买入与卖出标签。

"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from buy_point_project.config import Config
from buy_point_project.io import write_parquet
from buy_point_project.label import generate_labels

OUTPUT_COLUMNS = ["datetime", "order_book_id", "buy_label", "sell_label"]


def _format_duration(seconds: float) -> str:
    """把秒数转换成便于终端阅读的时分秒。"""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_adjusted_close(frame: pd.DataFrame) -> pd.DataFrame:
    """从原始 close/prev_close 日行情构造连续复权收盘价标签输入。"""
    required = {"order_book_id", "trade_date", "close", "prev_close"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"日行情缺少字段: {sorted(missing)}")
    source = frame[["order_book_id", "trade_date", "close", "prev_close"]].copy()
    source["order_book_id"] = source["order_book_id"].astype(str)
    source["trade_date"] = pd.to_datetime(source["trade_date"]).dt.normalize()
    source["close"] = pd.to_numeric(source["close"], errors="coerce")
    source["prev_close"] = pd.to_numeric(source["prev_close"], errors="coerce")
    source = source.sort_values(["order_book_id", "trade_date"], kind="stable")
    if source.duplicated(["order_book_id", "trade_date"]).any():
        raise ValueError("日行情存在重复 order_book_id × trade_date")
    previous_close = source.groupby("order_book_id", sort=False)["close"].shift(1)
    action_ratio = previous_close.div(source["prev_close"])
    action_ratio = action_ratio.where(action_ratio.gt(0) & np.isfinite(action_ratio), 1.0)
    adjustment_factor = action_ratio.groupby(source["order_book_id"], sort=False).cumprod()
    source["adjusted_close"] = source["close"] * adjustment_factor
    return source[["order_book_id", "trade_date", "adjusted_close"]]


def generate_label_partitions(source: pd.DataFrame,
                              output_dir: Path,
                              show_progress: bool = False) -> tuple[pd.DataFrame, list[Path]]:
    """计算买卖标签，并以 ``YYYY-MM-DD.parquet`` 按日期保存。

    Args:
        source: 含 order_book_id、trade_date、adjusted_close 的日频价格。
        output_dir: 标签日期分区的输出目录。
        show_progress: 是否使用 tqdm 显示逐股票标签计算进度条。

    Returns:
        固定四字段的完整标签表和本次写出的日期分区路径。
    """
    stock_count = source["order_book_id"].nunique()
    with tqdm(
            total=stock_count,
            desc="计算标签",
            unit="只股票",
            dynamic_ncols=True,
            disable=not show_progress,
    ) as calculation_bar:

        def update_calculation(completed: int, total: int, current_item: str) -> None:
            """把逐股票计算进度同步到 tqdm。"""
            calculation_bar.total = total
            calculation_bar.set_postfix_str(f"当前={current_item}", refresh=False)
            calculation_bar.update(completed - calculation_bar.n)

        labels = generate_labels(
            source,
            progress_callback=update_calculation if show_progress else None,
        ).rename(columns={
            "trade_date": "datetime",
            "buy_mild_time_penalty": "buy_label",
        })
    labels = labels[OUTPUT_COLUMNS].copy()
    labels["datetime"] = pd.to_datetime(labels["datetime"]).dt.normalize()
    labels["order_book_id"] = labels["order_book_id"].astype(str)
    labels = labels.sort_values(["datetime", "order_book_id"], kind="stable").reset_index(drop=True)
    paths = []
    grouped_days = labels.groupby("datetime", sort=True)
    for trade_date, day in grouped_days:
        filename = pd.Timestamp(cast(Any, trade_date)).strftime("%Y-%m-%d.parquet")
        path = output_dir / filename
        write_parquet(day[OUTPUT_COLUMNS].reset_index(drop=True), path)
        paths.append(path)
    return labels, paths


def main() -> None:
    """读取配置中的标签源价格并按日期写出两个目标标签。"""
    config = Config.load()
    print(f"[1/4 读取行情] {config.input.price_path}", flush=True)
    read_started_at = time.perf_counter()
    raw_prices = pd.read_parquet(
        config.input.price_path,
        columns=["order_book_id", "trade_date", "close", "prev_close"],
    )
    print(
        f"[1/4 读取行情] 完成 rows={len(raw_prices):,} "
        f"耗时={_format_duration(time.perf_counter() - read_started_at)}",
        flush=True,
    )
    print("[2/4 构造复权价格] 开始", flush=True)
    adjustment_started_at = time.perf_counter()
    source = build_adjusted_close(raw_prices)
    print(
        f"[2/4 构造复权价格] 完成 stocks={source['order_book_id'].nunique():,} "
        f"耗时={_format_duration(time.perf_counter() - adjustment_started_at)}",
        flush=True,
    )
    print("[3/4 计算标签] 开始", flush=True)
    labels, paths = generate_label_partitions(
        source,
        config.label_dir,
        show_progress=True,
    )
    print({
        "rows": len(labels),
        "stocks": labels["order_book_id"].nunique(),
        "date_partitions": len(paths),
        "output": str(config.label_dir),
    })


if __name__ == "__main__":
    main()
