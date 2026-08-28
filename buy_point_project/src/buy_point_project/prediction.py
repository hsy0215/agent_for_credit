"""构造统一预测 schema，并按数据集和交易日写出 Parquet。

内存中先用宽表收集 100 个模型，落盘时仅展开单个交易日，从而避免长期持有完整
长表。输出固定为 `train|test/YYYY-MM-DD.parquet`。

"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .io import write_parquet

PREDICTION_COLUMNS = ["datetime", "order_book_id", "model_id", "value", "target"]
"""日期分区预测文件的固定字段顺序。"""


def build_prediction_frame(source: pd.DataFrame, values: np.ndarray, model_id: str) -> pd.DataFrame:
    """构造统一的单模型预测 DataFrame。

    Args:
        source: 含 trade_date、order_book_id 和 target 的样本。
        values: 与 source 等长的模型预测值。
        model_id: 风格与时间方案组成的模型名称。

    Returns:
        字段顺序符合 `PREDICTION_COLUMNS` 的预测表。
    """
    result = source[["trade_date", "order_book_id"]].copy()
    result["model_id"] = model_id
    result["value"] = values
    result["target"] = source["target"].to_numpy()
    result = result.rename(columns={"trade_date": "datetime"})
    result["datetime"] = pd.to_datetime(result["datetime"])
    result["order_book_id"] = result["order_book_id"].astype(str)
    return result[PREDICTION_COLUMNS]


@dataclass
class PredictionCollector:
    """按 train/test 收集多个模型的预测序列。"""

    targets: dict[str, pd.Series] = field(default_factory=dict)
    predictions: dict[str, list[pd.Series]] = field(default_factory=lambda: {
        "train": [],
        "test": [],
    })

    def add(self, split: str, frame: pd.DataFrame, model_id: str) -> None:
        """添加一个模型在一个数据集上的预测。"""
        indexed = frame.set_index(["trade_date", "order_book_id"]).sort_index()
        target = indexed["target"].rename("target")
        if split not in self.targets:
            self.targets[split] = target
        else:
            self.targets[split] = self.targets[split].combine_first(target)
        self.predictions[split].append(indexed["value"].rename(model_id))

    def write(self, prediction_dir: Path) -> None:
        """按 split 和日期写出包含所有模型的长表。

        Args:
            prediction_dir: `output/model_prediction/prediction` 目录。
        """
        for split, series in self.predictions.items():
            if not series:
                continue
            wide = pd.concat([self.targets[split], *series], axis=1, join="outer").copy()
            wide = wide.reset_index().sort_values(["trade_date", "order_book_id"])
            model_ids = [column for column in wide.columns if column.startswith("rf_")]
            for trade_date, day in wide.groupby("trade_date", sort=True):
                long = day.melt(
                    id_vars=["trade_date", "order_book_id", "target"],
                    value_vars=model_ids,
                    var_name="model_id",
                    value_name="value",
                ).dropna(subset=["value"])
                long = long.rename(columns={"trade_date": "datetime"})
                long = long[PREDICTION_COLUMNS].sort_values(["order_book_id", "model_id"])
                filename = pd.Timestamp(cast(Any, trade_date)).strftime("%Y-%m-%d.parquet")
                write_parquet(long.reset_index(drop=True), prediction_dir / split / filename)
