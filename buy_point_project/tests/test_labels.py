"""验证买入和卖出标签的核心数学口径。

"""

import numpy as np
import pandas as pd
from scripts.generate_labels import build_adjusted_close, generate_label_partitions

from buy_point_project.label import (
    compute_buy_mild_time_penalty,
    compute_sell_label,
    generate_labels,
)


def test_buy_mild_time_penalty_uses_future_20_days() -> None:
    """递增价格首日标签应等于 20 日最大收益除以时间惩罚。"""
    close = np.arange(1.0, 26.0)
    label = compute_buy_mild_time_penalty(close)
    expected = 20.0 / (1.0 + 0.1 * np.log(20.0))

    assert np.isclose(label[0], expected)
    assert np.isnan(label[-20:]).all()


def test_sell_label_and_generated_schema() -> None:
    """显著冲高回落应产生有限卖出分数且生成表 schema 固定。"""
    close = np.array([10.0, 10.2, 10.5, 11.0, 12.0, 13.0, 12.5, 11.5, 10.5, 10.0] * 4)
    sell_label = compute_sell_label(close)
    source = pd.DataFrame({
        "order_book_id": ["000001.XSHE"] * len(close),
        "trade_date": pd.date_range("2024-01-01", periods=len(close)),
        "adjusted_close": close,
    })
    result = generate_labels(source)

    assert np.isfinite(sell_label).all()
    assert sell_label.min() >= 0.0
    assert sell_label.max() <= 1.0
    assert sell_label.max() > 0.0
    assert result.columns.tolist() == [
        "order_book_id", "trade_date", "adjusted_close", "buy_mild_time_penalty", "sell_label"
    ]


def test_label_script_writes_date_partitions_with_delivery_schema(tmp_path) -> None:
    """一键生成结果应使用四字段 schema 和带连字符的日期文件名。"""
    close = np.linspace(10.0, 14.0, 25)
    source = pd.DataFrame({
        "order_book_id": ["000001.XSHE"] * len(close),
        "trade_date": pd.date_range("2024-01-01", periods=len(close)),
        "adjusted_close": close,
    })

    labels, paths = generate_label_partitions(
        source,
        tmp_path / "label",
    )

    assert labels.columns.tolist() == ["datetime", "order_book_id", "buy_label", "sell_label"]
    assert paths[0].name == "2024-01-01.parquet"
    assert paths[-1].name == "2024-01-25.parquet"
    saved = pd.read_parquet(paths[0])
    assert saved.columns.tolist() == ["datetime", "order_book_id", "buy_label", "sell_label"]


def test_build_adjusted_close_uses_raw_daily_prices() -> None:
    """生成脚本应从原始日行情构造连续复权价格，不读取旧标签文件。"""
    source = pd.DataFrame({
        "order_book_id": ["000001.XSHE"] * 3,
        "trade_date": pd.date_range("2024-01-01", periods=3),
        "close": [10.0, 5.0, 6.0],
        "prev_close": [9.8, 5.0, 5.0],
    })

    result = build_adjusted_close(source)

    assert np.allclose(result["adjusted_close"], [10.0, 10.0, 12.0])
