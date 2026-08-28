"""验证滚动百分位买入与卖出信号按股票和模型独立计算。

"""

import pandas as pd
from scripts.optimize_signals import optimize_signals

from buy_point_project.io import write_parquet


def test_optimize_signals_builds_buy_and_sell_events(tmp_path) -> None:
    """递增序列达到前 20%，递减序列第十日达到后 10%。"""
    prediction_dir = tmp_path / "prediction/test"
    for index in range(1, 11):
        day = pd.DataFrame({
            "datetime": [pd.Timestamp(2024, 1, index)] * 2,
            "order_book_id": ["BUY.XSHG", "SELL.XSHG"],
            "model_id": ["rf_momentum_T01"] * 2,
            "value": [float(index), float(11 - index)],
            "target": [0.0, 0.0],
        })
        write_parquet(day, prediction_dir / f"202401{index:02d}.parquet")

    artifacts = optimize_signals(
        prediction_dir,
        tmp_path / "signals",
        window=120,
        buy_percentile=0.80,
        sell_percentile=0.10,
    )

    signals = artifacts.model_signals
    assert signals.columns.tolist() == [
        "datetime", "order_book_id", "model_id", "value", "percent", "buy_signal", "sell_signal"
    ]
    assert len(artifacts.model_buy_points.loc[artifacts.model_buy_points["order_book_id"].eq(
        "BUY.XSHG")]) == 10
    assert len(artifacts.model_sell_points.loc[artifacts.model_sell_points["order_book_id"].eq(
        "SELL.XSHG")]) == 1
    assert artifacts.model_sell_points.iloc[0]["datetime"] == pd.Timestamp("2024-01-10")
    assert artifacts.ensemble_signals["model_id"].eq("ensemble_mean").all()
