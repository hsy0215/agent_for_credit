"""验证从100模型预测生成优选8模型组合的双向信号。"""

import pandas as pd
from scripts.build_ensemble_buy_signals import (
    SELECTED_MODEL_IDS,
    build_ensemble_buy_sell_signals,
)

from buy_point_project.io import write_parquet


def test_selected_ensemble_builds_buy_and_sell_results(tmp_path) -> None:
    """只平均固定8模型，并为每条组合事件保留全部来源模型名称。"""
    prediction_dir = tmp_path / "prediction/test"
    filler_model_ids = tuple(f"rf_filler_T{index:03d}" for index in range(92))
    all_model_ids = (*SELECTED_MODEL_IDS, *filler_model_ids)
    for day_index in range(1, 11):
        rows = []
        for model_index, model_id in enumerate(all_model_ids):
            selected_offset = model_index / 1000 if model_id in SELECTED_MODEL_IDS else 1000.0
            rows.extend([
                {
                    "datetime": pd.Timestamp(2024, 1, day_index),
                    "order_book_id": "BUY.XSHG",
                    "model_id": model_id,
                    "value": float(day_index) + selected_offset,
                    "target": 0.0,
                },
                {
                    "datetime": pd.Timestamp(2024, 1, day_index),
                    "order_book_id": "SELL.XSHG",
                    "model_id": model_id,
                    "value": float(11 - day_index) + selected_offset,
                    "target": 0.0,
                },
            ])
        write_parquet(
            pd.DataFrame(rows),
            prediction_dir / f"202401{day_index:02d}.parquet",
        )

    output_dir = tmp_path / "ensemble_8_models"
    artifacts = build_ensemble_buy_sell_signals(prediction_dir, output_dir)

    assert len(artifacts.summary) == 20
    assert artifacts.summary["source_model_count"].eq(8).all()
    assert artifacts.summary["datetime"].dt.strftime("%H:%M:%S").eq("09:30:00").all()
    source_groups = artifacts.events.groupby(["datetime", "order_book_id"])
    assert source_groups["model_id"].nunique().eq(8).all()
    assert set(artifacts.events["model_id"]) == set(SELECTED_MODEL_IDS)
    assert artifacts.events.columns.tolist() == ["datetime", "order_book_id", "model_id", "signal"]

    buy_points = artifacts.events.loc[artifacts.events["signal"].eq(1)]
    sell_points = artifacts.events.loc[artifacts.events["signal"].eq(2)]
    assert len(buy_points.loc[buy_points["order_book_id"].eq("BUY.XSHG")]) == 80
    assert len(sell_points.loc[sell_points["order_book_id"].eq("SELL.XSHG")]) == 8
    assert sell_points["datetime"].nunique() == 1
    assert sell_points.iloc[0]["datetime"] == pd.Timestamp("2024-01-10 09:30:00")

    expected_files = {f"2024-01-{day:02d}.parquet" for day in range(1, 11)}
    assert {path.name for path in output_dir.glob("*.parquet")} == expected_files
    saved = pd.read_parquet(output_dir / "2024-01-10.parquet")
    assert saved.columns.tolist() == ["datetime", "order_book_id", "model_id", "signal"]
