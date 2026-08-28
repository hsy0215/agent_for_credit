"""验证仅含 train/test 的方案B能够端到端训练100个模型。"""

from pathlib import Path

import numpy as np
import pandas as pd

from buy_point_project.config import Config, InputConfig, ModelConfig, TimeSpec
from main import main


def _write_scheme_b_inputs(root: Path) -> InputConfig:
    """写出十风格小型因子、标签和互斥风格清单。"""
    factor_dir = root / "factors"
    factor_dir.mkdir()
    dates = pd.bdate_range("2020-01-02", "2022-04-15")
    stocks = ["000001.XSHE", "600000.XSHG"]
    keys = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "order_book_id"])
    factors = keys.to_frame(index=False)
    row_number = np.arange(len(factors), dtype=float)
    factor_names = [f"factor_{index}" for index in range(1, 11)]
    for index, factor_name in enumerate(factor_names, start=1):
        factors[factor_name] = np.sin(row_number / (index + 2)) + row_number * 1e-4
    factors.to_parquet(factor_dir / "2020_2022.parquet", index=False)

    labels = factors[["datetime", "order_book_id"]].copy()
    labels["buy_label"] = np.cos(row_number / 17.0) * 0.1
    labels["sell_label"] = 0.0
    label_dir = root / "output/label"
    label_dir.mkdir(parents=True)
    labels.to_parquet(label_dir / "2020-01-02.parquet", index=False)

    prices = factors[["datetime", "order_book_id"]].rename(columns={"datetime": "trade_date"})
    prices["close"] = 10.0
    prices["prev_close"] = 10.0
    price_path = root / "prices.parquet"
    prices.to_parquet(price_path, index=False)

    style_names = {
        1: "momentum",
        2: "reversal",
        3: "technical_state",
        4: "volatility_risk",
        5: "liquidity_turnover",
        6: "microstructure_orderflow",
        7: "valuation",
        8: "quality_financial_health",
        9: "growth_expectation",
        10: "correlation_size_exposure",
    }
    manifest = pd.DataFrame({
        "factor_name": factor_names,
        "style_id": list(style_names),
        "style_name": list(style_names.values()),
    })
    manifest_path = root / "style_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return InputConfig(
        factor_dir=factor_dir,
        price_path=price_path,
        label_path=label_dir,
        style_manifest_path=manifest_path,
    )


def test_scheme_b_trains_once_and_only_writes_train_test(tmp_path: Path) -> None:
    """100个模型应只拟合完整train，并且只生成train/test两个预测目录。"""
    inputs = _write_scheme_b_inputs(tmp_path)
    styles = {
        1: "momentum",
        2: "reversal",
        3: "technical_state",
        4: "volatility_risk",
        5: "liquidity_turnover",
        6: "microstructure_orderflow",
        7: "valuation",
        8: "quality_financial_health",
        9: "growth_expectation",
        10: "correlation_size_exposure",
    }
    config = Config(
        input=inputs,
        output=tmp_path / "output",
        model=ModelConfig(
            target_column="buy_label",
            test_start=pd.Timestamp("2022-04-01").date(),
            test_end=pd.Timestamp("2022-04-15").date(),
            label_lookahead_trading_days=20,
            training_dates_per_quarter=2,
            window_lags=(5, 4, 3, 2, 1),
            feature_clip_abs=1e30,
            random_state=7,
            n_estimators=1,
            min_samples_leaf=1,
            max_features="sqrt",
            styles=styles,
            time_specs={f"T{index:02d}": TimeSpec()
                        for index in range(1, 11)},
        ),
    )

    main(config)

    model_states = list(config.models_dir.glob("*/model.state"))
    split_directories = {path.name for path in config.prediction_dir.iterdir() if path.is_dir()}
    metrics = pd.read_csv(config.metrics_dir / "model_metrics.csv")
    assert len(model_states) == 100
    assert split_directories == {"train", "test"}
    assert set(metrics["split"]) == {"train", "test"}
    assert len(metrics) == 200
    assert "cross_section_rank_ic_mean" not in metrics.columns
    prediction_files = list(config.prediction_dir.glob("*/*.parquet"))
    assert prediction_files
    assert all(
        len(path.stem) == 10 and path.stem[4] == "-" and path.stem[7] == "-"
        for path in prediction_files)
