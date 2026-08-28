"""训练并交付方案B的 10 风格 × 10 时间方案树模型。

每个模型只使用 T-5 至 T-1 的风格特征拟合一次完整 train，随后序列化模型和
预处理器并分别预测 train/test。所有路径和超参数来自 Config。

"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd

from .config import Config
from .data import DataLoader
from .feature import add_previous_window_features, model_columns
from .io import write_csv, write_json
from .log import LoggerBase
from .metrics import prediction_metrics
from .model import TreeModel, TreeModelConfig
from .prediction import PredictionCollector, build_prediction_frame
from .preprocessing import FeatureProcessor
from .sampler import (
    deterministic_quarter_dates,
    purged_train_end,
    quarter_balanced_weights,
    time_spec_start,
)


def _tree_model_config(config: Config, seed: int) -> TreeModelConfig:
    """由顶层配置派生一个模型实例的最小配置。"""
    return TreeModelConfig(
        n_estimators=config.model.n_estimators,
        min_samples_leaf=config.model.min_samples_leaf,
        max_features=config.model.max_features,
        random_state=seed,
    )


def _fit_model(
    train: pd.DataFrame,
    feature_columns: list[str],
    config: Config,
    sample_weights: np.ndarray,
    seed: int,
) -> tuple[FeatureProcessor, TreeModel]:
    """拟合一对配套的预处理器和随机森林。"""
    processor = FeatureProcessor(feature_columns, config.model.feature_clip_abs)
    features = processor.fit_transform(train)
    model = TreeModel(_tree_model_config(config, seed))
    model.fit(features, train["target"].to_numpy(dtype=float), sample_weights)
    return processor, model


def _predict(frame: pd.DataFrame, processor: FeatureProcessor, model: TreeModel,
             model_id: str) -> pd.DataFrame:
    """使用配套预处理器和模型生成统一 schema 预测。"""
    values = model.predict(processor.transform(frame))
    return build_prediction_frame(frame, values,
                                  model_id).rename(columns={"datetime": "trade_date"})


def _style_features(manifest: pd.DataFrame, config: Config) -> dict[int, list[str]]:
    """按配置顺序生成十个风格的互斥因子列表。

    Raises:
        ValueError: 清单中的风格名称与配置不一致，或某风格没有因子。
    """
    result: dict[int, list[str]] = {}
    for style_id, style_name in config.model.styles.items():
        rows = manifest.loc[manifest["style_id"].eq(style_id)]
        actual_names = set(rows["style_name"].dropna().astype(str))
        if actual_names != {style_name}:
            raise ValueError(f"style_id={style_id} 名称不一致: {actual_names} != {style_name}")
        features = rows["factor_name"].astype(str).tolist()
        if not features:
            raise ValueError(f"style_id={style_id} 没有因子")
        result[style_id] = features
    return result


def run_training(config: Config, data_loader: DataLoader, logger: LoggerBase) -> None:
    """执行完整模型训练并写出可单独打包的结果。

    Args:
        config: 全局配置，是路径和超参数的唯一来源。
        data_loader: 已由编排层组装的数据加载器。
        logger: 自动携带任务来源的日志对象。

    Raises:
        RuntimeError: 最终没有完整生成配置要求的全部模型。
        ValueError: 输入数据无法满足特征、采样或训练约束。
    """
    test_start = pd.Timestamp(config.model.test_start)
    test_end = pd.Timestamp(config.model.test_end)
    calendar = data_loader.load_calendar(test_end)
    data_start = calendar.min()
    train_end = purged_train_end(
        calendar,
        test_start,
        config.model.label_lookahead_trading_days,
    )
    training_dates = deterministic_quarter_dates(calendar, config.model.training_dates_per_quarter)
    reference_quarter = pd.Period(test_start, freq="Q")
    manifest = data_loader.load_style_manifest()
    style_features = _style_features(manifest, config)
    collector = PredictionCollector()
    metrics_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []

    config.model_prediction_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.model_dump(mode="json"), config.model_prediction_dir / "resolved_config.json")
    write_csv(manifest, config.model_prediction_dir / "input_snapshot/style_manifest.csv")

    for style_id, style_name in config.model.styles.items():
        logger.info(f"读取并构建风格特征: {style_name}")
        factor_columns = style_features[style_id]
        panel = data_loader.load_style_panel(
            factor_columns,
            config.model.target_column,
            data_start,
            test_end,
        ).rename(columns={config.model.target_column: "target"})
        panel = add_previous_window_features(
            panel,
            factor_columns,
            config.model.window_lags,
        )
        enough_history = panel["__history_count"].ge(max(config.model.window_lags))
        test = panel.loc[panel["trade_date"].between(test_start, test_end) & enough_history].copy()
        feature_columns = model_columns(factor_columns, config.model.window_lags)

        for time_id, time_spec in config.model.time_specs.items():
            model_id = f"rf_{style_name}_{time_id}"
            logger.info(f"训练模型: {model_id}")
            train_start = time_spec_start(data_start, reference_quarter, time_spec)
            candidate_mask = (panel["trade_date"].between(train_start, train_end)
                              & panel["trade_date"].isin(training_dates) & enough_history)
            train = panel.loc[candidate_mask].dropna(subset=["target"]).copy()
            if train.empty or test.empty:
                raise ValueError(f"{model_id} 的 train 或 test 为空")
            seed = config.model.random_state + style_id * 100 + int(time_id[1:])

            processor, model = _fit_model(
                train,
                feature_columns,
                config,
                quarter_balanced_weights(train, reference_quarter, time_spec.half_life_quarters),
                seed,
            )
            train_prediction = _predict(train, processor, model, model_id)
            test_prediction = _predict(test, processor, model, model_id)

            model_dir = config.models_dir / model_id
            model.serialize(model_dir / "model.state")
            processor.serialize(model_dir / "feature_processor.pkl")
            write_csv(processor.statistics_frame(), model_dir / "feature_statistics.csv")

            split_predictions = {
                "train": train_prediction,
                "test": test_prediction,
            }
            split_metrics: dict[str, dict[str, float | int]] = {}
            for split, prediction in split_predictions.items():
                collector.add(split, prediction, model_id)
                values = prediction_metrics(prediction)
                split_metrics[split] = values
                metrics_rows.append({
                    "model_id": model_id,
                    "split": split,
                    **values,
                })

            metadata = {
                "model_id": model_id,
                "style_id": style_id,
                "style_name": style_name,
                "time_id": time_id,
                "input_window": "T-5..T-1",
                "feature_count": len(feature_columns),
                "train_start": train_start,
                "train_end_after_purge": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "rows": {
                    "train": len(train),
                    "test": len(test),
                },
                "metrics": split_metrics,
            }
            write_json(metadata, model_dir / "metadata.json")
            artifact_rows.append({
                "model_id": model_id,
                "model_state": str(model_dir / "model.state"),
                "feature_processor": str(model_dir / "feature_processor.pkl"),
            })
            write_csv(pd.DataFrame(metrics_rows), config.metrics_dir / "model_metrics.csv")
            del processor, model

        del panel, test
        gc.collect()

    expected_models = len(config.model.styles) * len(config.model.time_specs)
    if len(artifact_rows) != expected_models:
        raise RuntimeError(f"应生成 {expected_models} 个模型，实际 {len(artifact_rows)}")
    logger.info("按 train/test 和交易日写出预测")
    collector.write(config.prediction_dir)
    write_csv(pd.DataFrame(artifact_rows), config.model_prediction_dir / "artifact_manifest.csv")
    write_csv(pd.DataFrame(metrics_rows), config.metrics_dir / "model_metrics.csv")
