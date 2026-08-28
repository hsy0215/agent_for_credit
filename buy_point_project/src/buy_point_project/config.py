"""定义项目配置模型与 YAML 单一加载入口。

YAML 明确配置外部输入、生成标签目录和一个 output 根路径；模型、预测、指标和信号
路径由 `Config` 派生。所有相对路径以配置文件目录为基准。

"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TimeSpec(BaseModel):
    """一种训练时段或季度半衰期方案。

    两种参数互斥；二者均为空表示使用全部历史且季度等权。
    """

    model_config = ConfigDict(frozen=True)

    lookback_quarters: int | None = Field(default=None, gt=0)
    half_life_quarters: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_exclusive_mode(self) -> TimeSpec:
        """校验固定回看与半衰期没有同时启用。"""
        if self.lookback_quarters is not None and self.half_life_quarters is not None:
            raise ValueError("lookback_quarters 与 half_life_quarters 不能同时设置")
        return self


class InputConfig(BaseModel):
    """外部数据路径以及标签生成与训练共用目录。"""

    factor_dir: Path
    factor_classification_path: Path | None = None
    price_path: Path
    event_path: Path | None = None
    label_path: Path
    style_manifest_path: Path


class DataConfig(BaseModel):
    """模型数据加载器使用的因子、生成标签和风格清单路径。"""

    factor_dir: Path
    label_dir: Path
    style_manifest_path: Path


class ModelConfig(BaseModel):
    """五日窗口随机森林模型配置。"""

    target_column: str
    test_start: date
    test_end: date
    label_lookahead_trading_days: int = Field(gt=0)
    training_dates_per_quarter: int = Field(gt=0)
    window_lags: tuple[int, ...]
    feature_clip_abs: float = Field(gt=0)
    random_state: int
    n_estimators: int = Field(gt=0)
    min_samples_leaf: int = Field(gt=0)
    max_features: Literal["sqrt", "log2"] | float
    styles: dict[int, str]
    time_specs: dict[str, TimeSpec]

    @model_validator(mode="after")
    def validate_model_matrix(self) -> ModelConfig:
        """校验日期、滞后以及 10×10 模型矩阵的完整性。"""
        if self.test_start > self.test_end:
            raise ValueError("test_start 不能晚于 test_end")
        if not self.window_lags or any(lag <= 0 for lag in self.window_lags):
            raise ValueError("window_lags 必须全部为正整数")
        if len(set(self.window_lags)) != len(self.window_lags):
            raise ValueError("window_lags 不能重复")
        if set(self.styles) != set(range(1, 11)):
            raise ValueError("styles 必须包含 1~10 共十个风格")
        if set(self.time_specs) != {f"T{index:02d}" for index in range(1, 11)}:
            raise ValueError("time_specs 必须包含 T01~T10")
        return self


class Config(BaseModel):
    """全局配置；由该类统一派生各模块所需配置。"""

    input: InputConfig
    output: Path
    model: ModelConfig

    @classmethod
    def load(cls) -> Config:
        """从唯一默认位置或环境变量指定位置加载全局配置。

        环境变量 ``BUY_POINT_CONFIG_PATH`` 可指定另一份配置文件；相对路径以当前工作
        目录为基准。不设置时固定读取项目根目录的 ``config/config.yaml``。
        """
        project_dir = Path(__file__).resolve().parents[2]
        configured_path = os.environ.get("BUY_POINT_CONFIG_PATH")
        path = Path(configured_path).expanduser() if configured_path else (project_dir /
                                                                           "config/config.yaml")
        return cls.from_yaml(path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """读取 YAML，并将相对路径按配置文件目录解析。

        Args:
            path: YAML 配置文件路径。

        Returns:
            已完成类型校验和绝对路径解析的全局配置。

        Raises:
            FileNotFoundError: 配置文件不存在。
            pydantic.ValidationError: 配置字段或约束不合法。
        """
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as stream:
            expanded = os.path.expandvars(stream.read())
        if "${" in expanded:
            raise ValueError(f"配置仍含未设置的环境变量: {config_path}")
        raw = yaml.safe_load(expanded)
        config = cls.model_validate(raw)
        base_dir = config_path.parent
        for field_name in InputConfig.model_fields:
            value = getattr(config.input, field_name)
            if value is None:
                continue
            if not value.is_absolute():
                setattr(config.input, field_name, (base_dir / value).resolve())
        if not config.output.is_absolute():
            config.output = (base_dir / config.output).resolve()
        return config

    @property
    def model_prediction_dir(self) -> Path:
        """模型预测任务的输出根目录。"""
        return self.output / "model_prediction"

    @property
    def models_dir(self) -> Path:
        """100 个序列化模型及其预处理器目录。"""
        return self.model_prediction_dir / "models"

    @property
    def prediction_dir(self) -> Path:
        """train/test 日期分区预测目录。"""
        return self.model_prediction_dir / "prediction"

    @property
    def metrics_dir(self) -> Path:
        """模型指标目录。"""
        return self.model_prediction_dir / "metrics"

    @property
    def signals_dir(self) -> Path:
        """信号优化脚本的输出目录。"""
        return self.model_prediction_dir / "signals"

    @property
    def ensemble_8_models_dir(self) -> Path:
        """优选八模型融合后的日期分区买卖事件目录。"""
        return self.output / "ensemble_8_models"

    @property
    def label_dir(self) -> Path:
        """返回标签生成与模型训练共用的日期分区目录。"""
        return self.input.label_path

    @property
    def strategy_dir(self) -> Path:
        """建仓、加仓和组合策略脚本的输出目录。"""
        return self.output / "strategy"

    def data_config(self) -> DataConfig:
        """派生数据加载模块所需的因子、生成标签和风格清单路径。"""
        return DataConfig(
            factor_dir=self.input.factor_dir,
            label_dir=self.input.label_path,
            style_manifest_path=self.input.style_manifest_path,
        )
