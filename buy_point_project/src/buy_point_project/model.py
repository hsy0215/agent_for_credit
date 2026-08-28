"""定义可实例化并可持久化的随机森林回归模型。

`TreeModel.serialize` 写出带状态版本和超参数的 `model.state`；
`TreeModel.deserialize` 将其恢复为可直接预测的对象。

"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor


class TreeModelConfig(BaseModel):
    """实例化随机森林所需的最小超参数。"""

    n_estimators: int = Field(gt=0)
    min_samples_leaf: int = Field(gt=0)
    max_features: Literal["sqrt", "log2"] | float
    random_state: int
    n_jobs: int = -1


class TreeModel:
    """对 `RandomForestRegressor` 的稳定持久化封装。

    Args:
        config: 随机森林实例化配置。
    """

    STATE_VERSION = 1
    """序列化格式版本。"""

    def __init__(self, config: TreeModelConfig) -> None:
        """根据显式配置实例化尚未拟合的随机森林。"""
        self.config = config
        self.estimator = RandomForestRegressor(
            n_estimators=config.n_estimators,
            min_samples_leaf=config.min_samples_leaf,
            max_features=config.max_features,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
        )

    def fit(self, features: np.ndarray, target: np.ndarray, sample_weight: np.ndarray) -> None:
        """使用输入特征、目标和样本权重拟合随机森林。"""
        self.estimator.fit(features, target, sample_weight=sample_weight)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """返回连续买点评分。"""
        return self.estimator.predict(features)

    def serialize(self, path: Path) -> None:
        """原子保存带版本号的模型状态。

        Args:
            path: 目标 `model.state` 路径。
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(
            {
                "state_version": self.STATE_VERSION,
                "config": self.config.model_dump(),
                "estimator": self.estimator,
            }, temporary)
        temporary.replace(path)

    @classmethod
    def deserialize(cls, path: Path) -> TreeModel:
        """从模型状态恢复可预测实例。

        Raises:
            ValueError: 状态格式版本不受支持。
            TypeError: 状态中不包含随机森林对象。
        """
        state = joblib.load(path)
        if state.get("state_version") != cls.STATE_VERSION:
            raise ValueError(f"不支持的模型状态版本: {state.get('state_version')}")
        model = cls(TreeModelConfig.model_validate(state["config"]))
        if not isinstance(state.get("estimator"), RandomForestRegressor):
            raise TypeError(f"{path} 不包含随机森林")
        model.estimator = state["estimator"]
        return model
