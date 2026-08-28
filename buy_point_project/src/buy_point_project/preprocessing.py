"""定义可序列化、可复现的训练特征预处理器。

处理器保存固定输入列、数值裁剪口径、中位数缺失值填充器，以及最终训练集逐特征
均值和标准差。随机森林不执行标准化，但这些统计量作为数据漂移和复现审计状态保存。

"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .feature import safe_feature_matrix


class FeatureProcessor:
    """记录训练统计量并执行中位数缺失值填充。

    Args:
        feature_columns: 有序模型输入列。
        clip_abs: 有限特征值的绝对值裁剪上限。
    """

    def __init__(self, feature_columns: list[str], clip_abs: float) -> None:
        """初始化尚未拟合的预处理器。"""
        self.feature_columns = list(feature_columns)
        self.clip_abs = float(clip_abs)
        self.feature_mean_: np.ndarray | None = None
        self.feature_std_: np.ndarray | None = None
        self.imputer = SimpleImputer(strategy="median", add_indicator=True)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        """在训练集拟合统计量和缺失值填充器后转换。

        Args:
            frame: 最终模型拟合所使用的训练面板。

        Returns:
            已填充并添加缺失指示列的 float32 特征矩阵。
        """
        values = safe_feature_matrix(frame, self.feature_columns, self.clip_abs)
        # 模型输入保持 float32；审计统计提升到 float64，避免大数平方时方差溢出。
        statistics_values = values.astype(np.float64, copy=False)
        with np.errstate(invalid="ignore"):
            self.feature_mean_ = np.nanmean(statistics_values, axis=0)
            self.feature_std_ = np.nanstd(statistics_values, axis=0)
        return self.imputer.fit_transform(values).astype(np.float32, copy=False)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """使用已拟合状态转换任意数据集。

        Raises:
            RuntimeError: 处理器尚未拟合。
        """
        if self.feature_mean_ is None or self.feature_std_ is None:
            raise RuntimeError("FeatureProcessor 尚未拟合")
        values = safe_feature_matrix(frame, self.feature_columns, self.clip_abs)
        return self.imputer.transform(values).astype(np.float32, copy=False)

    def serialize(self, path: Path) -> None:
        """将预处理状态原子保存为 pkl。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(self, temporary)
        temporary.replace(path)

    @classmethod
    def deserialize(cls, path: Path) -> FeatureProcessor:
        """从 pkl 恢复预处理状态。

        Raises:
            TypeError: 文件内容不是 `FeatureProcessor`。
        """
        processor = joblib.load(path)
        if not isinstance(processor, cls):
            raise TypeError(f"{path} 不是 {cls.__name__}")
        return processor

    def statistics_frame(self) -> pd.DataFrame:
        """返回便于审计的特征均值和标准差表。

        Raises:
            RuntimeError: 处理器尚未拟合。
        """
        if self.feature_mean_ is None or self.feature_std_ is None:
            raise RuntimeError("FeatureProcessor 尚未拟合")
        return pd.DataFrame({
            "feature": self.feature_columns,
            "train_mean": self.feature_mean_,
            "train_std": self.feature_std_,
        })
