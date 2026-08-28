"""验证树模型和预处理器的序列化往返一致性。

"""

import warnings

import numpy as np
import pandas as pd

from buy_point_project.model import TreeModel, TreeModelConfig
from buy_point_project.preprocessing import FeatureProcessor


def test_model_and_processor_round_trip(tmp_path) -> None:
    """恢复后的模型预测与训练后内存实例必须完全一致。"""
    frame = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        "b": [2.0, 4.0, 6.0, 8.0, np.nan, 12.0],
    })
    target = np.array([0.1, 0.2, 0.25, 0.4, 0.45, 0.6])
    processor = FeatureProcessor(["a", "b"], clip_abs=1e30)
    features = processor.fit_transform(frame)
    model = TreeModel(
        TreeModelConfig(
            n_estimators=5,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=7,
        ))
    model.fit(features, target, np.ones(len(target)))
    expected = model.predict(features)

    model_path = tmp_path / "model.state"
    processor_path = tmp_path / "feature_processor.pkl"
    model.serialize(model_path)
    processor.serialize(processor_path)

    restored_processor = FeatureProcessor.deserialize(processor_path)
    restored_model = TreeModel.deserialize(model_path)
    actual = restored_model.predict(restored_processor.transform(frame))
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(restored_processor.feature_mean_, [3.6, 6.4])
    assert restored_processor.feature_std_ is not None


def test_large_float32_features_have_finite_audit_statistics() -> None:
    """接近裁剪上限的特征不应让标准差计算溢出或产生运行时警告。"""
    frame = pd.DataFrame({"large": [1e30, -1e30, 0.0, np.nan]})
    processor = FeatureProcessor(["large"], clip_abs=1e30)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        features = processor.fit_transform(frame)

    assert features.dtype == np.float32
    assert processor.feature_mean_ is not None
    assert processor.feature_std_ is not None
    assert np.isfinite(processor.feature_mean_).all()
    assert np.isfinite(processor.feature_std_).all()
