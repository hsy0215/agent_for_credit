"""验证五日窗口特征严格来自目标日之前。

"""

import numpy as np
import pandas as pd

from buy_point_project.feature import add_previous_window_features, model_columns


def test_window_features_only_use_previous_rows() -> None:
    """T 日的 tminus1/tminus2 应分别等于 T-1/T-2 原始值。"""
    frame = pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=7),
        "order_book_id": ["000001.XSHE"] * 7,
        "factor_a": np.arange(7, dtype=float),
        "financial_b": np.arange(10, 17, dtype=float),
    })

    result = add_previous_window_features(frame, ["factor_a", "financial_b"], (2, 1))

    assert result.loc[6, "factor_a__tminus1"] == 5.0
    assert result.loc[6, "factor_a__tminus2"] == 4.0
    assert result.loc[6, "financial_b__delta__tminus1"] == 1.0
    assert "factor_a__tminus1" in model_columns(["factor_a", "financial_b"], (2, 1))
