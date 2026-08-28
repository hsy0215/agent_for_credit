"""验证风格清单只依据训练期筛除全空原始因子。"""

import numpy as np
import pandas as pd
from scripts.generate_style_manifest import training_coverage


def test_training_coverage_excludes_test_only_and_non_finite_factors(tmp_path) -> None:
    """测试期才有值或始终非有限的因子不能泄漏进风格清单。"""
    path = tmp_path / "factors.parquet"
    pd.DataFrame({
        "datetime": pd.to_datetime(["2021-12-30", "2022-01-04", "2022-04-01"]),
        "order_book_id": ["A", "A", "A"],
        "train_observed": [1.0, np.nan, np.nan],
        "test_only": [np.nan, np.nan, 2.0],
        "all_missing": [np.nan, np.nan, np.nan],
        "all_infinite": [np.inf, -np.inf, np.inf],
    }).to_parquet(path, row_group_size=2)

    observed = training_coverage(
        [path],
        ["train_observed", "test_only", "all_missing", "all_infinite"],
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2022-03-01"),
    )

    assert observed == {"train_observed"}
