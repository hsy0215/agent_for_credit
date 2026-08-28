"""加载 A500 年度因子、目标标签与互斥风格清单。

所有输入路径由 `buy_point_project.config.Config` 注入。本模块不保存运行状态，也不
依赖调用时的工作目录。

"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DataConfig


def _normalize_panel_keys(frame: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """统一交易日和股票代码字段的数据类型。"""
    result = frame.copy()
    result[date_column] = pd.to_datetime(result[date_column]).dt.normalize()
    result["order_book_id"] = result["order_book_id"].astype(str)
    return result


class DataLoader:
    """按风格和日期范围读取本地 Parquet 数据。

    Args:
        config: 外部输入文件配置。
    """

    def __init__(self, config: DataConfig) -> None:
        """保存由编排层注入的输入配置。"""
        self.config = config

    def factor_files(self) -> list[Path]:
        """返回按名称排序的年度因子文件。"""
        return sorted(self.config.factor_dir.glob("*.parquet"))

    def label_files(self) -> list[Path]:
        """返回按日期名称排序的生成标签文件。"""
        return sorted(self.config.label_dir.glob("*.parquet"))

    def validate(self) -> None:
        """检查训练所需输入是否完整。

        Raises:
            FileNotFoundError: 因子目录、因子文件、标签或风格清单不存在。
        """
        if not self.config.factor_dir.is_dir():
            raise FileNotFoundError(f"因子目录不存在: {self.config.factor_dir}")
        if not self.factor_files():
            raise FileNotFoundError(f"因子目录没有 parquet: {self.config.factor_dir}")
        if not self.config.label_dir.is_dir():
            raise FileNotFoundError(f"生成标签目录不存在: {self.config.label_dir}")
        if not self.label_files():
            raise FileNotFoundError(f"生成标签目录没有 parquet: {self.config.label_dir}")
        if not self.config.style_manifest_path.is_file():
            raise FileNotFoundError(f"风格清单不存在: {self.config.style_manifest_path}")

    def load_style_manifest(self) -> pd.DataFrame:
        """读取并校验 notebook 已生成的互斥风格因子清单。

        Returns:
            包含 factor_name、style_id 和 style_name 的清单。

        Raises:
            KeyError: 清单缺少必需字段。
            ValueError: 同一个因子在清单中出现多次。
        """
        frame = pd.read_csv(self.config.style_manifest_path)
        required = {"factor_name", "style_id", "style_name"}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"风格清单缺少字段: {sorted(missing)}")
        frame["style_id"] = pd.to_numeric(frame["style_id"], errors="raise").astype(int)
        if frame["factor_name"].duplicated().any():
            raise ValueError("风格清单中 factor_name 不是唯一值")
        return frame

    def load_calendar(self, end_date: pd.Timestamp) -> pd.DatetimeIndex:
        """从全部年度因子文件构建交易日历。

        Args:
            end_date: 日历允许包含的最后日期。

        Returns:
            已去重、升序且归一化的交易日索引。
        """
        pieces = []
        for path in self.factor_files():
            dates = pd.read_parquet(path, columns=["datetime"])["datetime"]
            pieces.append(pd.to_datetime(dates).dt.normalize())
        calendar = pd.DatetimeIndex(
            pd.concat(pieces, ignore_index=True).drop_duplicates().sort_values())
        return calendar[calendar <= end_date]

    def load_style_panel(
        self,
        factor_columns: list[str],
        target_column: str,
        load_start: pd.Timestamp,
        load_end: pd.Timestamp,
    ) -> pd.DataFrame:
        """加载一个风格的因子面板并一对一连接标签。

        Args:
            factor_columns: 当前风格使用的原始因子列。
            target_column: 标签文件中的目标列。
            load_start: 闭区间起始交易日。
            load_end: 闭区间结束交易日。

        Returns:
            按股票、交易日排序的因子与标签面板。

        Raises:
            ValueError: 日期范围内没有因子，或连接键不满足一对一。
        """
        required = ["datetime", "order_book_id", *dict.fromkeys(factor_columns)]
        end_exclusive = load_end + pd.Timedelta(days=1)
        pieces = []
        for path in self.factor_files():
            try:
                part = pd.read_parquet(
                    path,
                    columns=required,
                    filters=[
                        ("datetime", ">=", load_start),
                        ("datetime", "<", end_exclusive),
                    ],
                )
            except Exception:
                # 部分历史 parquet 不支持 Timestamp predicate，回退到内存过滤。
                part = pd.read_parquet(path, columns=required)
                timestamps = pd.to_datetime(part["datetime"])
                part = part.loc[timestamps.ge(load_start) & timestamps.lt(end_exclusive)]
            if not part.empty:
                pieces.append(part)
        if not pieces:
            raise ValueError("指定日期区间没有因子数据")

        factors = pd.concat(pieces, ignore_index=True).rename(columns={"datetime": "trade_date"})
        factors = _normalize_panel_keys(factors, "trade_date")
        labels = pd.read_parquet(
            self.config.label_dir,
            columns=["order_book_id", "datetime", target_column],
            filters=[
                ("datetime", ">=", load_start),
                ("datetime", "<", end_exclusive),
            ],
        ).rename(columns={"datetime": "trade_date"})
        labels = _normalize_panel_keys(labels, "trade_date")
        panel = factors.merge(
            labels,
            on=["order_book_id", "trade_date"],
            how="left",
            validate="one_to_one",
        )
        return panel.sort_values(["order_book_id", "trade_date"]).reset_index(drop=True)
