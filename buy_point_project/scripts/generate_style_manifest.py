"""根据 A500 因子字段和因子分类表生成十风格因子清单。

分类前只使用模型训练期检查原始因子覆盖率。训练期内没有任何有限观测值的因子不会
进入风格清单，从而也不会在训练阶段扩展出全空的原值、差分、变化率和滞后特征。
测试期数据绝不参与该筛选。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from buy_point_project.config import Config
from buy_point_project.sampler import purged_train_end, time_spec_start

STYLE_NAMES = {
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

SOURCE_RULES = [
    (7, ["利润来源:估值"]),
    (8, ["利润来源:经营", "利润来源:现金流"]),
    (9, ["利润来源:成长"]),
    (4, ["利润来源:波动", "利润来源:波动性"]),
    (5, ["利润来源:流动性", "利润来源:成交占比"]),
    (6, ["利润来源:买卖不平衡"]),
    (2, ["利润来源:反转"]),
    (1, ["利润来源:动量"]),
    (3, ["利润来源:技术指标", "利润来源:状态类"]),
    (10, ["利润来源:相关性"]),
]

KEYWORD_RULES = [
    (6, [
        "oir", "soir", "of i", "ofi", "voi", "mpb", "mpc", "mci", "netbid", "spread", "orderflow",
        "order_flow", "买卖不平衡"
    ]),
    (5, ["illiq", "liquid", "turnover", "volume", "amount", "vwap", "lqs", "成交量", "成交额",
         "成交占比"]),  # noqa: E501
    (4, ["volatility", "volat", "std", "skew", "kurt", "amplitude", "jump", "drawdown", "波动",
         "振幅"]),
    (7, ["pe_", "pb_", "ps_", "pcf", "dividend", "book_to_market", "估值"]),
    (8, [
        "roe", "roa", "margin", "profit", "cashflow", "cash_flow", "debt", "leverage", "solvency",
        "资产负债", "现金流", "盈利质量"
    ]),
    (
        9,
        [
            "growth",
            "yoy",
            "qoq",
            "ttm",
            "forecast",
            "expect",
            "revenue",
            "income",
            "预期",
            "成长",
            "同比",  # noqa: E501
            "环比"
        ]),
    (10, ["corr", "correlation", "beta", "size", "market_cap", "市值", "指数", "行业中性"]),
    (1, ["rankmom", "momentum", "mom", "trend", "breakout", "ret", "收益率", "动量", "趋势"]),
    (2, ["reversal", "reverse", "mean_reversion", "反转", "超买", "超卖", "偏离"]),
    (
        3,
        [
            "rsi",
            "macd",
            "kdj",
            "cci",
            "boll",
            "bias",
            "shadow",
            "high",
            "low",
            "technical",
            "技术指标",  # noqa: E501
            "状态类"
        ]),
]


def standardize_factor_name(name: str) -> str:
    """把 A500 列名与 Excel 中的因子名映射到同一个命名空间。"""
    value = str(name).strip()
    for prefix in ("alpha191_", "alpha101_"):
        if value.startswith(prefix):
            return value[len(prefix):]
    if value.startswith("qlib158_qlib"):
        return value.replace("qlib158_qlib", "qlib158_", 1)
    prefixes = (
        "factor_founder_1_",
        "factor_founder_2_",
        "factor_intraday6_",
        "factor_intraday57_",
    )
    for prefix in prefixes:
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def classify_factor(name: str, metadata_text: str = "") -> tuple[int, str, str]:
    """按 Notebook 的优先级返回唯一风格、风格名称和辅助标签。"""
    raw_name = str(name)
    metadata_lower = str(metadata_text).lower()
    searchable_text = f"{standardize_factor_name(raw_name)} {metadata_text}".lower()
    for style_id, keywords in SOURCE_RULES:
        if any(keyword in metadata_lower for keyword in keywords):
            return style_id, STYLE_NAMES[style_id], ",".join(keywords)
    for style_id, keywords in KEYWORD_RULES:
        hits = [keyword for keyword in keywords if keyword in searchable_text]
        if hits:
            tags = ",".join(dict.fromkeys(hits[:3]))
            return style_id, STYLE_NAMES[style_id], tags
    if raw_name.startswith(("mrq0_", "financial_")):
        return 8, STYLE_NAMES[8], "financial_unclassified"
    return 10, STYLE_NAMES[10], "uninterpreted_or_exposure"


def _has_finite_parquet_statistics(statistics: object | None) -> bool:
    """返回列块统计量是否证明其中至少存在一个有限数值。"""
    if statistics is None or not statistics.has_min_max:
        return False
    try:
        return math.isfinite(float(statistics.min)) or math.isfinite(float(statistics.max))
    except (TypeError, ValueError):
        return False


def _finite_columns_in_frame(frame: pd.DataFrame, columns: list[str]) -> set[str]:
    """返回数据块中至少含一个有限数值的因子列。"""
    if not columns or frame.empty:
        return set()
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    return set(values.columns[np.isfinite(values.to_numpy(dtype=float, copy=False)).any(axis=0)])


def training_coverage(
    factor_files: list[Path],
    factor_columns: list[str],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> set[str]:
    """仅在闭区间训练期内找出至少含一个有限观测值的原始因子。

    完全位于训练期内的 Parquet row group 优先使用列统计量，避免为覆盖检查再次读取
    整个宽表；只有跨越训练期边界或缺少可靠统计量的候选列才读取实际数据。
    """
    observed: set[str] = set()
    candidates = set(factor_columns)
    start_value = train_start.to_pydatetime()
    end_value = (train_end + pd.Timedelta(days=1)).to_pydatetime()

    for path in factor_files:
        parquet = pq.ParquetFile(path)
        schema_names = parquet.schema.names
        date_index = schema_names.index("datetime")
        column_indexes = {name: schema_names.index(name) for name in candidates}

        for row_group_index in range(parquet.num_row_groups):
            metadata = parquet.metadata.row_group(row_group_index)
            date_statistics = metadata.column(date_index).statistics
            if date_statistics is None or not date_statistics.has_min_max:
                overlaps = True
                wholly_inside = False
            else:
                row_min = pd.Timestamp(date_statistics.min).to_pydatetime()
                row_max = pd.Timestamp(date_statistics.max).to_pydatetime()
                overlaps = row_max >= start_value and row_min < end_value
                wholly_inside = row_min >= start_value and row_max < end_value
            if not overlaps:
                continue

            unresolved: list[str] = []
            for factor in candidates.difference(observed):
                statistics = metadata.column(column_indexes[factor]).statistics
                if wholly_inside and _has_finite_parquet_statistics(statistics):
                    observed.add(factor)
                elif not wholly_inside or statistics is None or not statistics.has_min_max:
                    unresolved.append(factor)

            # 边界 row group 必须按日期过滤；缺少统计量的列也必须读取后判断 NaN/Inf。
            chunk_size = 128
            for offset in range(0, len(unresolved), chunk_size):
                chunk = unresolved[offset:offset + chunk_size]
                table = parquet.read_row_group(row_group_index, columns=["datetime", *chunk])
                frame = table.to_pandas()
                dates = pd.to_datetime(frame["datetime"])
                frame = frame.loc[dates.ge(train_start) & dates.le(train_end)]
                observed.update(_finite_columns_in_frame(frame, chunk))

        if observed == candidates:
            break
    return observed


def build_style_manifest(
    factor_dir: Path,
    classification_path: Path,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取首个因子文件的 schema 和 Excel 元数据，生成十风格清单。"""
    factor_files = sorted(factor_dir.glob("*.parquet"))
    if not factor_files:
        raise FileNotFoundError(f"因子目录下没有 Parquet 文件: {factor_dir}")
    if not classification_path.is_file():
        raise FileNotFoundError(f"因子分类表不存在: {classification_path}")

    schema = pq.ParquetFile(factor_files[0]).schema.names
    factor_columns = [name for name in schema if name not in {"datetime", "order_book_id"}]
    metadata = pd.read_excel(classification_path, sheet_name="因子归类")
    metadata = metadata.dropna(subset=["因子名称"]).copy()
    metadata["normalized_name"] = metadata["因子名称"].map(standardize_factor_name)
    metadata_lookup = metadata.drop_duplicates("normalized_name").set_index("normalized_name")

    observed = training_coverage(factor_files, factor_columns, train_start, train_end)
    excluded = set(factor_columns).difference(observed)
    rows = []
    excluded_rows = []
    metadata_columns = ["描述", "数据信息", "统计角度"]
    for factor in factor_columns:
        normalized_name = standardize_factor_name(factor)
        row = metadata_lookup.loc[
            normalized_name] if normalized_name in metadata_lookup.index else None
        metadata_text = ""
        if row is not None:
            descriptions = " ".join(str(row.get(column, "")) for column in metadata_columns)
            metadata_text = f"利润来源:{row.get('利润来源', '')} {descriptions}"
        style_id, style_name, tags = classify_factor(factor, metadata_text)
        classified = {
            "factor_name": factor,
            "normalized_name": normalized_name,
            "style_id": style_id,
            "style_name": style_name,
            "secondary_tags": tags,
            "is_fundamental": factor.startswith(("mrq0_", "financial_")),
            "metadata_matched": row is not None,
            "profit_source": row.get("利润来源", "") if row is not None else "",
        }
        if factor in excluded:
            excluded_rows.append({
                **classified,
                "coverage_start": train_start.date().isoformat(),
                "coverage_end": train_end.date().isoformat(),
                "exclusion_reason": "训练区间内无任何有限观测值",
            })
        else:
            rows.append(classified)
    return pd.DataFrame(rows), pd.DataFrame(excluded_rows)


def main() -> None:
    """从项目配置读取输入，并将风格清单写到配置指定位置。"""
    config = Config.load()
    classification_path = config.input.factor_classification_path
    if classification_path is None:
        raise ValueError("配置 input.factor_classification_path 不能为空")
    factor_files = sorted(config.input.factor_dir.glob("*.parquet"))
    if not factor_files:
        raise FileNotFoundError(f"因子目录下没有 Parquet 文件: {config.input.factor_dir}")
    calendar_parts = [
        pd.to_datetime(pd.read_parquet(path, columns=["datetime"])["datetime"]).dt.normalize()
        for path in factor_files
    ]
    calendar = pd.DatetimeIndex(
        pd.concat(calendar_parts, ignore_index=True).drop_duplicates().sort_values())
    data_start = calendar.min()
    train_end = purged_train_end(
        calendar,
        pd.Timestamp(config.model.test_start),
        config.model.label_lookahead_trading_days,
    )
    reference_quarter = pd.Period(pd.Timestamp(config.model.test_start), freq="Q")
    # 清单由全部 10 个时间方案共用，因此按最晚的方案起点筛选；否则一个仅在早期有值
    # 的因子虽能进入清单，仍会在短回看模型中扩展成整列为空的滞后特征。
    train_start = max(
        time_spec_start(data_start, reference_quarter, spec)
        for spec in config.model.time_specs.values()
    )
    manifest, excluded = build_style_manifest(
        config.input.factor_dir,
        classification_path,
        train_start,
        train_end,
    )
    output_path = config.input.style_manifest_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False, encoding="utf-8-sig")
    excluded_path = output_path.with_name(f"{output_path.stem}_excluded.csv")
    excluded.to_csv(excluded_path, index=False, encoding="utf-8-sig")

    counts = manifest.groupby(["style_id", "style_name"]).size().rename("factor_count")
    print(counts.to_string(), flush=True)
    print(
        f"生成完成: rows={len(manifest):,}, "
        f"metadata_matched={int(manifest['metadata_matched'].sum()):,}, "
        f"train_period={train_start.date()}..{train_end.date()}, excluded={len(excluded):,}, "
        f"output={output_path}, excluded_output={excluded_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
