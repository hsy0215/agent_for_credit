"""复现完整的建仓、加仓筛选、组合策略和研究图表。

脚本读取优选八模型的日期分区买卖事件、本地日行情和震荡事件。它支持 ``new`` 与
``original`` 两种建仓筛选模式，输出理论建仓点、实际建仓/加仓点、完整交易流水、
每日净值、每日持仓权重以及与 Notebook 对齐的事件和组合图。

``original`` 模式按 Notebook 原样把确认点标在连续信号组的末日；组尾必须等后续信号
中断后才能确认，因此该模式属于研究回测口径，不应直接当作无延迟实盘信号。

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from buy_point_project.config import Config
from buy_point_project.io import write_json, write_parquet

EntryFilterMode = Literal["new", "original"]


@dataclass(frozen=True)
class EntryParameters:
    """定义 cell9 两套建仓点筛选参数。"""

    mode: EntryFilterMode = "new"
    event_window_days: int = 180
    price_rank_window: int = 10
    max_price_percentile: float = 0.50
    minimum_group_signals: int = 15
    maximum_signal_gap: int = 2
    confirmed_buy_cooldown: int = 20

    def validate(self) -> None:
        """校验建仓筛选参数范围。"""
        if self.mode not in {"new", "original"}:
            raise ValueError("mode 只能是 new 或 original")
        integer_values = (
            self.event_window_days,
            self.price_rank_window,
            self.minimum_group_signals,
            self.maximum_signal_gap,
            self.confirmed_buy_cooldown,
        )
        if any(value <= 0 for value in integer_values):
            raise ValueError("建仓窗口、信号数量、间隔和冷却参数必须为正整数")
        if not 0.0 <= self.max_price_percentile <= 1.0:
            raise ValueError("max_price_percentile 必须位于 [0, 1]")


@dataclass(frozen=True)
class StrategyParameters:
    """定义 cell9 组合交易与风险控制参数。"""

    initial_cash: float = 1_000_000.0
    maximum_holdings: int = 10
    initial_position_weight: float = 0.05
    add_position_weight: float = 0.05
    maximum_stock_weight: float = 0.10
    add_on_holding_day: int = 5
    stop_loss_return: float = -0.05
    take_profit_return: float = 0.10
    half_take_profit_return: float = 0.07
    maximum_holding_days: int = 30

    def validate(self) -> None:
        """校验资金、仓位、收益阈值和持有期参数。"""
        if self.initial_cash <= 0 or self.maximum_holdings <= 0:
            raise ValueError("initial_cash 和 maximum_holdings 必须为正数")
        weights = (
            self.initial_position_weight,
            self.add_position_weight,
            self.maximum_stock_weight,
        )
        if any(not 0.0 < value <= 1.0 for value in weights):
            raise ValueError("仓位权重必须位于 (0, 1]")
        if self.initial_position_weight > self.maximum_stock_weight:
            raise ValueError("初始仓位不能超过单票上限")
        if self.stop_loss_return >= 0:
            raise ValueError("stop_loss_return 必须小于 0")
        if not 0 < self.half_take_profit_return < self.take_profit_return:
            raise ValueError("减半止盈阈值必须位于 0 与全仓止盈阈值之间")
        if self.add_on_holding_day <= 0 or self.maximum_holding_days <= 0:
            raise ValueError("加仓日和最大持有日必须为正整数")


@dataclass
class Holding:
    """记录单只股票的持仓状态。"""

    shares: float
    average_cost: float
    days_held: int = 0
    halved: bool = False


@dataclass(frozen=True)
class StrategyResult:
    """保存策略的建仓点、交易、净值和权重结果。"""

    entry_points: pd.DataFrame
    executed_buys: pd.DataFrame
    trades: pd.DataFrame
    portfolio: pd.DataFrame
    weights: pd.DataFrame
    metrics: dict[str, float | int | str]


def read_events(path: Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """读取 CSV、Excel 或 Parquet 震荡事件并保留每只股票首个事件。"""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        events = pd.read_excel(path)
    elif suffix == ".csv":
        events = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        events = pd.read_parquet(path)
    else:
        raise ValueError(f"不支持的事件文件格式: {path.suffix}")
    required = {"trade_date", "order_book_id"}
    missing = required.difference(events.columns)
    if missing:
        raise KeyError(f"事件文件缺少字段: {sorted(missing)}")
    result = events[["trade_date", "order_book_id"]].copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
    result["order_book_id"] = result["order_book_id"].astype(str)
    result = result.loc[result["trade_date"].between(start_date, end_date)]
    return (result.sort_values(["order_book_id", "trade_date"],
                               kind="stable").drop_duplicates("order_book_id",
                                                              keep="first").reset_index(drop=True))


def read_adjusted_close(
    path: Path,
    order_book_ids: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """按 cell3 口径读取日行情并利用 prev_close 构造连续后复权收盘价。"""
    columns = ["trade_date", "order_book_id", "close", "prev_close"]
    filters = [
        ("order_book_id", "in", order_book_ids),
        ("trade_date", ">=", start_date.date()),
        ("trade_date", "<=", end_date.date()),
    ]
    try:
        frame = pd.read_parquet(path, columns=columns, filters=filters)
    except Exception:
        frame = pd.read_parquet(path, columns=columns)
        dates = pd.to_datetime(frame["trade_date"])
        frame = frame.loc[frame["order_book_id"].astype(str).isin(order_book_ids)
                          & dates.between(start_date, end_date)]
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame["order_book_id"] = frame["order_book_id"].astype(str)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["prev_close"] = pd.to_numeric(frame["prev_close"], errors="coerce")
    frame = frame.sort_values(["order_book_id", "trade_date"], kind="stable")
    if frame.duplicated(["trade_date", "order_book_id"]).any():
        raise ValueError("日行情存在重复 trade_date × order_book_id")
    previous_raw_close = frame.groupby("order_book_id", sort=False)["close"].shift(1)
    action_ratio = previous_raw_close.div(frame["prev_close"])
    action_ratio = action_ratio.where(action_ratio.gt(0) & np.isfinite(action_ratio), 1.0)
    frame["adjustment_factor"] = action_ratio.groupby(frame["order_book_id"], sort=False).cumprod()
    frame["adjusted_close"] = frame["close"] * frame["adjustment_factor"]
    return (frame.pivot(index="trade_date", columns="order_book_id",
                        values="adjusted_close").sort_index())


def read_ensemble_signals(path: Path) -> pd.DataFrame:
    """读取优选八模型的单文件或日期分区四字段买卖事件。"""
    if path.is_dir():
        paths = sorted(path.glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"信号目录中没有 Parquet: {path}")
        parts = [pd.read_parquet(part) for part in paths]
        frame = pd.concat(parts, ignore_index=True)
    else:
        frame = pd.read_parquet(path)
    required = {"datetime", "order_book_id", "model_id", "signal"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"信号文件缺少字段: {sorted(missing)}")
    source = frame[["datetime", "order_book_id", "model_id", "signal"]].copy()
    source["date"] = pd.to_datetime(source["datetime"]).dt.normalize()
    source["order_book_id"] = source["order_book_id"].astype(str)
    source["model_id"] = source["model_id"].astype(str)
    source["signal"] = pd.to_numeric(source["signal"], errors="raise").astype("int8")
    if not source["signal"].isin([1, 2]).all():
        raise ValueError("signal 只能为 1（买）或 2（卖）")
    if source.duplicated(["date", "order_book_id", "model_id"]).any():
        raise ValueError("信号存在重复 datetime × order_book_id × model_id")
    result = source.groupby(["date", "order_book_id"], as_index=False, sort=False).agg(
        buy_signal=("signal", lambda values: values.eq(1).any()),
        sell_signal=("signal", lambda values: values.eq(2).any()),
    )
    # 四字段事件不再携带连续预测值；同日候选在策略中采用稳定的股票代码顺序。
    result["buy_prediction"] = 0.0
    return result.sort_values(["order_book_id", "date"], kind="stable").reset_index(drop=True)


def read_ensemble_events(path: Path) -> pd.DataFrame:
    """读取并返回策略落盘和绘图所需的原始四字段事件。"""
    if path.is_dir():
        paths = sorted(path.glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"信号目录中没有 Parquet: {path}")
        frame = pd.concat((pd.read_parquet(part) for part in paths), ignore_index=True)
    else:
        frame = pd.read_parquet(path)
    required = {"datetime", "order_book_id", "model_id", "signal"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"信号文件缺少字段: {sorted(missing)}")
    result = frame[["datetime", "order_book_id", "model_id", "signal"]].copy()
    result["datetime"] = pd.to_datetime(result["datetime"])
    result["order_book_id"] = result["order_book_id"].astype(str)
    result["model_id"] = result["model_id"].astype(str)
    result["signal"] = pd.to_numeric(result["signal"], errors="raise").astype("int8")
    return result


def build_entry_points(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    events: pd.DataFrame,
    parameters: EntryParameters,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """根据 cell9 的 new/original 模式筛选理论建仓点。

    Returns:
        长表建仓点、布尔建仓矩阵和连续预测矩阵。
    """
    parameters.validate()
    relevant = signals.loc[signals["order_book_id"].isin(prices.columns)].copy()
    prediction_matrix = relevant.pivot(index="date",
                                       columns="order_book_id",
                                       values="buy_prediction")
    candidate_matrix = (relevant.pivot(
        index="date", columns="order_book_id",
        values="buy_signal").reindex(index=prices.index, columns=prices.columns,
                                     fill_value=False).astype("boolean").fillna(False).astype(bool))
    event_window = pd.DataFrame(False, index=prices.index, columns=prices.columns, dtype=bool)
    for event in events.itertuples(index=False):
        order_book_id = str(event.order_book_id)
        if order_book_id not in event_window.columns:
            continue
        price_dates = prices[order_book_id].dropna().index
        following = price_dates[price_dates >= pd.Timestamp(str(event.trade_date))]
        if following.empty:
            continue
        raw_start_position = price_dates.get_loc(following[0])
        if not isinstance(raw_start_position, (int, np.integer)):
            raise ValueError(f"{order_book_id} 的交易日索引不是唯一值")
        start_position = int(raw_start_position)
        end_position = min(len(price_dates) - 1, start_position + parameters.event_window_days)
        event_window.loc[price_dates[start_position:end_position + 1], order_book_id] = True

    if parameters.mode == "new":
        price_percentile = prices.rolling(
            parameters.price_rank_window,
            min_periods=parameters.price_rank_window,
        ).rank(method="average", pct=True)
        entry_matrix = (candidate_matrix & price_percentile.le(parameters.max_price_percentile)
                        & event_window).fillna(False).astype(bool)
    else:
        event_candidates = (candidate_matrix & event_window).fillna(False)
        entry_matrix = pd.DataFrame(False, index=prices.index, columns=prices.columns, dtype=bool)
        for order_book_id in entry_matrix.columns:
            price_dates = prices[order_book_id].dropna().index
            if price_dates.empty:
                continue
            candidate_dates = event_candidates.index[event_candidates[order_book_id]]
            candidate_indices = sorted(
                int(index) for index in price_dates.get_indexer(candidate_dates) if index >= 0)
            if not candidate_indices:
                continue
            groups: list[list[int]] = [[candidate_indices[0]]]
            for index in candidate_indices[1:]:
                if index - groups[-1][-1] <= parameters.maximum_signal_gap:
                    groups[-1].append(index)
                else:
                    groups.append([index])
            last_confirmed = -parameters.confirmed_buy_cooldown - 1
            for group in groups:
                if len(group) < parameters.minimum_group_signals:
                    continue
                candidate_index = group[-1]
                if candidate_index - last_confirmed > parameters.confirmed_buy_cooldown:
                    entry_matrix.loc[price_dates[candidate_index], order_book_id] = True
                    last_confirmed = candidate_index

    entry_matrix.index.name = "date"
    entry_matrix.columns.name = "order_book_id"
    stacked = cast(pd.Series, entry_matrix.stack(future_stack=True))
    entry_points = stacked.loc[stacked].rename("entry_signal").reset_index()
    entry_points = entry_points.merge(
        signals[["date", "order_book_id", "buy_prediction"]],
        on=["date", "order_book_id"],
        how="left",
        validate="one_to_one",
    )
    entry_points["entry_filter_mode"] = parameters.mode
    return entry_points, entry_matrix, prediction_matrix


def _portfolio_value(cash: float, holdings: dict[str, Holding], prices: pd.Series) -> float:
    """使用当日价格或持仓成本估算组合总资产。"""
    total = cash
    for stock, holding in holdings.items():
        raw_price = prices.get(stock)
        price = float(cast(float, raw_price)) if pd.notna(raw_price) else holding.average_cost
        total += holding.shares * price
    return total


def run_strategy(
    prices: pd.DataFrame,
    entry_points: pd.DataFrame,
    entry_matrix: pd.DataFrame,
    prediction_matrix: pd.DataFrame,
    parameters: StrategyParameters,
) -> StrategyResult:
    """执行 cell9 的建仓、第五日加仓、退出和单票上限再平衡策略。"""
    parameters.validate()
    cash = parameters.initial_cash
    holdings: dict[str, Holding] = {}
    trades: list[dict[str, object]] = []
    portfolio_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []

    def record_trade(
        date: pd.Timestamp,
        stock: str,
        price: float,
        shares: float,
        amount: float,
        action: str,
    ) -> None:
        """追加一条具有统一 schema 的交易流水。"""
        trades.append({
            "date": date,
            "order_book_id": stock,
            "price": price,
            "shares": shares,
            "amount": amount,
            "action": action,
        })

    for current_date in prices.index.sort_values():
        prices_today = prices.loc[current_date]
        for holding in holdings.values():
            holding.days_held += 1

        remove: list[str] = []
        for stock, holding in list(holdings.items()):
            current_price = prices_today.get(stock)
            if pd.isna(current_price):
                continue
            current_price = float(current_price)
            current_return = current_price / holding.average_cost - 1.0
            exit_reason = ""
            if current_return <= parameters.stop_loss_return:
                exit_reason = "full_exit_stop_loss"
            elif current_return >= parameters.take_profit_return:
                exit_reason = "full_exit_take_profit"
            elif holding.days_held >= parameters.maximum_holding_days:
                exit_reason = "full_exit_max_holding"
            if exit_reason:
                amount = holding.shares * current_price
                cash += amount
                record_trade(current_date, stock, current_price, holding.shares, amount,
                             exit_reason)
                remove.append(stock)
            elif current_return >= parameters.half_take_profit_return and not holding.halved:
                sold_shares = holding.shares / 2.0
                amount = sold_shares * current_price
                holding.shares -= sold_shares
                holding.halved = True
                cash += amount
                record_trade(current_date, stock, current_price, sold_shares, amount,
                             "half_take_profit")
        for stock in remove:
            del holdings[stock]

        portfolio_value = _portfolio_value(cash, holdings, prices_today)
        for stock, holding in holdings.items():
            if holding.days_held != parameters.add_on_holding_day:
                continue
            if not bool(entry_matrix.at[current_date, stock]):
                continue
            current_price = prices_today.get(stock)
            if pd.isna(current_price):
                continue
            current_price = float(current_price)
            current_value = holding.shares * current_price
            capacity = max(0.0, parameters.maximum_stock_weight * portfolio_value - current_value)
            investment = min(parameters.add_position_weight * portfolio_value, capacity, cash)
            if investment <= 0:
                continue
            added_shares = investment / current_price
            total_cost = holding.shares * holding.average_cost + investment
            holding.shares += added_shares
            holding.average_cost = total_cost / holding.shares
            cash -= investment
            record_trade(current_date, stock, current_price, added_shares, investment, "add")

        candidates = [str(stock) for stock in entry_matrix.columns[entry_matrix.loc[current_date]]]
        candidates = [
            stock for stock in candidates
            if stock not in holdings and pd.notna(prices_today.get(stock))
        ]
        slots = max(0, parameters.maximum_holdings - len(holdings))
        if candidates and slots:
            scores = [
                (stock, float(cast(float, prediction_matrix.at[current_date, stock]))
                 if current_date in prediction_matrix.index and stock in prediction_matrix.columns
                 and pd.notna(prediction_matrix.at[current_date, stock]) else -np.inf)
                for stock in candidates
            ]
            scores.sort(key=lambda item: item[1], reverse=True)
            for stock, _ in scores[:slots]:
                current_price = float(prices_today[stock])
                investment = min(parameters.initial_position_weight * portfolio_value, cash)
                if investment <= 0:
                    continue
                shares = investment / current_price
                holdings[stock] = Holding(shares=shares, average_cost=current_price)
                cash -= investment
                record_trade(current_date, stock, current_price, shares, investment, "initial")

        estimated_value = _portfolio_value(cash, holdings, prices_today)
        cap_value = parameters.maximum_stock_weight * estimated_value
        for stock, holding in list(holdings.items()):
            current_price = prices_today.get(stock)
            if pd.isna(current_price):
                continue
            current_price = float(current_price)
            current_value = holding.shares * current_price
            if current_value <= cap_value:
                continue
            trim_amount = current_value - cap_value
            trim_shares = min(holding.shares, trim_amount / current_price)
            holding.shares -= trim_shares
            cash += trim_shares * current_price
            record_trade(current_date, stock, current_price, trim_shares,
                         trim_shares * current_price, "rebalance_trim")
            if holding.shares <= 1e-12:
                del holdings[stock]

        final_value = _portfolio_value(cash, holdings, prices_today)
        portfolio_rows.append({
            "date": current_date,
            "cash": cash,
            "port_val": final_value,
            "num_holdings": len(holdings),
            "NAV": final_value / parameters.initial_cash,
        })
        for stock, holding in holdings.items():
            current_price = prices_today.get(stock)
            reference_price = (float(current_price)
                               if pd.notna(current_price) else holding.average_cost)
            position_value = holding.shares * reference_price
            weight_rows.append({
                "date": current_date,
                "order_book_id": stock,
                "position_value": position_value,
                "weight": position_value / final_value if final_value > 0 else 0.0,
            })

    trade_frame = pd.DataFrame(
        trades, columns=["date", "order_book_id", "price", "shares", "amount", "action"])
    buy_frame = trade_frame.loc[trade_frame["action"].isin(["initial", "add"])].copy()
    buy_frame["buy_type"] = buy_frame["action"]
    portfolio = pd.DataFrame(portfolio_rows)
    weights = pd.DataFrame(weight_rows,
                           columns=["date", "order_book_id", "position_value", "weight"])
    if portfolio.empty:
        raise ValueError("价格矩阵为空，策略没有可运行交易日")
    nav = portfolio["NAV"]
    drawdown = nav.div(nav.cummax()).sub(1.0)
    metrics: dict[str, float | int | str] = {
        "final_nav":
        float(nav.iloc[-1]),
        "max_drawdown":
        float(drawdown.min()),
        "trading_days":
        len(portfolio),
        "initial_buys":
        int(trade_frame["action"].eq("initial").sum()),
        "add_on_buys":
        int(trade_frame["action"].eq("add").sum()),
        "entry_filter_mode":
        str(entry_points["entry_filter_mode"].iloc[0] if not entry_points.empty else "unknown"),
    }
    return StrategyResult(entry_points, buy_frame, trade_frame, portfolio, weights, metrics)


def filter_events_to_oscillation_windows(
    raw_events: pd.DataFrame,
    prices: pd.DataFrame,
    oscillation_events: pd.DataFrame,
    window_days: int,
) -> pd.DataFrame:
    """只保留每只股票首个震荡事件日起指定交易日窗口内的原始买卖事件。"""
    selected = []
    normalized = raw_events.copy()
    normalized["date"] = pd.to_datetime(normalized["datetime"]).dt.normalize()
    for event in oscillation_events.itertuples(index=False):
        stock = str(event.order_book_id)
        if stock not in prices.columns:
            continue
        stock_dates = prices[stock].dropna().index
        following = stock_dates[stock_dates >= pd.Timestamp(str(event.trade_date))]
        if following.empty:
            continue
        raw_start = stock_dates.get_loc(following[0])
        if not isinstance(raw_start, (int, np.integer)):
            raise ValueError(f"{stock} 的交易日索引不是唯一值")
        start = int(raw_start)
        end = min(len(stock_dates) - 1, start + window_days)
        selected.append(
            normalized.loc[normalized["order_book_id"].eq(stock)
                           & normalized["date"].between(stock_dates[start], stock_dates[end])])
    if not selected:
        return raw_events.iloc[0:0].copy()
    return pd.concat(selected,
                     ignore_index=True)[["datetime", "order_book_id", "model_id",
                                         "signal"]].drop_duplicates().reset_index(drop=True)


def expand_strategy_points(points: pd.DataFrame, raw_events: pd.DataFrame) -> pd.DataFrame:
    """将策略日期点与原始买信号来源模型关联为统一四字段格式。"""
    if points.empty:
        return pd.DataFrame(columns=["datetime", "order_book_id", "model_id", "signal"])
    point_keys = points[["date", "order_book_id"]].copy()
    point_keys["date"] = pd.to_datetime(point_keys["date"]).dt.normalize()
    point_keys = point_keys.drop_duplicates()
    buys = raw_events.loc[raw_events["signal"].eq(1)].copy()
    buys["date"] = pd.to_datetime(buys["datetime"]).dt.normalize()
    result = point_keys.merge(
        buys[["date", "order_book_id", "datetime", "model_id", "signal"]],
        on=["date", "order_book_id"],
        how="inner",
        validate="one_to_many",
    )
    return result[["datetime", "order_book_id", "model_id", "signal"]].reset_index(drop=True)


def plot_event_signals(
    prices: pd.DataFrame,
    oscillation_events: pd.DataFrame,
    raw_events: pd.DataFrame,
    entry_points: pd.DataFrame,
    executed_buys: pd.DataFrame,
    output_path: Path,
    event_window_days: int,
) -> None:
    """绘制震荡事件、原始买卖信号、策略建仓点和实际建仓/加仓点。"""
    plot_items = []
    raw = raw_events.copy()
    raw["date"] = pd.to_datetime(raw["datetime"]).dt.normalize()
    for event in oscillation_events.itertuples(index=False):
        stock = str(event.order_book_id)
        event_date = pd.Timestamp(str(event.trade_date)).normalize()
        if stock not in prices.columns:
            continue
        series = prices[stock].dropna()
        following = series.index[series.index >= event_date]
        if following.empty:
            continue
        raw_start = series.index.get_loc(following[0])
        if not isinstance(raw_start, (int, np.integer)):
            raise ValueError(f"{stock} 的交易日索引不是唯一值")
        start = int(raw_start)
        plot_start = max(0, start - 90)
        plot_end = min(len(series), start + event_window_days + 1)
        plot_items.append((stock, event_date, series.iloc[plot_start:plot_end]))
    if not plot_items:
        return

    columns = 4
    rows = (len(plot_items) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(16, max(3, 3 * rows)), squeeze=False)
    for axis, item in zip(axes.ravel(), plot_items, strict=False):
        stock, event_date, series = item
        axis.plot(series.index, series.values, color="steelblue", linewidth=1.0, label="Close")
        axis.axvline(event_date,
                     color="black",
                     linestyle="--",
                     linewidth=1.0,
                     label="Oscillation event")
        stock_raw = raw.loc[raw["order_book_id"].eq(stock)
                            & raw["date"].between(series.index.min(), series.index.max())]
        for signal, color, marker, label in [
            (1, "seagreen", "^", "Raw buy"),
            (2, "purple", "v", "Raw sell"),
        ]:
            dates = stock_raw.loc[stock_raw["signal"].eq(signal), "date"].drop_duplicates()
            dates = dates[dates.isin(series.index)]
            if not dates.empty:
                axis.scatter(dates,
                             series.reindex(dates),
                             color=color,
                             marker=marker,
                             s=28,
                             label=label,
                             zorder=4)
        stock_entries = entry_points.loc[entry_points["order_book_id"].eq(stock)].copy()
        entry_dates = pd.to_datetime(stock_entries["date"]).dt.normalize()
        entry_dates = entry_dates[entry_dates.isin(series.index)]
        if not entry_dates.empty:
            axis.scatter(entry_dates,
                         series.reindex(entry_dates),
                         facecolors="none",
                         edgecolors="crimson",
                         marker="o",
                         s=65,
                         label="Filtered entry",
                         zorder=5)
        stock_buys = executed_buys.loc[executed_buys["order_book_id"].eq(stock)]
        for buy_type, color, size, label in [
            ("initial", "crimson", 95, "Executed initial"),
            ("add", "darkorange", 80, "Executed add-on"),
        ]:
            selected = stock_buys.loc[stock_buys["buy_type"].eq(buy_type)]
            dates = pd.to_datetime(selected["date"]).dt.normalize()
            dates = dates[dates.isin(series.index)]
            if not dates.empty:
                axis.scatter(dates,
                             series.reindex(dates),
                             color=color,
                             marker="*",
                             s=size,
                             label=label,
                             zorder=6)
        axis.set_title(f"{stock}\n{event_date:%Y-%m-%d}", fontsize=9)
        axis.tick_params(axis="x", rotation=30, labelsize=7)
        axis.tick_params(axis="y", labelsize=7)
        axis.legend(loc="best", fontsize=5, framealpha=0.75)
    for axis in axes.ravel()[len(plot_items):]:
        axis.axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_portfolio_summary(result: StrategyResult, output_path: Path,
                           maximum_holdings: int) -> None:
    """绘制策略累计收益、持仓数量和逐股票持仓权重。"""
    portfolio = result.portfolio.set_index("date")
    weights = result.weights.pivot(index="date", columns="order_book_id", values="weight")
    weights = weights.reindex(portfolio.index).fillna(0.0)
    active_columns = [column for column in weights if weights[column].gt(1e-12).any()]

    figure, axes = plt.subplots(3,
                                1,
                                figsize=(14, 12),
                                sharex=True,
                                gridspec_kw={"height_ratios": [3, 1, 2]})
    axes[0].plot(portfolio.index, portfolio["NAV"], color="firebrick", label="Strategy NAV")
    axes[0].axhline(1.0, color="gray", linestyle="--")
    axes[0].set_title("Strategy Cumulative Returns & Portfolio Holdings", fontweight="bold")
    axes[0].set_ylabel("Net Asset Value (NAV)")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.35)
    axes[1].fill_between(portfolio.index, portfolio["num_holdings"], color="steelblue", alpha=0.6)
    axes[1].plot(portfolio.index, portfolio["num_holdings"], color="steelblue")
    axes[1].axhline(maximum_holdings, color="red", linestyle="-.", label="Maximum holdings")
    axes[1].set_ylabel("Holdings Count")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.35)
    if active_columns:
        axes[2].stackplot(weights.index,
                          *[weights[column] for column in active_columns],
                          labels=active_columns,
                          alpha=0.8)
        if len(active_columns) <= 20:
            axes[2].legend(loc="upper left", fontsize=6, ncol=2)
    axes[2].set_ylabel("Portfolio Weight")
    axes[2].set_xlabel("Date")
    axes[2].grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _default_signal_path(config: Config) -> Path:
    """返回优选八模型日期分区买卖事件目录。"""
    return config.ensemble_8_models_dir


def main() -> None:
    """解析输入路径和策略参数，运行建仓筛选及回测并写出结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--entry-filter-mode", choices=["new", "original"], default="new")
    parser.add_argument("--price-start", default="2023-01-01")
    parser.add_argument("--price-end", default="2026-06-30")
    parser.add_argument("--event-start", default="2023-01-01")
    parser.add_argument("--event-end", default="2025-12-30")
    parser.add_argument("--event-window-days", type=int, default=180)
    parser.add_argument("--price-rank-window", type=int, default=10)
    parser.add_argument("--max-price-percentile", type=float, default=0.50)
    parser.add_argument("--minimum-group-signals", type=int, default=15)
    parser.add_argument("--maximum-signal-gap", type=int, default=2)
    parser.add_argument("--confirmed-buy-cooldown", type=int, default=20)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--maximum-holdings", type=int, default=10)
    parser.add_argument("--initial-position-weight", type=float, default=0.05)
    parser.add_argument("--add-position-weight", type=float, default=0.05)
    parser.add_argument("--maximum-stock-weight", type=float, default=0.10)
    parser.add_argument("--add-on-holding-day", type=int, default=5)
    parser.add_argument("--stop-loss-return", type=float, default=-0.05)
    parser.add_argument("--take-profit-return", type=float, default=0.10)
    parser.add_argument("--half-take-profit-return", type=float, default=0.07)
    parser.add_argument("--maximum-holding-days", type=int, default=30)
    args = parser.parse_args()

    config = Config.load()
    if config.input.event_path is None:
        raise ValueError("配置 input.event_path 不能为空")
    signal_file = args.signal_file or _default_signal_path(config)
    output_dir = args.output_dir or config.strategy_dir
    asset_dir = args.asset_dir or Path(__file__).resolve().parents[1] / "asset"
    events = read_events(
        config.input.event_path,
        pd.Timestamp(args.event_start),
        pd.Timestamp(args.event_end),
    )
    prices = read_adjusted_close(
        config.input.price_path,
        events["order_book_id"].tolist(),
        pd.Timestamp(args.price_start),
        pd.Timestamp(args.price_end),
    )
    raw_events = read_ensemble_events(signal_file)
    signals = read_ensemble_signals(signal_file)
    entry_parameters = EntryParameters(
        mode=args.entry_filter_mode,
        event_window_days=args.event_window_days,
        price_rank_window=args.price_rank_window,
        max_price_percentile=args.max_price_percentile,
        minimum_group_signals=args.minimum_group_signals,
        maximum_signal_gap=args.maximum_signal_gap,
        confirmed_buy_cooldown=args.confirmed_buy_cooldown,
    )
    entry_points, entry_matrix, prediction_matrix = build_entry_points(
        signals, prices, events, entry_parameters)
    strategy_parameters = StrategyParameters(
        initial_cash=args.initial_cash,
        maximum_holdings=args.maximum_holdings,
        initial_position_weight=args.initial_position_weight,
        add_position_weight=args.add_position_weight,
        maximum_stock_weight=args.maximum_stock_weight,
        add_on_holding_day=args.add_on_holding_day,
        stop_loss_return=args.stop_loss_return,
        take_profit_return=args.take_profit_return,
        half_take_profit_return=args.half_take_profit_return,
        maximum_holding_days=args.maximum_holding_days,
    )
    result = run_strategy(
        prices,
        entry_points,
        entry_matrix,
        prediction_matrix,
        strategy_parameters,
    )
    event_window_signals = filter_events_to_oscillation_windows(raw_events, prices, events,
                                                                entry_parameters.event_window_days)
    strategy_entry_signals = expand_strategy_points(result.entry_points, event_window_signals)
    initial_entries = result.executed_buys.loc[result.executed_buys["buy_type"].eq("initial")]
    add_on_entries = result.executed_buys.loc[result.executed_buys["buy_type"].eq("add")]
    initial_entry_signals = expand_strategy_points(initial_entries, event_window_signals)
    add_on_signals = expand_strategy_points(add_on_entries, event_window_signals)
    sell_signals = event_window_signals.loc[event_window_signals["signal"].eq(2)].reset_index(
        drop=True)

    write_parquet(event_window_signals, output_dir / "event_signals.parquet")
    write_parquet(strategy_entry_signals, output_dir / "entry_points.parquet")
    write_parquet(result.executed_buys, output_dir / "executed_buys.parquet")
    write_parquet(initial_entry_signals, output_dir / "initial_entry_points.parquet")
    write_parquet(add_on_signals, output_dir / "add_on_points.parquet")
    write_parquet(sell_signals, output_dir / "sell_points.parquet")
    write_parquet(result.trades, output_dir / "trades.parquet")
    write_parquet(result.portfolio, output_dir / "portfolio.parquet")
    write_parquet(result.weights, output_dir / "weights.parquet")
    write_json(result.metrics, output_dir / "metrics.json")
    plot_event_signals(
        prices,
        events,
        event_window_signals,
        result.entry_points,
        result.executed_buys,
        asset_dir / "oscillation_event_buy_sell_entry_points.png",
        entry_parameters.event_window_days,
    )
    plot_portfolio_summary(
        result,
        asset_dir / "strategy_cumulative_returns_portfolio_holdings.png",
        strategy_parameters.maximum_holdings,
    )
    print({**result.metrics, "output": str(output_dir), "asset": str(asset_dir)})


if __name__ == "__main__":
    main()
