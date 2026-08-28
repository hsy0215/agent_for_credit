"""计算的买入与卖出目标标签。

本模块计算交付字段 `buy_label` 对应的内部 `buy_mild_time_penalty`，以及用户指定的
`sell_label`。前者衡量未来 20 日收益、回撤与到达高点速度；后者融合涨后横盘、
波峰和未来回撤三种卖点形状。标签使用未来信息，只能用于训练或评估。

"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

LabelProgressCallback = Callable[[int, int, str], None]
"""标签计算进度回调：已完成股票数、股票总数和当前股票代码。"""


def finite(value: Any) -> bool:
    """判断输入是否可转换为有限浮点数。"""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def clamp01(value: Any) -> float:
    """把有限数值裁剪至 [0, 1]，无效值按 0 处理。"""
    if not finite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def bounded_norm(value: Any) -> float:
    """用 x/(1+x) 将正的无界强度映射到 [0, 1)。"""
    if not finite(value):
        return 0.0
    number = float(value)
    if number <= 0:
        return 0.0
    return number / (1.0 + number)


def safe_div(numerator: float, denominator: float) -> float:
    """执行安全除法，输入无效或分母过小时返回 NaN。"""
    if (not finite(numerator) or not finite(denominator) or abs(float(denominator)) <= 1e-12):
        return np.nan
    return float(numerator) / float(denominator)


def median_abs(values: list[float]) -> float:
    """返回绝对收益列表的上中位数，空列表返回 NaN。"""
    if not values:
        return np.nan
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2])


def bidirectional_median_sigma(close: np.ndarray, lookback: int = 30) -> np.ndarray:
    """计算前后各半个窗口绝对 log return 的稳健波动率。"""
    size = len(close)
    half = max(1, int(lookback) // 2)
    log_returns = np.full(size, np.nan, dtype="float64")
    valid = (np.isfinite(close[1:]) & np.isfinite(close[:-1]) & (close[1:] > 0) & (close[:-1] > 0))
    log_returns[1:][valid] = np.log(close[1:][valid] / close[:-1][valid])
    sigma = np.full(size, np.nan, dtype="float64")
    for index in range(size):
        start = max(1, index - half)
        end = min(size - 1, index + half)
        values = [
            abs(float(log_returns[position])) for position in range(start, end + 1)
            if position != index and finite(log_returns[position])
        ]
        if values:
            sigma[index] = 1.4826 * median_abs(values)
    return sigma


def trailing_median_sigma(close: np.ndarray, lookback: int = 30) -> np.ndarray:
    """计算只含当日及过去收益的稳健波动率。"""
    size = len(close)
    log_returns = np.full(size, np.nan, dtype="float64")
    valid = (np.isfinite(close[1:]) & np.isfinite(close[:-1]) & (close[1:] > 0) & (close[:-1] > 0))
    log_returns[1:][valid] = np.log(close[1:][valid] / close[:-1][valid])
    sigma = np.full(size, np.nan, dtype="float64")
    for index in range(1, size):
        start = max(1, index - max(1, int(lookback)) + 1)
        values = [abs(float(value)) for value in log_returns[start:index + 1] if finite(value)]
        if values:
            sigma[index] = 1.4826 * median_abs(values)
    return sigma


def threshold_at(sigma: np.ndarray, index: int, multiplier: float, delta_floor: float) -> float:
    """返回 max(multiplier×sigma, delta_floor) 的 swing 阈值。"""
    value = sigma[index]
    if not finite(value) or value <= 0:
        return np.nan
    return max(float(multiplier) * float(value), float(delta_floor or 0.0))


def segment_median_sigma(close: np.ndarray, start_index: int, end_index: int) -> float:
    """计算一个闭区间价格片段的绝对收益稳健波动率。"""
    values: list[float] = []
    start = max(1, int(start_index) + 1)
    end = min(len(close) - 1, int(end_index))
    for index in range(start, end + 1):
        previous = close[index - 1]
        current = close[index]
        if finite(previous) and finite(current) and previous > 0 and current > 0:
            values.append(abs(math.log(float(current) / float(previous))))
    sigma = 1.4826 * median_abs(values) if values else np.nan
    return sigma if finite(sigma) and sigma > 0 else np.nan


def average_confirmed_swing_duration(close: np.ndarray, sigma: np.ndarray, multiplier: float,
                                     delta_floor: float) -> float:
    """通过自适应阈值估计相邻已确认 swing 转折的平均间隔。"""
    if len(close) <= 1:
        return np.nan
    turns: list[tuple[str, int]] = []
    direction: str | None = None
    high = low = extreme = float(close[0])
    high_index = low_index = extreme_index = 0
    for index in range(1, len(close)):
        price = close[index]
        delta = threshold_at(sigma, index, multiplier, delta_floor)
        if not finite(price) or price <= 0 or not finite(delta) or delta <= 0:
            continue
        price = float(price)
        if direction is None:
            if price >= high:
                high, high_index = price, index
            if price <= low:
                low, low_index = price, index
            if high > 0 and (high - price) / high >= delta:
                turns.append(("high", high_index))
                direction = "down"
                extreme, extreme_index = price, index
            elif low > 0 and (price - low) / low >= delta:
                turns.append(("low", low_index))
                direction = "up"
                extreme, extreme_index = price, index
            continue
        if direction == "up":
            if price >= extreme:
                extreme, extreme_index = price, index
            elif (extreme - price) / extreme >= delta:
                turns.append(("high", extreme_index))
                direction = "down"
                extreme, extreme_index = price, index
        elif price <= extreme:
            extreme, extreme_index = price, index
        elif (price - extreme) / extreme >= delta:
            turns.append(("low", extreme_index))
            direction = "up"
            extreme, extreme_index = price, index
    durations = [
        turns[index + 1][1] - turns[index][1] for index in range(len(turns) - 1)
        if turns[index + 1][1] - turns[index][1] > 0
    ]
    return float(np.mean(durations)) if durations else np.nan


def flat_days_from(close: np.ndarray, start_index: int, sigma: np.ndarray, multiplier: float,
                   delta_floor: float) -> float:
    """统计起点以后价格保持在自适应窄幅带内的连续天数。"""
    start_price = close[start_index]
    delta = threshold_at(sigma, start_index, multiplier, delta_floor)
    if (not finite(start_price) or start_price <= 0 or not finite(delta) or delta <= 0):
        return np.nan
    low_band = float(start_price) * (1.0 - delta)
    high_band = float(start_price) * (1.0 + delta)
    days = 0
    for index in range(start_index + 1, len(close)):
        price = close[index]
        if not finite(price) or price <= 0:
            continue
        if price < low_band or price > high_band:
            break
        days = index - start_index
    return float(days)


def percentile_normalize(raw: np.ndarray) -> np.ndarray:
    """将单股内部的正分数按从小到大排名映射到 (0, 1]。"""
    values = np.asarray(raw, dtype="float64")
    result = np.zeros(len(values), dtype="float64")
    mask = np.isfinite(values) & (values > 0)
    if not mask.any():
        return result
    result[mask] = pd.Series(values[mask]).rank(method="max", pct=True).to_numpy(dtype="float64")
    return result


def future_return_drawdown_20(close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算 T+1~T+20 最大收益、相对 T 最大回撤和到高点天数。"""
    prices = np.asarray(close, dtype="float64")
    size = len(prices)
    future_return = np.full(size, np.nan, dtype="float64")
    future_drawdown = np.full(size, np.nan, dtype="float64")
    days_to_max = np.full(size, np.nan, dtype="float64")
    if size <= 20:
        return future_return, future_drawdown, days_to_max
    windows = np.lib.stride_tricks.sliding_window_view(prices[1:], 20)
    valid = np.isfinite(windows).all(axis=1) & (windows > 0).all(axis=1)
    valid &= np.isfinite(prices[:size - 20]) & (prices[:size - 20] > 0)
    positions = np.arange(size - 20)[valid]
    valid_windows = windows[valid]
    starts = prices[:size - 20][valid]
    maximum_positions = np.argmax(valid_windows, axis=1)
    future_return[positions] = (
        valid_windows[np.arange(len(positions)), maximum_positions] / starts - 1.0)
    future_drawdown[positions] = np.minimum(valid_windows.min(axis=1) / starts - 1.0, 0.0)
    days_to_max[positions] = maximum_positions + 1
    return future_return, future_drawdown, days_to_max


def compute_buy_mild_time_penalty(close: np.ndarray) -> np.ndarray:
    """计算模型训练使用的温和时间惩罚买入标签。

    公式为 `(future_max_return + 0.7×future_drawdown) /
    (1 + 0.1×log(1 + days_to_max - 1))`。未来窗口不足 20 日的尾部为 NaN。
    """
    future_return, future_drawdown, days_to_max = future_return_drawdown_20(close)
    days = np.maximum(days_to_max, 1.0)
    base = future_return + 0.7 * future_drawdown
    time_penalty = 1.0 + 0.1 * np.log1p(days - 1.0)
    return np.divide(
        base,
        time_penalty,
        out=np.full(len(close), np.nan, dtype="float64"),
        where=np.isfinite(base) & np.isfinite(time_penalty),
    )


def compute_sell_label(close: np.ndarray,
                       multiplier: float = 2.5,
                       smooth_days: float = 1.0,
                       delta_floor: float = 0.0) -> np.ndarray:
    """计算涨后横盘、波峰、后续回撤融合的卖出标签。

    factory 与 peak 在单股内做百分位归一化，drawdown 保留绝对强度；三者取最大值
    后乘以过去 30 日位置和前后 3 日局部位置的几何平均门控。
    """
    prices = np.asarray(close, dtype="float64")
    size = len(prices)
    sigma = bidirectional_median_sigma(prices, 30)
    trailing_sigma = trailing_median_sigma(prices, 30)
    average_duration = average_confirmed_swing_duration(prices, sigma, multiplier, delta_floor)
    if not finite(average_duration):
        average_duration = 10.0

    horizon = 30
    factory_raw = np.zeros(size, dtype="float64")
    drawdown = np.zeros(size, dtype="float64")
    for index in range(size):
        price = prices[index]
        delta = threshold_at(sigma, index, multiplier, delta_floor)
        if not finite(price) or price <= 0:
            continue
        price = float(price)
        past = prices[max(0, index - horizon):index + 1]
        past = past[np.isfinite(past) & (past > 0)]
        drawdown_past = prices[max(0, index - horizon + 1):index + 1]
        drawdown_past = drawdown_past[np.isfinite(drawdown_past) & (drawdown_past > 0)]
        if len(past) >= 2 and finite(delta) and delta > 0:
            past_low = float(past.min())
            past_high = float(past.max())
            past_gain = price / past_low - 1.0 if past_low > 0 else 0.0
            position = ((price - past_low) /
                        (past_high - past_low) if past_high > past_low else 0.0)
            flat_days = flat_days_from(prices, index, sigma, multiplier, delta_floor)
            if finite(flat_days) and flat_days > 0:
                flat_end = min(size - 1, index + int(flat_days))
                flat_prices = prices[index:flat_end + 1]
                flat_prices = flat_prices[np.isfinite(flat_prices) & (flat_prices > 0)]
                flat_range = (float(flat_prices.max() / flat_prices.min() -
                                    1.0) if len(flat_prices) >= 2 else 0.0)
                gain_norm = clamp01(past_gain / max(3.0 * delta, 1e-12))
                position_norm = clamp01((position - 0.5) / 0.5)
                flat_norm = clamp01(float(flat_days) / max(average_duration, 5.0))
                tight_norm = 1.0 - min(1.0, flat_range / max(2.0 * delta, 1e-12))
                factory_raw[index] = clamp01(
                    (0.32 * gain_norm + 0.28 * position_norm + 0.25 * flat_norm + 0.15 * tight_norm)
                    * max(gain_norm, position_norm))

        end = min(size - 1, index + horizon)
        low = price
        low_index = index
        for future_index in range(index + 1, end + 1):
            future_price = prices[future_index]
            if finite(future_price) and future_price > 0 and future_price < low:
                low = float(future_price)
                low_index = future_index
        prior_sigma = trailing_sigma[index]
        if (low_index > index and len(drawdown_past) >= 2 and finite(prior_sigma)
                and prior_sigma > 0):
            future_drop = (price - low) / price
            drop_days = low_index - index
            past_median = float(np.sort(drawdown_past)[len(drawdown_past) // 2])
            drop_from_past = max(0.0, (past_median - low) / past_median)
            volatility_scale = max(multiplier * float(prior_sigma), 1e-12)
            magnitude_norm = (0.55 * bounded_norm(future_drop / volatility_scale) +
                              0.45 * bounded_norm(drop_from_past / volatility_scale))
            drop_speed = safe_div(
                future_drop,
                float(prior_sigma) * math.sqrt(drop_days + smooth_days),
            )
            drawdown[index] = clamp01(0.65 * magnitude_norm + 0.35 * bounded_norm(drop_speed))

    peak_raw = np.zeros(size, dtype="float64")

    def peak_delta_at(index: int) -> float:
        """优先返回只看历史的峰值确认阈值。"""
        trailing_delta = threshold_at(trailing_sigma, index, multiplier, 0.0)
        if finite(trailing_delta) and trailing_delta > 0:
            return trailing_delta
        return threshold_at(sigma, index, multiplier, 0.0)

    turns: list[tuple[str, int]] = []
    if size:
        direction: str | None = None
        extreme = initial_high = initial_low = float(prices[0])
        extreme_index = initial_high_index = initial_low_index = 0
        for index in range(1, size):
            price = prices[index]
            if (not finite(price) or price <= 0 or not finite(extreme) or extreme <= 0):
                continue
            price = float(price)
            if direction is None:
                if price >= initial_high:
                    initial_high, initial_high_index = price, index
                if price <= initial_low:
                    initial_low, initial_low_index = price, index
                high_delta = peak_delta_at(initial_high_index)
                low_delta = peak_delta_at(initial_low_index)
                if (finite(high_delta) and high_delta > 0
                        and (initial_high - price) / initial_high >= high_delta):
                    turns.append(("high", initial_high_index))
                    direction = "down"
                    extreme, extreme_index = price, index
                elif (finite(low_delta) and low_delta > 0
                      and (price - initial_low) / initial_low >= low_delta):
                    turns.append(("low", initial_low_index))
                    direction = "up"
                    extreme, extreme_index = price, index
                continue
            if direction == "up":
                if price >= extreme:
                    extreme, extreme_index = price, index
                else:
                    delta = peak_delta_at(extreme_index)
                    if finite(delta) and (extreme - price) / extreme >= delta:
                        turns.append(("high", extreme_index))
                        direction = "down"
                        extreme, extreme_index = price, index
            elif price <= extreme:
                extreme, extreme_index = price, index
            else:
                delta = peak_delta_at(extreme_index)
                if finite(delta) and (price - extreme) / extreme >= delta:
                    turns.append(("low", extreme_index))
                    direction = "up"
                    extreme, extreme_index = price, index

    radius = 3
    decay = 2.0
    peak_candidates: dict[int, int] = {}

    def add_peak_candidate(high_index: int, low_index: int | None) -> None:
        """为同一高点保留后续价格更低的候选低点。"""
        if low_index is None or low_index <= high_index:
            return
        existing = peak_candidates.get(high_index)
        if existing is None or prices[low_index] < prices[existing]:
            peak_candidates[high_index] = low_index

    for turn_index, (kind, high_index) in enumerate(turns):
        if kind != "high":
            continue
        candidate_low_index: int | None = None
        for next_kind, next_index in turns[turn_index + 1:]:
            if next_kind == "low":
                candidate_low_index = next_index
                break
        if candidate_low_index is None:
            terminal_end = min(size - 1, high_index + horizon)
            for index in range(high_index + 1, terminal_end + 1):
                price = prices[index]
                if (finite(price) and price > 0
                        and (candidate_low_index is None or price < prices[candidate_low_index])):
                    candidate_low_index = index
        add_peak_candidate(high_index, candidate_low_index)

    for high_index in range(radius, max(radius, size - radius)):
        high = prices[high_index]
        if not finite(high) or high <= 0:
            continue
        neighborhood = prices[high_index - radius:high_index + radius + 1]
        neighborhood = neighborhood[np.isfinite(neighborhood) & (neighborhood > 0)]
        if (len(neighborhood) < 2 * radius + 1 or high != neighborhood.max()
                or high == neighborhood.min()):
            continue
        candidate_low_index = None
        terminal_end = min(size - 1, high_index + horizon)
        for index in range(high_index + 1, terminal_end + 1):
            price = prices[index]
            if (finite(price) and price > 0
                    and (candidate_low_index is None or price < prices[candidate_low_index])):
                candidate_low_index = index
        if candidate_low_index is not None and prices[candidate_low_index] < high:
            add_peak_candidate(high_index, candidate_low_index)

    for high_index, low_index in peak_candidates.items():
        high = prices[high_index]
        low = prices[low_index]
        if not finite(high) or not finite(low) or high <= 0:
            continue
        loss = (float(high) - float(low)) / float(high)
        days = low_index - high_index
        segment_sigma = segment_median_sigma(prices, high_index, low_index)
        peak_speed = safe_div(loss, segment_sigma * (days + smooth_days))
        history_sigma = trailing_sigma[high_index]
        fallback_sigma = sigma[high_index]
        peak_sigma = (history_sigma
                      if finite(history_sigma) and history_sigma > 0 else fallback_sigma)
        base = clamp01(0.65 * bounded_norm(3.0 * peak_speed) +
                       0.35 * clamp01(loss / max(4.0 * peak_sigma, 1e-12)))
        if base <= 0:
            continue
        for index in range(max(0, high_index - radius), high_index + 1):
            score = clamp01(base * math.exp(-(high_index - index) / decay))
            if score > peak_raw[index]:
                peak_raw[index] = score

    factory = percentile_normalize(factory_raw)
    peak = percentile_normalize(peak_raw)
    raw_maximum = np.maximum.reduce([factory, peak, drawdown])
    label = np.zeros(size, dtype="float64")
    for index in range(size):
        price = prices[index]
        recent = prices[max(0, index - 29):index + 1]
        recent = recent[np.isfinite(recent) & (recent > 0)]
        local = prices[max(0, index - 3):min(size, index + 4)]
        local = local[np.isfinite(local) & (local > 0)]
        if not finite(price) or price <= 0:
            continue
        recent_low = float(recent.min()) if len(recent) else float(price)
        recent_high = float(recent.max()) if len(recent) else float(price)
        local_low = float(local.min()) if len(local) else float(price)
        local_high = float(local.max()) if len(local) else float(price)
        recent_position = (clamp01((float(price) - recent_low) /
                                   (recent_high - recent_low)) if recent_high > recent_low else 0.0)
        local_position = (clamp01((float(price) - local_low) /
                                  (local_high - local_low)) if local_high > local_low else 0.0)
        label[index] = raw_maximum[index] * math.sqrt(recent_position * local_position)
    return label


def generate_labels(
    frame: pd.DataFrame,
    progress_callback: LabelProgressCallback | None = None,
) -> pd.DataFrame:
    """按股票生成 `buy_mild_time_penalty` 和 `sell_label`。

    Args:
        frame: 含 order_book_id、trade_date、adjusted_close 的日频价格表。
        progress_callback: 可选进度回调，在每只股票完成后调用。

    Returns:
        保留三个输入字段并追加两列 float32 标签的有序表。

    Raises:
        KeyError: 缺少必需字段。
        ValueError: 存在重复股票交易日。
    """
    required = {"order_book_id", "trade_date", "adjusted_close"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"标签输入缺少字段: {sorted(missing)}")
    source = frame[["order_book_id", "trade_date", "adjusted_close"]].copy()
    source["order_book_id"] = source["order_book_id"].astype(str)
    source["trade_date"] = pd.to_datetime(source["trade_date"]).dt.normalize()
    source["adjusted_close"] = pd.to_numeric(source["adjusted_close"], errors="coerce")
    duplicate_count = int(source.duplicated(["order_book_id", "trade_date"]).sum())
    if duplicate_count:
        raise ValueError(f"标签输入存在 {duplicate_count} 条重复股票交易日")
    parts = []
    grouped = source.groupby("order_book_id", sort=False)
    total_stocks = grouped.ngroups
    for completed, (order_book_id, group) in enumerate(grouped, start=1):
        ordered = group.sort_values("trade_date", kind="stable").copy()
        close = ordered["adjusted_close"].to_numpy(dtype="float64")
        ordered["buy_mild_time_penalty"] = compute_buy_mild_time_penalty(close).astype("float32")
        ordered["sell_label"] = compute_sell_label(close).astype("float32")
        parts.append(ordered)
        if progress_callback is not None:
            progress_callback(completed, total_stocks, str(order_book_id))
    if not parts:
        return pd.DataFrame(columns=[
            "order_book_id", "trade_date", "adjusted_close", "buy_mild_time_penalty", "sell_label"
        ])
    return pd.concat(parts, ignore_index=True).sort_values(["trade_date", "order_book_id"],
                                                           kind="stable").reset_index(drop=True)
