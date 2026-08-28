"""验证从 cell9 拆出的建仓筛选和第五日加仓策略。

"""

import pandas as pd
from scripts.run_entry_strategy import (
    EntryParameters,
    StrategyParameters,
    build_entry_points,
    plot_event_signals,
    plot_portfolio_summary,
    read_events,
    run_strategy,
)


def _strategy_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """构造低位价格、两个候选日和一个事件的小样本。"""
    dates = pd.bdate_range("2024-01-01", periods=25, name="trade_date")
    stock = "000001.XSHE"
    prices = pd.DataFrame({stock: [10.0 - index * 0.01 for index in range(25)]}, index=dates)
    signals = pd.DataFrame({
        "date": [dates[10], dates[15]],
        "order_book_id": [stock, stock],
        "buy_prediction": [0.8, 0.9],
        "buy_signal": [True, True],
    })
    events = pd.DataFrame({"trade_date": [dates[0]], "order_book_id": [stock]})
    return prices, signals, events


def test_new_entry_filter_and_fifth_day_add_on() -> None:
    """新模式应选中10日低位，第二个信号恰在持有第五日触发加仓。"""
    prices, signals, events = _strategy_inputs()
    entry_points, entry_matrix, prediction_matrix = build_entry_points(
        signals,
        prices,
        events,
        EntryParameters(mode="new"),
    )
    result = run_strategy(
        prices,
        entry_points,
        entry_matrix,
        prediction_matrix,
        StrategyParameters(maximum_holding_days=100),
    )

    assert len(entry_points) == 2
    assert result.executed_buys["buy_type"].tolist() == ["initial", "add"]
    assert result.metrics["initial_buys"] == 1
    assert result.metrics["add_on_buys"] == 1


def test_original_entry_filter_takes_long_group_tail() -> None:
    """原模式应把连续15个以上候选信号归组并只取最后一天。"""
    prices, _, events = _strategy_inputs()
    dates = prices.index[:20]
    signals = pd.DataFrame({
        "date": dates,
        "order_book_id": ["000001.XSHE"] * len(dates),
        "buy_prediction": range(len(dates)),
        "buy_signal": [True] * len(dates),
    })
    entry_points, _, _ = build_entry_points(
        signals,
        prices,
        events,
        EntryParameters(mode="original"),
    )

    assert len(entry_points) == 1
    assert entry_points.iloc[0]["date"] == dates[-1]


def test_read_events_supports_current_xlsx_format(tmp_path) -> None:
    """事件读取器应能直接读取 Notebook 当前使用的 xlsx 格式。"""
    path = tmp_path / "events.xlsx"
    pd.DataFrame({
        "trade_date": ["2024-01-02", "2024-02-02"],
        "order_book_id": ["000001.XSHE", "000001.XSHE"],
    }).to_excel(path, index=False)

    events = read_events(path, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))

    assert len(events) == 1
    assert events.iloc[0]["trade_date"] == pd.Timestamp("2024-01-02")


def test_strategy_plots_are_written(tmp_path) -> None:
    """策略应在无界面环境写出事件图和累计收益持仓图。"""
    prices, signals, events = _strategy_inputs()
    entry_points, entry_matrix, prediction_matrix = build_entry_points(
        signals, prices, events, EntryParameters(mode="new"))
    result = run_strategy(
        prices,
        entry_points,
        entry_matrix,
        prediction_matrix,
        StrategyParameters(maximum_holding_days=100),
    )
    model_ids = ["rf_momentum_T02", "rf_reversal_T04"]
    raw_events = pd.DataFrame([{
        "datetime": date + pd.Timedelta(hours=9, minutes=30),
        "order_book_id": "000001.XSHE",
        "model_id": model_id,
        "signal": signal,
    } for date, signal in [(prices.index[10], 1), (prices.index[20], 2)] for model_id in model_ids])
    event_plot = tmp_path / "event.png"
    portfolio_plot = tmp_path / "portfolio.png"

    plot_event_signals(
        prices,
        events,
        raw_events,
        result.entry_points,
        result.executed_buys,
        event_plot,
        180,
    )
    plot_portfolio_summary(result, portfolio_plot, maximum_holdings=10)

    assert event_plot.stat().st_size > 0
    assert portfolio_plot.stat().st_size > 0
