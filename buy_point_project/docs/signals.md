# 标签、买卖信号与策略

模型之后严格分为三阶段：第一阶段由 `src/main.py` 输出100模型连续预测；第二阶段由
`build_ensemble_buy_signals.py` 对固定8模型求均值并计算百分位，只输出四字段原始买卖
事件；第三阶段由 `run_entry_strategy.py` 结合震荡事件和建仓规则生成真正建仓点、加仓
点、回测结果及图表。`optimize_signals.py` 只保留为100模型逐模型和全模型均值的研究
基线，不是优选八模型正式策略链路的必要步骤。

## 标签生成

标签公式位于 `src/buy_point_project/label.py`，一键生成入口为：

```bash
uv run python scripts/generate_labels.py
```

脚本从 `input.price_path` 读取原始日行情的
`order_book_id, trade_date, close, prev_close`，先构造连续复权收盘价，再按日期写入
`input.label_path/YYYY-MM-DD.parquet`。`input.label_path` 在YAML中显式配置，默认指向
`output/label`，输出固定为：

```text
datetime, order_book_id, buy_label, sell_label
```

`buy_label` 对应内部公式 `buy_mild_time_penalty`。对日期 T 观察 T+1 至 T+20：

```text
future_max_return = max(close[T+1:T+20]) / close[T] - 1
future_drawdown = min(min(close[T+1:T+20]) / close[T] - 1, 0)
base = future_max_return + 0.7 × future_drawdown
time_penalty = 1 + 0.1 × log(1 + days_to_max - 1)
buy_label = base / time_penalty
```

最后 20 个交易日因未来窗口不足而为 NaN。`sell_label` 融合涨后横盘 `factory`、
波峰 `peak` 和未来回撤 `drawdown`；前两项在单股内做百分位归一化，三项取最大值后
乘以过去 30 日价格位置与前后 3 日局部价格位置的几何平均门控。两种标签都使用未来
信息，只能作为训练或评估目标。

## 100 模型买入与卖出信号

完整计算实现直接位于 `scripts/optimize_signals.py`，读取
`prediction/test/YYYY-MM-DD.parquet`。对每个
`model_id × order_book_id` 独立排序，当前预测的百分位只使用截至当天的最多 120
条预测：

```text
percent = rolling(120, min_periods=1).rank(method="average", pct=True)
buy_signal  = percent >= 0.80
sell_signal = percent <= 0.10
```

百分位按预测值从小到大计算，因此最低预测接近 0，最高预测接近 1。并列值使用平均
排名。测试期初历史不足 120 条时使用已发生的全部可用记录，不使用未来数据。

全量信号 schema：

```text
datetime, order_book_id, model_id, value, percent, buy_signal, sell_signal
```

买点和卖点文件都只保留 `datetime, order_book_id, model_id`。窗口与两个阈值均可由
命令行覆盖，默认分别为 120、0.80、0.10。

除逐模型结果外，脚本还会将同日同股的全部 100 个模型 `value` 取均值，写出
`ensemble_signals_*.parquet`。其中 `model_id=ensemble_mean`、`model_count` 记录
实际参与均值的模型数，`buy_prediction` 是集成连续预测值。这组结果作为全量模型基线。

## 优选 8 模型组合

`scripts/build_ensemble_buy_signals.py` 读取相同的 100 模型测试预测，但只组合经
[Calmar 排序和预测曲线走势复核](model/index.md#100-模型评估与优选-8-模型组合)固定的
8 个模型。每个同日同股必须集齐 8 个来源模型，然后对 `value` 取均值，再计算组合
预测自身的 120 日滚动百分位和 80%/10% 买卖信号。

脚本在内存中计算组合均值和百分位，最终只将达到买卖阈值的事件按目标交易日写入：

```text
output/ensemble_8_models/YYYY-MM-DD.parquet
```

每条组合事件展开成8行，`model_id` 分别保留八个实际来源模型；固定字段为：

```text
datetime, order_book_id, model_id, signal
```

`datetime` 统一到目标交易日09:30，`signal=1` 表示原始买点，`signal=2` 表示原始
卖点。没有事件的交易日仍保存空的四字段分区，以保证日期序列完整。策略默认直接读取
整个 `output/ensemble_8_models` 目录。

```bash
uv run python scripts/build_ensemble_buy_signals.py \
  --window 120 --buy-percentile 0.80 --sell-percentile 0.10
```

## 建仓与加仓策略

`scripts/run_entry_strategy.py` 将建仓策略拆成无状态脚本。
策略从配置中的 `input.event_path` 读取震荡事件，默认文件为
`output/data/oscillation_signals.xlsx`，至少需要 `trade_date, order_book_id` 两列。
通过 `--entry-filter-mode` 可以直接切换：

- `new`：当天为集成买入信号，且当前价格的近10日百分位不高于50%。
- `original`：相邻信号间隔不超过2个交易日视为一组；每组至少15个信号，只取组尾，
  并要求相邻确认买点间隔超过20个交易日。

`original` 的“组尾”必须等后续信号中断后才能确认。脚本为复现 Notebook 会把买点
标回组尾当天，因此它有回看偏差，只适合研究比较；`new` 模式的条件可在当天确认。

两种模式都只保留每只股票首个震荡事件日至后180个交易日。四字段事件不保存连续预测
值，因此同日候选按稳定的股票代码顺序补足槽位。默认最多持有10只，初始建仓5%；持有第5日再次出现
建仓信号时加仓5%，单票总仓位不超过10%。退出规则为亏损5%、盈利10%或持有30个
交易日全仓退出；盈利7%时先减半。完整交易和净值均落盘，不依赖绘图单元。
`event_signals.parquet`、`entry_points.parquet`、`initial_entry_points.parquet`、
`add_on_points.parquet` 和 `sell_points.parquet` 均使用相同的四字段事件格式。卖点只
落盘并显示在事件图上，当前不参与策略退出计算。事件买卖/建仓图和
`Strategy Cumulative Returns & Portfolio Holdings` 图写入项目 `asset/`。

```bash
uv run python scripts/run_entry_strategy.py --entry-filter-mode new
```

## 与 sell_label 的区别

这里的 `sell_signal` 是**模型预测值的后处理规则**：同一买点模型的预测值落入自身
120 日分布最低 10% 时触发。`sell_label` 则是上文说明的价格监督目标，使用未来
价格形态计算。
当前 100 个模型直接读取 `input.label_path` 中的 `buy_label` 训练；所以预测低 10% 的
`sell_signal` 与未来价格计算的 `sell_label` 不能视为同一列。
