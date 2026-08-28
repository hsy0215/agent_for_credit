# buy_point_project

从 `buy_and_sell/model/tree_model/model.ipynb` 拆出的单项目版本，只复现方案B：A500
`T-5..T-1` 五日时序窗口、10 风格 × 10 时间方案的 100 个随机森林模型。模型不拆
validation，每个模型使用完整 train 拟合一次并预测 train/test。

完整流程、输入输出和模型说明见 [docs/index.md](docs/index.md)。

首次运行或生产运行直接执行：

```bash
./run.sh
```

运行前需要准备 `output/data` 数据包，其中包含A500因子、日行情、因子分类表和震荡
事件；脚本不会生成这些外部源数据。

脚本先用 `uv sync --frozen` 根据锁文件创建或同步 `.venv`，随后依次执行
完整流水线。运行机器需要预先安装 `uv`，不需要手动创建、激活虚拟环境。

```bash
uv sync --frozen
uv run python scripts/generate_style_manifest.py
uv run python scripts/generate_labels.py
uv run src/main.py
uv run python scripts/optimize_signals.py \
  --window 120 --buy-percentile 0.80 --sell-percentile 0.10
uv run python scripts/build_ensemble_buy_signals.py \
  --window 120 --buy-percentile 0.80 --sell-percentile 0.10
uv run python scripts/run_entry_strategy.py --entry-filter-mode new
```

优选八模型买卖事件按日写入 `output/ensemble_8_models/YYYY-MM-DD.parquet`；策略图表
写入 `asset/`。

所有Python入口统一调用 `Config.load()`；默认读取 `config/config.yaml`。如需切换配置，
只设置 `BUY_POINT_CONFIG_PATH` 环境变量，不使用命令行配置参数：

```bash
BUY_POINT_CONFIG_PATH=config/config_variant.yaml ./run.sh
```

模型预测的外层调用方式为：

```python
from main import main

main()
```

代码质量检查：`bash scripts/check.sh`。API 文档生成：
`uv run pdoc -o docs/api buy_point_project`。

项目使用 uv 管理 Python 环境
