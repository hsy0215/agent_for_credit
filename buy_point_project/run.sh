#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
    echo "未找到 uv，请先在运行机器上安装 uv。" >&2
    exit 1
fi

# 根据 .python-version、pyproject.toml 和 uv.lock 创建或同步可复现的运行环境。
uv sync --frozen

# 完整流水线：基础数据生成 → 100模型训练预测 → 8模型信号 → 策略和图表。
uv run --no-sync python scripts/generate_style_manifest.py #因子分类，读取原始的因子分类文件之后按照收益来源划分为10个类别
# uv run --no-sync python scripts/generate_labels.py #计算并生成标签
uv run --no-sync src/main.py #100个模型训练和预测
uv run --no-sync python scripts/build_ensemble_buy_signals.py #筛选的8个模型的预测结果取平均之后筛选的买入和卖出信号
uv run --no-sync python scripts/run_entry_strategy.py #运行建仓，持仓和止盈止损策略
