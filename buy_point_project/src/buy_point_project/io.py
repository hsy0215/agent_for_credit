"""提供 JSON、CSV 和 Parquet 的原子写入函数。

所有结果先写入同目录临时文件，再执行替换，避免任务中断后留下半文件。

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _temporary_path(path: Path) -> Path:
    """返回和目标文件位于同一目录的临时路径。"""
    return path.with_suffix(path.suffix + ".tmp")


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """以 zstd 压缩原子写入 Parquet。

    Args:
        frame: 待写出的表格。
        path: 目标 Parquet 路径。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """以 UTF-8 BOM 原子写入 CSV。

    Args:
        frame: 待写出的表格。
        path: 目标 CSV 路径。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def write_json(value: Any, path: Path) -> None:
    """以 UTF-8 原子写入 JSON。

    Args:
        value: 可被 JSON 序列化的对象。
        path: 目标 JSON 路径。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
