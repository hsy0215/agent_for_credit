"""提供带固定来源前缀的项目日志。

日志格式为 `<source>:<config>\t<level>\t<message>`，便于多模型并行运行时检索。

"""

from __future__ import annotations

import logging


class LoggerBase:
    """自动注入模块名和配置名的日志对象。

    Args:
        source: 产生日志的模块或任务名。
        config_name: 当前配置变体名称。
    """

    def __init__(self, source: str, config_name: str = "default") -> None:
        """绑定日志来源和配置名称。"""
        self._source = f"{source}:{config_name}"
        self._logger = logging.getLogger("buy_point_project")

    def _write(self, level: int, message: str) -> None:
        """按固定三段格式写入一条日志。"""
        self._logger.log(level, "%s\t%s\t%s", self._source, logging.getLevelName(level), message)

    def info(self, message: str) -> None:
        """写入 INFO 日志。"""
        self._write(logging.INFO, message)

    def warn(self, message: str) -> None:
        """写入 WARNING 日志。"""
        self._write(logging.WARNING, message)

    def error(self, message: str) -> None:
        """写入 ERROR 日志。"""
        self._write(logging.ERROR, message)

    def debug(self, message: str) -> None:
        """写入 DEBUG 日志。"""
        self._write(logging.DEBUG, message)

    def trace(self, message: str) -> None:
        """以 DEBUG 等级写入带 TRACE 标识的日志。"""
        self._write(logging.DEBUG, f"TRACE {message}")


def configure_logging(level: str = "INFO") -> None:
    """配置应用统一日志格式。

    Args:
        level: Python 日志等级名称。
    """
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(message)s")
