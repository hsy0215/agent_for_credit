"""提供项目唯一的一键入口并编排完整模型预测流程。

"""

from __future__ import annotations

from buy_point_project.config import Config
from buy_point_project.data import DataLoader
from buy_point_project.log import LoggerBase, configure_logging
from buy_point_project.train import run_training


def main(config: Config | None = None) -> None:
    """加载配置并完成数据载入、模型训练、预测和结果保存。

    Args:
        config: 可选的已校验配置；不传时读取项目 ``config/config.yaml``。
    """
    configure_logging()
    resolved_config = config or Config.load()
    logger = LoggerBase("model_prediction", "pipeline")
    data_loader = DataLoader(resolved_config.data_config())
    data_loader.validate()
    resolved_config.output.mkdir(parents=True, exist_ok=True)
    run_training(resolved_config, data_loader, logger)
    logger.info(f"模型预测完成: {resolved_config.model_prediction_dir}")


if __name__ == "__main__":
    main()
