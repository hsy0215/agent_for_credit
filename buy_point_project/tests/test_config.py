"""验证仓库默认配置的解析和 10×10 模型矩阵约束。

"""

from pathlib import Path

import pytest

from buy_point_project.config import Config


def test_repository_config_resolves_paths() -> None:
    """默认配置应解析到真实输入和唯一项目输出目录。"""
    project_dir = Path(__file__).resolve().parents[1]
    config = Config.from_yaml(project_dir / "config/config.yaml")

    assert config.output == project_dir / "output"
    assert config.model_prediction_dir == project_dir / "output/model_prediction"
    assert config.label_dir == project_dir / "output/label"
    assert config.input.factor_dir.is_dir()
    assert config.input.factor_classification_path is not None
    assert config.input.factor_classification_path.is_file()
    assert config.input.price_path.is_file()
    assert config.input.event_path is not None
    assert config.input.event_path.is_file()
    assert config.input.label_path == project_dir / "output/label"
    assert config.input.style_manifest_path == (project_dir /
                                                "output/data/a500_factor_style_manifest.csv")
    assert config.data_config().label_dir == project_dir / "output/label"
    assert config.model.target_column == "buy_label"
    assert len(config.model.styles) == 10
    assert len(config.model.time_specs) == 10


def test_load_is_the_single_default_and_environment_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """load 应读取默认配置，并允许只通过环境变量切换配置文件。"""
    project_dir = Path(__file__).resolve().parents[1]
    config_path = project_dir / "config/config.yaml"

    monkeypatch.delenv("BUY_POINT_CONFIG_PATH", raising=False)
    default_config = Config.load()
    assert default_config.output == project_dir / "output"

    monkeypatch.setenv("BUY_POINT_CONFIG_PATH", str(config_path))
    selected_config = Config.load()
    assert selected_config == default_config
