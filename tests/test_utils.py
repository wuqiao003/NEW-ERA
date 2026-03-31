"""
工具模块测试
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import ConfigManager, get_project_root


class TestConfigManager:
    """配置管理器测试"""

    def test_load_config(self):
        config_path = os.path.join(get_project_root(), "configs", "base_config.yaml")
        if os.path.exists(config_path):
            cm = ConfigManager.load(config_path)
            assert cm.config is not None
            assert "model" in cm.config

    def test_get_nested_key(self):
        config_path = os.path.join(get_project_root(), "configs", "base_config.yaml")
        if os.path.exists(config_path):
            cm = ConfigManager.load(config_path)
            # 测试点号嵌套访问
            val = cm.get("model.fusion.type")
            assert val is not None

    def test_get_default(self):
        config_path = os.path.join(get_project_root(), "configs", "base_config.yaml")
        if os.path.exists(config_path):
            cm = ConfigManager.load(config_path)
            val = cm.get("nonexistent.key", "default_value")
            assert val == "default_value"

    def test_merge_overrides(self):
        config_path = os.path.join(get_project_root(), "configs", "base_config.yaml")
        if os.path.exists(config_path):
            overrides = {"model": {"fusion": {"type": "mlp"}}}
            cm = ConfigManager.load(config_path, overrides)
            assert cm.get("model.fusion.type") == "mlp"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ConfigManager.load("nonexistent_config.yaml")

    def test_project_root(self):
        root = get_project_root()
        assert root.exists()
