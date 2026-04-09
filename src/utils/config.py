"""
配置管理模块
支持YAML配置加载、合并、验证
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger


class ConfigManager:
    """工业级配置管理器，支持多环境配置、动态覆盖"""

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def load(cls, config_path: str, overrides: Optional[Dict[str, Any]] = None) -> "ConfigManager":
        """加载配置文件"""
        instance = cls()
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            instance._config = yaml.safe_load(f)

        if overrides:
            instance._merge_overrides(overrides)

        logger.info(f"✅ 配置加载完成: {config_path}")
        return instance

    def _merge_overrides(self, overrides: Dict[str, Any]):
        """深度合并覆盖配置"""
        def deep_merge(base: dict, override: dict) -> dict:
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        self._config = deep_merge(self._config, overrides)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的嵌套键 (e.g., 'model.vision_encoder.name')"""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def __repr__(self) -> str:
        return f"ConfigManager(keys={list(self._config.keys())})"


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent


def get_config(config_name: str = "base_config.yaml") -> ConfigManager:
    """快捷配置加载"""
    config_path = get_project_root() / "configs" / config_name
    return ConfigManager.load(str(config_path))


def setup_seed(seed: int = 42):
    """设置全局随机种子"""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"🎲 随机种子设置: {seed}")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """配置日志系统"""
    import sys
    logger.remove()

    # Windows 终端 GBK 编码下无法输出 emoji，使用 utf-8 安全输出
    sink = sys.stderr
    try:
        import io
        if hasattr(sys.stderr, "buffer"):
            sink = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

    logger.add(
        sink,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    if log_file:
        logger.add(log_file, rotation="100 MB", retention="30 days", level=log_level, encoding="utf-8")
    logger.info(f"📋 日志系统初始化完成, 级别: {log_level}")
