"""全局配置加载工具。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_CACHE: dict[str, Any] | None = None


def _find_config_path() -> Path:
    """从项目根目录查找 config/config.yaml。"""
    # 优先使用环境变量
    env = os.environ.get("MULTIAGENT_CONFIG")
    if env:
        return Path(env)
    # 默认路径：项目根/config/config.yaml
    root = Path(__file__).resolve().parent.parent
    return root / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """加载并缓存 YAML 配置。"""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and path is None:
        return _CONFIG_CACHE
    p = Path(path) if path else _find_config_path()
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path is None:
        _CONFIG_CACHE = cfg
    return cfg


def reset_config_cache() -> None:
    """重置配置缓存（测试用）。"""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
