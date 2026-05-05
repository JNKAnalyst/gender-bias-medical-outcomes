"""Utility helpers for configuration loading and Spark session management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pyspark.sql import SparkSession


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_spark(app_name: str = "silent-disparities") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def resolve_path(path: str) -> str:
    if path.startswith("/dbfs/") or path.startswith("dbfs:/"):
        return path
    env_root = os.getenv("DATA_PATH")
    if env_root and not os.path.isabs(path):
        return str(Path(env_root) / path)
    return path
