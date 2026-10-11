from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_config() -> dict:
    # .env supplies secrets (e.g. NAPI_API_KEY) without touching config.yaml.
    load_dotenv(ENV_PATH)
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


CONFIG = load_config()