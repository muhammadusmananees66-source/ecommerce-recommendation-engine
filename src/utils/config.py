"""
Configuration loader.

Loads from an optional YAML file, then overlays environment variables
prefixed with RAG_ (e.g. RAG_RATE_LIMITING__REQUESTS_PER_MINUTE=200).
Environment variables always win over file config, which is the
standard 12-factor precedence.
"""

import os
from typing import Any

import yaml

ENV_PREFIX = "RAG_"


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from a YAML file and/or environment variables."""
    config: dict[str, Any] = {}

    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            loaded = yaml.safe_load(f)
            if loaded:
                config = loaded

    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        config_key = key[len(ENV_PREFIX):].lower()
        config[config_key] = _parse_env_value(value)

    return config


def _parse_env_value(value: str) -> Any:
    """Coerce a raw environment variable string to bool/int/float where possible."""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.lstrip("-").isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value