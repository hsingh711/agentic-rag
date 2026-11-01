"""Small configuration loader for the project.

All tunable values (chunk size, top_k, max_retries, model names, ...) live in
configs/config.yaml. Code should call ``load_config()`` instead of hard-coding them.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load the YAML configuration file and return it as a nested dictionary.

    Args:
        path: Optional path to a YAML file. Defaults to configs/config.yaml
            in the project root.

    Returns:
        The parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file does not contain a YAML mapping at the top level.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {config_path}")
    return config
