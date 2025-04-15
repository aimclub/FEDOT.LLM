import logging
import re
from copy import deepcopy
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, List, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf


def _get_default_config_path(
    presets: str,
) -> Path:
    """
    Get default config folder under package root
    Returns Path to the config.yaml file
    """
    try:
        traversable_path = files("fedotllm") / "configs" / f"{presets}.yaml"

        with as_file(traversable_path) as actual_path:
            if not actual_path.exists():
                raise ValueError(
                    f"Config file not found at expected location: {actual_path}\n"
                    "Please ensure the config files are properly installed in the configs directory."
                )
            return Path(str(actual_path))
    except Exception:
        # Fallback for development environment
        package_root = Path(__file__).parent.parent
        config_path = package_root / "configs" / f"{presets}.yaml"
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")
        return config_path


def parse_override(override: str) -> tuple:
    """
    Parse a single override string in the format 'key=value' or 'key.nested=value'

    Args:
        override: String in format "key=value" or "key.nested=value"

    Returns:
        Tuple of (key, value)

    Raises:
        ValueError: If override string is not in correct format
    """
    if "=" not in override:
        raise ValueError(
            f"Invalid override format: {override}. Must be in format 'key=value' or 'key.nested=value'"
        )
    key, value = override.split("=", 1)
    return key, value


def apply_overrides(
    config: DictConfig | ListConfig, overrides: List[str]
) -> DictConfig | ListConfig:
    """
    Apply command-line overrides to config
    Args:
        config: Base configuration
        overrides: List of overrides in format ["key1=value1", "key2.nested=value2"]
    Returns:
        Updated configuration
    """
    if not overrides:
        return config

    # Convert overrides to nested dict
    override_conf: dict[Any, Any] = {}
    # Split by comma but preserve commas inside square brackets
    overrides = re.split(r",(?![^\[]*\])", ",".join(overrides))

    for override in overrides:
        override = override.strip()
        key, value = parse_override(override)

        # Handle list values enclosed in square brackets
        if value.startswith("[") and value.endswith("]"):
            # Extract items between brackets and split by comma
            items = value[1:-1].split(",")
            # Clean up each item and convert to list
            value = [item.strip() for item in items if item.strip()]
        else:
            # Try to convert value to appropriate type for non-list values
            try:
                value = eval(value)
            except Exception:
                # Keep as string if eval fails
                pass

        # Handle nested keys
        current = override_conf
        key_parts = key.split(".")
        print(f"DEBUG: key_parts: {key_parts}")
        print(f"DEBUG: current: {current}")
        for part in key_parts[:-1]:
            current = current.setdefault(part, {})
            print(f"DEBUG: current: {current}, part: {part}")
        current[key_parts[-1]] = value
        print(f"DEBUG: current: {current}")

    # Convert override dict to OmegaConf and merge
    new_conf = OmegaConf.merge(config, OmegaConf.create(override_conf))
    assert isinstance(new_conf, DictConfig), (
        f"New config is in unsupported type: {type(new_conf)}"
    )
    return new_conf


def load_config(
    presets: str = "default",
    config_path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """
    Load configuration from yaml file, merging with default config and applying overrides

    Args:
        config_path: Optional path to config file. If provided, will merge with and override default config
        overrides: Optional list of command-line overrides in format ["key1=value1", "key2.nested=value2"]

    Returns:
        Loaded and merged configuration

    Raises:
        ValueError: If config file not found or invalid
    """
    # Load default config
    default_config_path = _get_default_config_path(presets="default")
    logging.info(f"Loading default config from: {default_config_path}")
    config = OmegaConf.load(default_config_path)

    # Apply Presets
    if presets != "default":
        presets_config_path = _get_default_config_path(presets=presets)
        presets_config = OmegaConf.load(presets_config_path)
        logging.info(f"Merging {presets} config from: {presets_config_path}")
        config = OmegaConf.merge(config, presets_config)

    # If custom config provided, merge it
    if config_path:
        custom_config_path = Path(config_path)
        if not custom_config_path.is_file():
            raise ValueError(f"Custom config file not found at: {custom_config_path}")

        logging.info(f"Loading custom config from: {custom_config_path}")
        custom_config = OmegaConf.load(custom_config_path)
        config = OmegaConf.merge(config, custom_config)
        logging.info("Successfully merged custom config with default config")

    # Apply command-line overrides if any
    if overrides:
        logging.info(f"Applying command-line overrides: {overrides}")
        config = apply_overrides(config, overrides)
        logging.info("Successfully applied command-line overrides")

    assert isinstance(config, DictConfig), (
        f"Config has not supported type: {type(config)}"
    )
    return config


def unpack_omega_config(config):
    temp_config = deepcopy(config)
    dict_config = OmegaConf.to_container(temp_config, resolve=True)
    return dict_config
