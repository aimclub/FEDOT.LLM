import logging
import re
from importlib.resources import files
from pathlib import Path
from typing import List, Optional, Type, TypeVar

from dotenv import load_dotenv
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel

from fedotllm.configs.schema import AppConfig


def _get_default_config_path(
    presets: str,
) -> Path:
    """
    Get the path to a config YAML file under the package's configs directory.

    Args:
        presets: Name of the preset config (without .yaml extension).

    Returns:
        Path to the config YAML file.

    Raises:
        ValueError: If the config file is not found.
    """
    try:
        config_path = Path(files("fedotllm") / "configs" / f"{presets}.yaml")

        if not config_path.exists():
            raise ValueError(
                f"Config file not found at expected location: {config_path}\n"
                "Please ensure the config files are properly installed in the configs directory."
            )
        return config_path
    except Exception:
        # Fallback for development environment
        package_root = Path(__file__).parent.parent
        config_path = package_root / "configs" / f"{presets}.yaml"
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")
        return config_path


def parse_override(override: str) -> tuple:
    """
    Parse a single override string in the format 'key=value' or 'key.nested=value'.

    Args:
        override: String in format "key=value" or "key.nested=value".

    Returns:
        Tuple of (key, value).

    Raises:
        ValueError: If override string is not in correct format.
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
    Apply command-line style overrides to a configuration dictionary.

    Args:
        config: Base configuration as a dictionary.
        overrides: List of overrides in format ["key1=value1", "key2.nested=value2"].

    Returns:
        Updated configuration dictionary with overrides applied.
    """
    if not overrides:
        return config

    # Convert overrides to nested dict
    override_conf = {}
    # Split by comma but preserve commas inside square brackets
    overrides = re.split(pattern=r",(?![^\[]*\])", string=",".join(overrides))

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
        for part in key_parts[:-1]:
            current = current.setdefault(part, {})
        current[key_parts[-1]] = value

    # Convert override dict to OmegaConf and merge
    override_conf = OmegaConf.create(override_conf)
    return OmegaConf.merge(config, override_conf)


def _path_resolver(path: str | Path):
    """
    Resolve a config path, supporting special 'fedotllm:xxx' syntax for package configs.

    Args:
        path: Path string, possibly with 'fedotllm:' prefix.

    Returns:
        Path object to the resolved config file.
    """
    match = re.search("^fedotllm\s*:\s*.*", str(path).strip())
    if match:
        path = re.sub("^fedotllm\s*:\s*", "", match.group())
        logging.info(f"Config path resolved: {path}")
        return _get_default_config_path(path)
    return Path(path)


def _load_config_file(
    config_path: str | Path, name: Optional[str] = None
) -> DictConfig | ListConfig:
    """
    Load a configuration YAML file using OmegaConf.

    Args:
        config_path: Path to the config file (can use 'fedotllm:xxx' syntax).
        name: Optional name for logging and error messages.

    Returns:
        Loaded configuration as an OmegaConf object.

    Raises:
        ValueError: If the config file is not found.
    """
    custom_config_path = _path_resolver(config_path)
    name = name if name else custom_config_path.name
    if not custom_config_path.is_file():
        raise ValueError(
            f"{name.capitalize()} config file not found at: {custom_config_path}"
        )
    logging.info(f"Loading {name} config from: {custom_config_path}")
    custom_config_path = OmegaConf.load(custom_config_path)
    return custom_config_path


T = TypeVar("T", bound=BaseModel)


def load_config(
    presets: Optional[str | Path | List[str | Path]] = None,
    config_path: Optional[str | Path] = None,
    overrides: Optional[List[str]] = None,
    schema: Type[T] = AppConfig,
) -> T:
    """
    Load and merge configuration from YAML files and command-line overrides.

    Loads the default config, applies one or more preset configs (merging them in order),
    and applies any command-line style overrides. The final config is validated against
    the provided Pydantic schema.

    Args:
        schema: Pydantic model class to validate the config against.
        presets: Single preset name or list of preset names to merge into the config.
        config_path: Path or alias to the default config file.
        overrides: Optional list of command-line overrides in format ["key1=value1", "key2.nested=value2"].

    Returns:
        Loaded and merged configuration as an instance of the provided schema.

    Raises:
        ValueError: If any config file is not found or invalid.
        pydantic.ValidationError: If the merged config does not match the schema.
    """
    # Load env
    load_dotenv()

    # Load default config
    config_path = config_path if config_path else "fedotllm:default"
    config = _load_config_file(config_path, name="default")

    # Apply Presets
    if presets:
        presets = [presets] if isinstance(presets, (str, Path)) else presets
        for preset in presets:
            presets_config = _load_config_file(preset)
            logging.info(f"Merging {preset} config")
            config = OmegaConf.merge(config, presets_config)
            logging.info("Successfully merged custom config with default config")

    # Apply command-line overrides if any
    if overrides:
        logging.info(f"Applying command-line overrides: {overrides}")
        config = apply_overrides(config, overrides)
        logging.info("Successfully applied command-line overrides")

    # Set pydantic schema
    config_dict = OmegaConf.to_object(config)
    config = schema.model_validate(config_dict)
    return config
