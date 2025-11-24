"""
Configuration loader and validator.

This module provides utilities to load and validate configuration files.
"""

from pathlib import Path
from typing import Union

from omegaconf import OmegaConf

from .structs import Environment


def load_config(config_path: Union[str, Path]) -> Environment:
    """
    Load configuration from YAML file and validate it.

    The factories (TrainingStrategyFactory, AggregationFactory, ClientSelectionFactory,
    MetricFactory) will handle the conversion from generic configs to specific configs
    using params_from_json internal method.

    Args:
        config_path: Path to the configuration YAML file

    Returns:
        Environment: Validated configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML
    yaml_config = OmegaConf.load(config_path)

    # Create structured config using Environment as schema
    structured_config = OmegaConf.structured(Environment)

    # Merge YAML into structured config
    # This preserves type safety while allowing flexible YAML structure
    config = OmegaConf.merge(structured_config, yaml_config)

    # Convert to Python object
    config_obj = OmegaConf.to_object(config)

    # Validate
    _validate_config(config_obj)

    return config_obj


def _validate_config(config: Environment) -> None:
    """
    Validate configuration parameters.

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    if config.rounds < 1:
        raise ValueError("rounds must be at least 1")

    if config.n_clients < 1:
        raise ValueError("n_clients must be at least 1")

    if not 0 < config.init_clients <= 1:
        raise ValueError("init_clients must be between 0 and 1")

    # Validate client config
    if config.client.epochs < 1:
        raise ValueError("client epochs must be at least 1")

    if config.client.learning_rate <= 0:
        raise ValueError("client learning_rate must be greater than 0")

    # Note: We don't validate strategy-specific parameters here anymore.
    # The factories will handle validation via params_from_json methods.


def save_config(config: Environment, config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration to save
        config_path: Path where to save the configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        OmegaConf.save(config, f)
