"""
Configuration loader and validator.

This module provides utilities to load and validate configuration files.
"""

from pathlib import Path
from typing import Type, Union

from omegaconf import OmegaConf

from .structs import (
    DeevSelection,
    Environment,
    FedAvgAggregation,
    FedPerTraining,
    LetsFedSelection,
    LetsFedTraining,
    MaxFLAggregation,
    MaxFLTraining,
    NormalTraining,
    PoCSelection,
    QFFLAggregation,
    QFFLTraining,
    RandomSelection,
    RoundRobinSelection,
)

# Register all structured configs
OmegaConf.register_new_resolver("aggregation_factory", lambda name: _get_aggregation_class(name))
OmegaConf.register_new_resolver("selection_factory", lambda name: _get_selection_class(name))
OmegaConf.register_new_resolver("training_factory", lambda name: _get_training_class(name))


def _get_aggregation_class(name: str) -> Type:
    """Get aggregation method class by name."""
    mapping = {
        "fedavg": FedAvgAggregation,
        "maxfl": MaxFLAggregation,
        "qffl": QFFLAggregation,
    }
    return mapping.get(name, FedAvgAggregation)


def _get_selection_class(name: str) -> Type:
    """Get selection method class by name."""
    mapping = {
        "random": RandomSelection,
        "deev": DeevSelection,
        "poc": PoCSelection,
        "round_robin": RoundRobinSelection,
        "letsfed": LetsFedSelection,
    }
    return mapping.get(name, RandomSelection)


def _get_training_class(name: str) -> Type:
    """Get training strategy class by name."""
    mapping = {
        "normal": NormalTraining,
        "letsfed": LetsFedTraining,
        "maxfl": MaxFLTraining,
        "fedper": FedPerTraining,
        "qffl": QFFLTraining,
    }
    return mapping.get(name, NormalTraining)


def load_config(config_path: Union[str, Path]) -> Environment:
    """
    Load configuration from YAML file and validate it.

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

    # Convert structured configs based on 'name' fields
    if "server" in yaml_config and "aggregation_method" in yaml_config.server:
        agg_name = yaml_config.server.aggregation_method.get("name", "fedavg")
        agg_class = _get_aggregation_class(agg_name)
        yaml_config.server.aggregation_method = OmegaConf.merge(
            OmegaConf.structured(agg_class), yaml_config.server.aggregation_method
        )

    if "server" in yaml_config and "selection_method" in yaml_config.server:
        sel_name = yaml_config.server.selection_method.get("name", "random")
        sel_class = _get_selection_class(sel_name)
        yaml_config.server.selection_method = OmegaConf.merge(
            OmegaConf.structured(sel_class), yaml_config.server.selection_method
        )

    if "client" in yaml_config and "training_strategy" in yaml_config.client:
        train_name = yaml_config.client.training_strategy.get("name", "normal")
        train_class = _get_training_class(train_name)
        yaml_config.client.training_strategy = OmegaConf.merge(
            OmegaConf.structured(train_class), yaml_config.client.training_strategy
        )

    # Create structured config and merge
    config = OmegaConf.structured(Environment)
    config = OmegaConf.merge(config, yaml_config)

    # Validate
    _validate_config(config)

    return OmegaConf.to_object(config)


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

    # Validate server config
    if hasattr(config.server.selection_method, "perc_of_clients"):
        if not 0 < config.server.selection_method.perc_of_clients <= 1:
            raise ValueError("selection_method.perc_of_clients must be between 0 and 1")

    # Validate client config
    if config.client.epochs < 1:
        raise ValueError("client epochs must be at least 1")

    if config.client.learning_rate <= 0:
        raise ValueError("client learning_rate must be positive")


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
