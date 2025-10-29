"""
Factory for aggregation methods.

This module implements the Factory pattern for creating aggregation strategies.
"""

from typing import Type

from .base import TrainingStrategy
from .structs import TrainingStrategyConfig
from .types.fedper import FedPerClient
from .types.letsfed import LetsFedClient
from .types.maxfl import MaxFLClient
from .types.normal import NormalClient
from .types.qffl import QFFLClient


class TrainingStrategyFactory:
    """
    Factory for creating training strategies.

    Implements the Factory design pattern for instantiating different
    training strategies based on configuration.
    """

    _registry: dict[str, Type[TrainingStrategy]] = {
        "fedper": FedPerClient,
        "letsfed": LetsFedClient,
        "maxfl": MaxFLClient,
        "normal": NormalClient,
        "qffl": QFFLClient,
    }

    @classmethod
    def create(cls, config: TrainingStrategyConfig) -> TrainingStrategy:
        """
        Create a training strategy based on configuration.

        Args:
            config: Environment configuration

        Returns:
            Instance of appropriate TrainingStrategy subclass

        Raises:
            ValueError: If training strategy is not recognized
        """
        method = config.name.lower()

        if method not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown training strategy: '{method}'. " + f"Available strategies: {available}"
            )

        strategy_class = cls._registry[method]
        return strategy_class(config)

    @classmethod
    def register(cls, name: str, strategy_class: Type[TrainingStrategy]) -> None:
        """
        Register a new training strategy.

        Args:
            name: Method name to register
            strategy_class: Training strategy class to associate with the name

        Raises:
            ValueError: If name already registered or class is invalid
        """
        if name in cls._registry:
            raise ValueError(f"Training strategy '{name}' is already registered")

        if not issubclass(strategy_class, TrainingStrategy):
            raise ValueError(
                "Training strategy class must be a subclass of TrainingStrategy, "
                + f"got {strategy_class}"
            )

        cls._registry[name] = strategy_class

    @classmethod
    def get_available_methods(cls) -> list[str]:
        """Get list of available training strategies."""
        return list(cls._registry.keys())
