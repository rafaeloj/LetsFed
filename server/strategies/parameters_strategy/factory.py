"""
Factory for parameters strategies (Server).

This module implements the Factory pattern for creating parameters strategies on the server.
"""

from typing import Type

from ....utils.logger import Logger
from .base import ParametersStrategy
from .structs import ParametersStrategyConfig
from .types.layerwise import LayerWiseParametersStrategy
from .types.normal import NormalParametersStrategy

logger = Logger(__name__)


class ParametersStrategyFactory:
    """
    Factory for creating parameters strategies (server-side).

    Implements the Factory design pattern for instantiating different
    parameters sharing strategies based on configuration.
    """

    _registry: dict[str, Type[ParametersStrategy]] = {
        "normal": NormalParametersStrategy,
        "layerwise": LayerWiseParametersStrategy,
    }

    @classmethod
    def create(cls, config: ParametersStrategyConfig) -> ParametersStrategy:
        """
        Create a parameters strategy based on configuration.

        Args:
            config: Parameters strategy configuration

        Returns:
            Instance of appropriate ParametersStrategy subclass

        Raises:
            ValueError: If parameters strategy is not recognized
        """
        logger.info(f"Creating server parameters strategy: {config.name}")
        method = config.name.lower()

        if method not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown parameters strategy: '{method}'. " + f"Available strategies: {available}"
            )

        strategy_class = cls._registry[method]
        strategy_class_config = strategy_class.params_from_json(config.params)
        return strategy_class(strategy_class_config)

    @classmethod
    def register(cls, name: str, strategy_class: Type[ParametersStrategy]) -> None:
        """
        Register a new parameters strategy.

        Args:
            name: Strategy name to register
            strategy_class: Parameters strategy class to associate with the name

        Raises:
            ValueError: If name already registered or class is invalid
        """
        if name in cls._registry:
            raise ValueError(f"Parameters strategy '{name}' is already registered")

        if not issubclass(strategy_class, ParametersStrategy):
            raise ValueError(
                "Parameters strategy class must be a subclass of ParametersStrategy, "
                + f"got {strategy_class}"
            )

        cls._registry[name] = strategy_class

    @classmethod
    def get_available_strategies(cls) -> list[str]:
        """Get list of available parameters strategies."""
        return list(cls._registry.keys())
