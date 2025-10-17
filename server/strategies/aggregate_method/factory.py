"""
Factory for aggregation methods.

This module implements the Factory pattern for creating aggregation strategies.
"""

from typing import Type

from conf.structs import Environment

from .base import AggregateMethod
from .types.fedavg import FedAVG
from .types.maxfl import MaxFL


class AggregationFactory:
    """
    Factory for creating aggregation strategies.

    Implements the Factory design pattern for instantiating different
    aggregation methods based on configuration.
    """

    _registry: dict[str, Type[AggregateMethod]] = {
        "fedavg": FedAVG,
        "maxfl": MaxFL,
    }

    @classmethod
    def create(cls, config: Environment) -> AggregateMethod:
        """
        Create an aggregation method based on configuration.

        Args:
            config: Environment configuration

        Returns:
            Instance of appropriate AggregateMethod subclass

        Raises:
            ValueError: If aggregation method is not recognized
        """
        method = config.server.aggregation_method.name.lower()

        if method not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown aggregation method: '{method}'. " + f"Available methods: {available}"
            )

        agg_class = cls._registry[method]
        return agg_class()

    @classmethod
    def register(cls, name: str, agg_class: Type[AggregateMethod]) -> None:
        """
        Register a new aggregation method.

        Args:
            name: Method name to register
            agg_class: Aggregation class to associate with the name

        Raises:
            ValueError: If name already registered or class is invalid
        """
        if name in cls._registry:
            raise ValueError(f"Aggregation method '{name}' is already registered")

        if not issubclass(agg_class, AggregateMethod):
            raise ValueError(
                "Aggregation class must be a subclass of AggregateMethod, " + f"got {agg_class}"
            )

        cls._registry[name] = agg_class

    @classmethod
    def get_available_methods(cls) -> list[str]:
        """Get list of available aggregation methods."""
        return list(cls._registry.keys())
