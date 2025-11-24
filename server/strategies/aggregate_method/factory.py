"""
Factory for aggregation methods.

This module implements the Factory pattern for creating aggregation strategies.
"""

from typing import Type

from ....utils.logger import Logger
from .base import AggregationMethod
from .structs import AggregationMethodConfig
from .types.fedavg import FedAVG
from .types.maxfl import MaxFL

logger = Logger(__name__)


class AggregationFactory:
    """
    Factory for creating aggregation strategies.

    Implements the Factory design pattern for instantiating different
    aggregation methods based on configuration.
    """

    _registry: dict[str, Type[AggregationMethod]] = {
        "fedavg": FedAVG,
        "maxfl": MaxFL,
    }

    @classmethod
    def create(cls, config: AggregationMethodConfig) -> AggregationMethod:
        """
        Create an aggregation method based on configuration.

        Args:
            config: AggregationMethod configuration with name and params

        Returns:
            Instance of appropriate AggregateMethod subclass

        Raises:
            ValueError: If aggregation method is not recognized
        """
        logger.info(f"Creating aggregation method: {config.name}")
        method = config.name.lower()

        if method not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown aggregation method: '{method}'. " + f"Available methods: {available}"
            )

        agg_class = cls._registry[method]

        # Use params_from_json to create strategy-specific config
        specific_config = agg_class.params_from_json(config.params)

        return agg_class(specific_config)

    @classmethod
    def register(cls, name: str, agg_class: Type[AggregationMethod]) -> None:
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

        if not issubclass(agg_class, AggregationMethod):
            raise ValueError(
                "Aggregation class must be a subclass of AggregationMethod, " + f"got {agg_class}"
            )

        cls._registry[name] = agg_class

    @classmethod
    def get_available_methods(cls) -> list[str]:
        """Get list of available aggregation methods."""
        return list(cls._registry.keys())
