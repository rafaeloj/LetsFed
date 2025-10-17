"""
Factory for creating federated learning servers.

This module implements the Factory pattern for instantiating different
types of federated learning servers.
"""

from typing import Type

from ...conf.structs import Environment
from .aggregate_method.factory import AggregationFactory
from .client_selection_method.factory import ClientSelectionFactory
from .fl_server import FLServer


class ServerFactory:
    """
    Factory for creating federated learning servers.
    Implements the Factory design pattern for server instantiation.
    """

    @classmethod
    def create(cls, config: Environment) -> FLServer:
        """
        Create a federated learning server based on configuration.

        The server type is inferred from the aggregation and selection methods.

        Args:
            config: Environment configuration

        Returns:
            Instance of FLServer class

        Raises:
            ValueError: If configuration is invalid
        """
        server_type = config.server.type.lower()

        if server_type not in cls._registry:
            raise ValueError(
                f"Unknown server type: '{server_type}'. "
                + f"Available types: {cls.get_available_types()}"
            )

        server_class = cls._registry[server_type]

        # Instantiating the aggregation and selection methods
        aggregation_method = AggregationFactory.create(config)
        selection_method = ClientSelectionFactory.create(config)

        return server_class(
            conf=config, client_selection=selection_method, aggregate_method=aggregation_method
        )

    @classmethod
    def register(cls, name: str, server_class: Type[FLServer]) -> None:
        """
        Register a new server type.

        Args:
            name: Server type name to register
            server_class: Server class to associate with the name

        Raises:
            ValueError: If name already registered or class is invalid
        """
        if name in cls._registry:
            raise ValueError(f"Server type '{name}' is already registered")

        if not issubclass(server_class, FLServer):
            raise ValueError(
                "Server class must be a subclass of FLServer, " + f"got {server_class}"
            )

        cls._registry[name] = server_class

    @classmethod
    def get_available_types(cls) -> list[str]:
        """Get list of available server types."""
        return list(cls._registry.keys())
