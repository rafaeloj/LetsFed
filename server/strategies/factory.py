"""
Factory for creating federated learning servers.

This module implements the Factory pattern for instantiating different
types of federated learning servers.
"""

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
        # Instantiating the aggregation and selection methods
        aggregation_method = AggregationFactory.create(config)
        selection_method = ClientSelectionFactory.create(config)

        return FLServer(
            conf=config, client_selection=selection_method, aggregate_method=aggregation_method
        )
