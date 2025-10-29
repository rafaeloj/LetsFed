"""
Builder for creating federated learning servers.

This module implements the Builder pattern for constructing federated
learning servers with their required strategies (aggregation and client selection).
"""

from ...conf.structs import Environment
from .aggregate_method.factory import AggregationFactory
from .client_selection_method.factory import ClientSelectionFactory
from .fl_server import FLServer


class ServerBuilder:
    """
    Builder for creating federated learning servers.
    Implements the Builder design pattern for server construction with dependency injection.
    """

    @classmethod
    def create(cls, config: Environment) -> FLServer:
        """
        Build a federated learning server based on configuration.

        Constructs the server with the appropriate aggregation and selection strategies.

        Args:
            config: Environment configuration

        Returns:
            Instance of FLServer class configured with the required strategies

        Raises:
            ValueError: If configuration is invalid
        """
        # Instantiating the aggregation and selection methods
        aggregation_method = AggregationFactory.create(config.server.aggregation_method)
        selection_method = ClientSelectionFactory.create(config.server.selection_method)

        return FLServer(
            conf=config, client_selection=selection_method, aggregate_method=aggregation_method
        )
