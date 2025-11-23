"""
Builder for creating federated learning servers.

This module implements the Builder pattern for constructing federated
learning servers with their required strategies (aggregation and client selection).
"""

from ...conf.structs import Environment
from ...metrics import MetricsManager
from ...utils.logger import Logger
from .aggregate_method.factory import AggregationFactory
from .client_selection_method.factory import ClientSelectionFactory
from .fl_server import FLServer

logger = Logger(__name__)


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
        logger.info("Building FLServer with specified strategies")
        aggregation_method = AggregationFactory.create(config.server.aggregation_method)
        selection_method = ClientSelectionFactory.create(config.server.selection_method)

        # Instantiate MetricsManager with client's metrics configuration
        # Server uses the same metrics as clients for consistency
        metrics_manager = MetricsManager(config.client.metrics)
        logger.debug(f"MetricsManager created with metrics: {metrics_manager.get_metrics_names()}")

        return FLServer(
            conf=config,
            client_selection=selection_method,
            aggregate_method=aggregation_method,
            metrics_manager=metrics_manager,
        )
