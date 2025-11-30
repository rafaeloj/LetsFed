"""
Builder for creating federated learning clients.

This module implements the Builder pattern for constructing federated
learning clients with their required training strategy.
"""

from ...conf.structs import Environment
from ...metrics import MetricsManager
from ...utils.logger import Logger
from .fl_client import FLClient
from .parameters_strategy.factory import ParametersStrategyFactory
from .training.factory import TrainingStrategyFactory

logger = Logger(__name__)


class ClientBuilder:
    """
    Builder class for creating federated learning clients.

    Uses the Builder design pattern to construct clients with dependency injection
    of the appropriate training strategy based on configuration.
    """

    @classmethod
    def create(
        cls,
        cid: int,
        config: Environment,
    ) -> FLClient:
        """
        Build a federated learning client based on configuration.

        Constructs the client with the appropriate training strategy and metrics manager.

        Args:
            cid: Client ID
            config: Environment configuration

        Returns:
            Instance of FLClient configured with the required training strategy and metrics

        Raises:
            ValueError: If training strategy is not recognized
        """
        logger.info(f"Building FLClient {cid}")

        # Create training strategy
        training_strategy = TrainingStrategyFactory.create(config.client.training_strategy)

        # Create parameters strategy
        parameters_strategy = ParametersStrategyFactory.create(config.client.parameters_strategy)

        # Create metrics manager from config (or use defaults if None)
        metrics_manager = MetricsManager(config.client.metrics)

        logger.debug(
            f"Client {cid}: Initialized with {len(metrics_manager.metrics)} metrics: "
            + f"{[m.name for m in metrics_manager.metrics]}"
        )

        return FLClient(
            cid=cid,
            config=config,
            training_strategy=training_strategy,
            parameters_strategy=parameters_strategy,
            metrics_manager=metrics_manager,
        )
