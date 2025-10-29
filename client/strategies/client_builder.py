"""
Builder for creating federated learning clients.

This module implements the Builder pattern for constructing federated
learning clients with their required training strategy.
"""

from ...conf.structs import Environment
from .fl_client import FLClient
from .training.factory import TrainingStrategyFactory


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

        Constructs the client with the appropriate training strategy.

        Args:
            cid: Client ID
            config: Environment configuration

        Returns:
            Instance of FLClient configured with the required training strategy

        Raises:
            ValueError: If training strategy is not recognized
        """
        training_strategy = TrainingStrategyFactory.create(config.client.training_strategy)

        return FLClient(cid=cid, config=config, training_strategy=training_strategy)
