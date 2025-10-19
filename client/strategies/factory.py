"""
Factory for creating federated learning clients.

This module implements the Factory pattern for instantiating different
types of federated learning clients based on configuration.
"""

from ...conf.structs import Environment
from .fl_client import FLClient
from .training.factory import TrainingStrategyFactory


class ClientFactory:
    """
    Factory class for creating federated learning clients.

    Uses the Factory design pattern to instantiate the appropriate
    client type based on configuration.
    """

    @classmethod
    def create(
        cls,
        cid: int,
        config: Environment,
    ) -> FLClient:
        """
        Create a federated learning client based on configuration.

        Args:
            cid: Client ID
            config: Environment configuration

        Returns:
            Instance of the appropriate FLClient subclass

        Raises:
            ValueError: If training strategy is not recognized
        """
        training_strategy = TrainingStrategyFactory.create(config)

        return FLClient(cid=cid, config=config, training_strategy=training_strategy)
