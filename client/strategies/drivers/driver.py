from abc import ABC, abstractmethod

from flwr.common import (
    Config,
    NDArrays,
)

from ..fl_client import FLClient
from .context import DriverContext


class Driver(ABC):
    """
    Base class for all driver implementations.

    Drivers compute results and store them in a DriverContext instead of
    directly modifying the client. This makes them more testable and their
    side effects explicit.
    """

    def get_name(self) -> str:
        """
        Get the name of the driver.

        Returns:
            str: The name of the driver.
        """
        return self.__class__.__name__

    @abstractmethod
    def run(
        self, client: FLClient, parameters: NDArrays, config: Config, context: DriverContext
    ) -> None:
        """
        Run the driver with the given client, parameters, and config.

        Results should be stored in the context object using context.set().
        This allows for better testability and makes side effects explicit.

        Args:
            client (FLClient): The federated learning client.
            parameters (NDArrays): The model parameters.
            config (Config): Configuration dictionary.
            context (DriverContext): Context for storing driver results.

        Example:
            context.set('qk', computed_qk_value)
            context.set('l_fit_loss', loss_value)
        """
        ...
