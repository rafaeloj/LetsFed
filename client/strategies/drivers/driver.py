from abc import ABC, abstractmethod

from flwr.common import (
    Config,
    NDArrays,
)

from ..fl_client import FLClient


class Driver(ABC):
    """
    Base class for all driver implementations.
    """

    def get_name(self) -> str:
        """
        Get the name of the driver.

        Returns:
            str: The name of the driver.
        """
        return self.__class__.__name__

    @abstractmethod
    def run(self, client: FLClient, parameters: NDArrays, config: Config) -> None:
        """
        Run the driver with the given client, parameters, and config.

        Args:
            client (FLClient): The federated learning client.
            parameters (NDArrays): The model parameters.
            config (Config): Configuration dictionary.
        """
        ...
