from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)

from ..drivers.context import DriverContext
from ..drivers.driver import Driver
from .structs import TrainingStrategyConfig

if TYPE_CHECKING:
    from client.strategies.fl_client import FLClient


class TrainingStrategy(ABC):
    """
    Abstract base class for federated learning training strategies.
    """

    def __init__(self, config: TrainingStrategyConfig) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            config: The training strategy configuration.
        """
        self.config = config
        self.drivers: list[Driver] = []

    def add_drivers(self, drivers: list[Driver]) -> None:
        """
        Add a list of drivers to the client.
        Implements the Plugin Architecture pattern.

        Args:
            drivers: List of driver instances to add
        """
        self.drivers.extend(drivers)

    def apply_drivers(
        self, client: FLClient, parameters: NDArrays, config: Config
    ) -> dict[str, float | int | bool]:
        """
        Apply all registered drivers in sequence using DriverContext.
        Implements the Chain of Responsibility pattern with explicit side effects.

        Args:
            client: The federated learning client instance.
            parameters: Model parameters
            config: Configuration dictionary

        Returns:
            Dictionary of all modifications made by drivers
        """
        context = DriverContext()

        # Run all drivers, collecting their results in the context
        for driver in self.drivers:
            driver.run(client, parameters, config, context)

        # Apply all modifications from context to client attributes
        modifications = context.get_all()
        for key, value in modifications.items():
            setattr(self, key, value)

        return modifications

    def get_drivers(self) -> list[Driver]:
        """
        Get the list of registered drivers.

        Returns:
            List of driver instances
        """
        return self.drivers

    @abstractmethod
    def fit(
        self, client: FLClient, parameters: NDArrays, config: Config
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Train the model on the client data.
        Args:
            client (FLClient): The federated learning client.
            parameters (NDArrays): The model parameters.
            config (Config): Configuration dictionary.

        Returns:
            tuple[NDArrays, int, dict[str, Scalar]]: Updated model parameters,
            number of examples used for training, and additional metrics.
        """
        ...

    @abstractmethod
    def evaluate(
        self, client: FLClient, parameters: NDArrays, config: Config
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Evaluate the model on the client test data.

        Args:
            client (FLClient): The federated learning client.
            parameters (NDArrays): The model parameters.
            config (Config): Configuration dictionary.

        Returns:
            tuple[NDArrays, int, dict[str, Scalar]]: Loss, number of examples used for evaluation,
            and additional metrics.
        """
        ...
