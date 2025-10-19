from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)

if TYPE_CHECKING:
    from client.strategies.fl_client import FLClient


class TrainingStrategy(ABC):
    @abstractmethod
    def init(self, client: FLClient) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            client: The federated learning client instance.
        """
        ...

    @abstractmethod
    def get_parameters(self, client: FLClient) -> NDArrays:
        """
        Get model parameters from the client.

        Args:
            client (FLClient): The federated learning client.

        Returns:
            NDArrays: The model parameters.
        """
        ...

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
