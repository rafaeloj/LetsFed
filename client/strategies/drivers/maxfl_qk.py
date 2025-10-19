from typing import TYPE_CHECKING

import numpy as np
from flwr.common import (
    Config,
    NDArrays,
)

from .driver import Driver

if TYPE_CHECKING:
    from ..fl_client import FLClient


class MaxFLQkDriver(Driver):
    """
    MaxFL QK Driver for federated learning clients.
    """

    def sigmoid(self, x: float) -> float:
        """
        Sigmoid activation function.

        Args:
            x (float): Input value.

        Returns:
            float: Sigmoid of the input value.
        """
        temp_loss = 2 * x
        return 2 * np.exp(temp_loss) / (1.0 + np.exp(temp_loss))

    def run(self, client: FLClient, parameters: NDArrays, config: Config) -> None:
        """
        Run the driver with the given client, parameters, and config.

        Args:
            client (FLClient): The federated learning client.
            parameters (NDArrays): The model parameters.
            config (Config): Configuration dictionary.
        """
        loss_weight = self.sigmoid(np.sum(client.maxfl_loss) - client.maxfl_threshold)
        client.qk = loss_weight * (1 - loss_weight)
