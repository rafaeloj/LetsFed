from typing import TYPE_CHECKING

import numpy as np
from flwr.common import (
    Config,
    NDArrays,
)

from .context import DriverContext
from .driver import Driver

if TYPE_CHECKING:
    from ..fl_client import FLClient


class MaxFLQkDriver(Driver):
    """
    MaxFL QK Driver for federated learning clients.

    This driver computes the quality metric (qk) for MaxFL algorithm
    based on the difference between global and local fit losses.

    Modifies:
        - qk: Quality metric computed using sigmoid function
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

    def run(
        self, client: FLClient, parameters: NDArrays, config: Config, context: DriverContext
    ) -> None:
        """
        Run the driver with the given client, parameters, and config.

        Args:
            client (FLClient): The federated learning client.
            parameters (NDArrays): The model parameters.
            config (Config): Configuration dictionary.
            context (DriverContext): Context for storing driver results.
        """
        loss_weight = self.sigmoid(np.sum(client.g_fit_loss) - np.sum(client.l_fit_loss))
        qk = loss_weight * (1 - loss_weight)

        # Store result in context instead of directly modifying client
        context.set("qk", float(qk))
