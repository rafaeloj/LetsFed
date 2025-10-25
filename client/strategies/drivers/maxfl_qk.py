import copy
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
        # Pre-training on client local data to compute the real loss
        local_model = copy.deepcopy(client.model)
        local_model.fit(
            client.x_train,
            client.y_train,
            epochs=client.conf.server.aggregation_method.pre_training_epochs,
        )
        true_loss, acc = local_model.evaluate(client.x_validation, client.y_validation)

        # Generating the global fit loss
        global_model = copy.deepcopy(client.model)
        global_model.set_weights(parameters)
        global_model.fit(
            client.x_train,
            client.y_train,
            epochs=client.conf.server.aggregation_method.pre_training_epochs,
        )
        g_loss, g_acc = global_model.evaluate(client.x_validation, client.y_validation)

        # Compute qk using sigmoid function
        loss_weight = self.sigmoid(np.sum(g_loss) - np.sum(true_loss))
        qk = loss_weight * (1 - loss_weight)

        # Store result in context instead of directly modifying client
        context.set("qk", float(qk))
