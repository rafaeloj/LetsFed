import copy
from typing import TYPE_CHECKING

import numpy as np
from flwr.common import (
    Config,
    NDArrays,
)

from ....utils.logger import Logger
from .context import DriverContext
from .driver import Driver

if TYPE_CHECKING:
    from ..fl_client import FLClient

logger = Logger(__name__)


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
        self, client: "FLClient", parameters: NDArrays, config: Config, context: DriverContext
    ) -> None:
        """
        Run the driver with the given client, parameters, and config.

        Args:
            client (FLClient): The federated learning client.
            parameters (NDArrays): The model parameters.
            config (Config): Configuration dictionary.
            context (DriverContext): Context for storing driver results.
        """
        logger.debug(f"Client {client.cid}: MaxFLQkDriver started")
        pre_training_epochs = client.training_strategy.config.pre_training_epochs

        # Step 1: Compute true_loss - train local model from global parameters
        # This represents the performance gain from local training
        logger.debug(
            f"Client {client.cid}: Training local model from global params "
            + f"for {pre_training_epochs} epochs"
        )
        local_model = copy.deepcopy(client.model)
        local_model.set_weights(parameters)  # Start from global parameters
        local_model.fit(
            client.x_train,
            client.y_train,
            epochs=pre_training_epochs,
            verbose=0,
        )
        true_loss, local_acc = local_model.evaluate(
            client.x_validation, client.y_validation, verbose=0
        )
        logger.debug(
            f"Client {client.cid}: Local model (trained from global) - "
            + f"Loss: {true_loss:.4f}, Acc: {local_acc:.4f}"
        )

        # Step 2: Compute g_loss - evaluate global model WITHOUT local training
        # This represents the baseline performance of the global model
        logger.debug(f"Client {client.cid}: Evaluating global model (no training)")
        global_model = copy.deepcopy(client.model)
        global_model.set_weights(parameters)  # Use global parameters as-is
        g_loss, global_acc = global_model.evaluate(
            client.x_validation, client.y_validation, verbose=0
        )
        logger.debug(
            f"Client {client.cid}: Global model (no training) - "
            + f"Loss: {g_loss:.4f}, Acc: {global_acc:.4f}"
        )

        # Step 3: Compute qk using sigmoid function
        # loss_diff > 0: local training IMPROVED the model (qk > 0.5, should participate)
        # loss_diff = 0: no improvement (qk = 0.5, neutral)
        # loss_diff < 0: local training WORSENED the model (qk < 0.5, should NOT participate)
        loss_diff = g_loss - true_loss  # How much local training improved the model
        qk = self.sigmoid(loss_diff)

        logger.info(
            f"Client {client.cid}: MaxFLQkDriver computed qk = {float(qk):.4f} "
            + f"(loss_diff={loss_diff:.4f}, threshold=0.5)"
        )

        # Store result in context instead of directly modifying client
        context.set("qk", float(qk))
