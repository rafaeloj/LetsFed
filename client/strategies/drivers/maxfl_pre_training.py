import copy
from typing import TYPE_CHECKING

from flwr.common import (
    Config,
    NDArrays,
)

from .context import DriverContext
from .driver import Driver

if TYPE_CHECKING:
    from ..fl_client import FLClient


class MaxFLPreTrainingDriver(Driver):
    """
    MaxFL Pre-Training Driver for federated learning clients.

    This driver performs pre-training on the client's local data to compute
    the local fit loss before receiving the global model.

    Modifies:
        - l_fit_loss: Local fit loss computed from pre-training
    """

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
        net_tmp = copy.deepcopy(client.model)
        net_tmp.fit(
            client.x_train,
            client.y_train,
            epochs=client.conf.server.aggregation_method.pre_training_epochs,
        )

        loss, acc = net_tmp.evaluate(client.x_validation, client.y_validation)

        # Store result in context instead of directly modifying client
        context.set("l_fit_loss", float(loss))
