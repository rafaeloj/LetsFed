import copy
from typing import TYPE_CHECKING

from flwr.common import (
    Config,
    NDArrays,
)

from .driver import Driver

if TYPE_CHECKING:
    from ..fl_client import FLClient


class MaxFLPreTrainingDriver(Driver):
    def run(self, client: FLClient, parameters: NDArrays, config: Config) -> None:
        """
        Run the driver with the given client, parameters, and config.

        Args:
            client (FLClient): The federated learning client.
            parameters (NDArrays): The model parameters.
            config (Config): Configuration dictionary.
        """
        net_tmp = copy.deepcopy(client.model)
        net_tmp.fit(
            client.x_train,
            client.y_train,
            epochs=client.conf.server.aggregation_method.pre_training_epochs,
        )

        loss, acc = net_tmp.evaluate(client.x_validation, client.y_validation)

        client.maxfl_threshold = loss
