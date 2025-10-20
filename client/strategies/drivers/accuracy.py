from typing import TYPE_CHECKING

from flwr.common import (
    Config,
    NDArrays,
)

from .driver import Driver

if TYPE_CHECKING:
    from ..fl_client import FLClient


class AccuracyDriver(Driver):
    """
    Driver for accuracy-based client selection.
    """

    def run(self, client: FLClient, parameters: NDArrays, config: Config) -> None:
        """
        Run the driver with the given client, parameters, and config.

        Args:
            client (FLClient): The federated learning client.
            parameters (NDArrays): The model parameters.
            config (Config): Configuration dictionary.
        """
        # Case for first round
        server_round = config["rounds"]
        if server_round == 1:
            return True

        g_tmp_loss, _ = client.g_model.evaluate(client.x_validation, client.y_validation, verbose=0)
        c_tmp_loss, _ = client.model.evaluate(client.x_validation, client.y_validation, verbose=0)

        willing = self._client_willing(
            global_loss=g_tmp_loss,
            client_loss=c_tmp_loss,
            threshold=client.conf.client.training_strategy.threshold_accuracy,
        )

        client.willing = willing

    def _client_willing(
        self, global_loss: float, client_loss: float, threshold: float = None
    ) -> bool:
        """
        Check if the client loss is better than the global loss.

        Args:
            global_loss (float): The global model loss.
            client_loss (float): The client model loss.
            threshold (float): The threshold for improvement.

        Returns:
            bool: True if the client loss is better than the global loss, False otherwise.
        """
        return (client_loss / global_loss) > threshold
