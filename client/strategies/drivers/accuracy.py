import copy
from typing import TYPE_CHECKING

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


class AccuracyDriver(Driver):
    """
    Driver for accuracy-based client selection.

    This driver determines if a client is willing to participate based on
    comparing local model performance against the global model.

    Modifies:
        - willing: Boolean indicating if client wants to participate
    """

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
        logger.debug(f"Client {client.cid}: AccuracyDriver started")

        # Case for first round
        server_round = config["rounds"]
        if server_round == 1:
            logger.info(f"Client {client.cid}: First round - willing to participate")
            context.set("willing", True)
            return

        # Creating the global model
        g_model = copy.deepcopy(client.model)
        g_model.set_weights(parameters)

        g_tmp_loss, _ = g_model.evaluate(client.x_validation, client.y_validation, verbose=0)
        c_tmp_loss, _ = client.model.evaluate(client.x_validation, client.y_validation, verbose=0)

        logger.debug(
            f"Client {client.cid}: AccuracyDriver - "
            + f"Global loss: {g_tmp_loss:.4f}, Client loss: {c_tmp_loss:.4f}"
        )

        willing = self._client_willing(
            global_loss=g_tmp_loss,
            client_loss=c_tmp_loss,
            threshold=client.training_strategy.config.threshold_accuracy,
        )

        logger.info(
            f"Client {client.cid}: AccuracyDriver - "
            + f"Willing: {willing} (ratio={g_tmp_loss / c_tmp_loss:.4f}, "
            + f"threshold={client.training_strategy.config.threshold_accuracy})"
        )

        # Store result in context instead of directly modifying client
        context.set("willing", willing)

    def _client_willing(
        self, global_loss: float, client_loss: float, threshold: float = None
    ) -> bool:
        """
        Check if the global loss is better than the client loss.

        Args:
            global_loss (float): The global model loss.
            client_loss (float): The client model loss.
            threshold (float): The threshold for improvement.

        Returns:
            bool: True if the global loss is better than the client loss, False otherwise.
        """
        return (global_loss / client_loss) >= threshold
