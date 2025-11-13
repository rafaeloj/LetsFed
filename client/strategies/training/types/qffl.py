import copy
from typing import TYPE_CHECKING

import numpy as np
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)

from .....utils.logger import Logger
from .....utils.utils import Utils
from ..base import TrainingStrategy
from ..structs import QFFLTrainingStrategyConfig

if TYPE_CHECKING:
    from ...fl_client import FLClient

logger = Logger(__name__)


class QFFLClient(TrainingStrategy):
    """
    Quantized Federated Learning (QFFL) client strategy.
    """

    def __init__(self, config: QFFLTrainingStrategyConfig) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            client: The training strategy configuration.
        """
        super().__init__(config)
        logger.info("QFFLClient training strategy initialized")

    def fit(
        self, client: "FLClient", parameters: NDArrays, config: Config
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
        logger.debug(f"Client {client.cid}: Starting QFFL fit operation")
        # Initialize fit response
        fit_response = {
            "cid": client.cid,
            "participating_state": client.get_participating_state(),
        }

        # Calculate model size
        model_size = sum([layer.nbytes for layer in parameters])
        client.model_size = model_size
        logger.debug(f"Client {client.cid}: Model size = {model_size / (1024**2):.2f} MB")

        # Analyze if the client is selected
        client.selected = Utils.is_select_by_server(
            client.cid, config["selected_by_server"].split(",")
        )
        logger.info(f"Client {client.cid}: Selected = {client.selected}")

        # Fitting model
        if client.selected:
            # Setting the parameters
            client.set_parameters(parameters)
            prev_model_parameters = copy.deepcopy(client.get_parameters())
            logger.debug(f"Client {client.cid}: Model parameters set for training")

            # Fitting model
            history = client.model.fit(
                client.x_train, client.y_train, epochs=client.conf.client.epochs, verbose=0
            )
            client.g_fit_acc = np.mean(history.history["accuracy"])
            client.g_fit_loss = np.mean(history.history["loss"])

            # Calculate delta parameters
            delta_parameters = [
                curr - prev
                for curr, prev in zip(client.get_parameters(), prev_model_parameters, strict=True)
            ]
            delta_parameters = delta_parameters * (1 / self.config.eta)  ## QFFL

            logger.info(
                f"Client {client.cid}: Training completed - "
                + f"Acc: {client.g_fit_acc:.4f}, Loss: {client.g_fit_loss:.4f}"
            )

            return delta_parameters, client.x_train.shape[0], fit_response

        return client.get_parameters(), client.x_train.shape[0], fit_response

    def evaluate(
        self, client: "FLClient", parameters: NDArrays, config: Config
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
        logger.debug(f"Client {client.cid}: Starting QFFL evaluation")
        # Analyze if the client is selected
        client.selected = Utils.is_select_by_server(
            client.cid, config["selected_by_server"].split(",")
        )
        logger.info(f"Client {client.cid}: Selected = {client.selected}")

        # Set weights if selected
        if client.selected:
            client.set_parameters(parameters)
            logger.debug(f"Client {client.cid}: Model parameters set for evaluation")

        # Evaluate the model
        loss, acc = client.model.evaluate(client.x_test, client.y_test)
        client.g_eval_acc = np.mean(acc)
        client.g_eval_loss = np.mean(loss)

        logger.info(
            f"Client {client.cid}: Evaluation completed - "
            + f"Acc: {client.g_eval_acc:.4f}, Loss: {client.g_eval_loss:.4f}"
        )
        eval_resp = {
            "cid": client.cid,
            "acc": client.g_eval_acc,
            "loss": client.g_eval_loss,
            "participating_state": client.get_participating_state(),
        }

        return loss, client.x_test.shape[0], eval_resp
