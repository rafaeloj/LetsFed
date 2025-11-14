from typing import TYPE_CHECKING

import numpy as np
from flwr.common import Config, NDArrays, Scalar

from .....utils.logger import Logger
from .....utils.utils import Utils
from ..base import TrainingStrategy
from ..structs import NormalTrainingStrategyConfig

if TYPE_CHECKING:
    from ..base import FLClient

logger = Logger(__name__)


class NormalClient(TrainingStrategy):
    """
    Normal training strategy for federated learning clients.
    """

    def __init__(self, config: NormalTrainingStrategyConfig) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            config: The training strategy configuration.
        """
        super().__init__(config)
        logger.info("NormalClient training strategy initialized")

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
        logger.debug(f"Client {client.cid}: Starting fit operation")

        # Initialize fit response
        fit_response = {"cid": client.cid, "participating_state": client.get_participating_state()}

        # Calculate model size
        model_size = sum([layer.nbytes for layer in parameters])
        client.model_size = model_size
        logger.debug(f"Client {client.cid}: Model size = {model_size / (1024**2):.2f} MB")

        # Anayze if the client is selected
        client.selected = Utils.is_select_by_server(
            client.cid, config["selected_by_server"].split(",")
        )
        logger.info(f"Client {client.cid}: Selected = {client.selected}")

        # Fitting model
        if client.selected:
            logger.info(
                (
                    f"Client {client.cid}: Starting training for "
                    + f"{client.conf.client.epochs} epochs"
                )
            )
            # Setting the parameters
            client.set_parameters(parameters)

            # Fitting model
            history = client.model.fit(
                client.x_train, client.y_train, epochs=client.conf.client.epochs, verbose=0
            )
            client.g_fit_acc = np.mean(history.history["accuracy"])
            client.g_fit_loss = np.mean(history.history["loss"])
            logger.info(
                (
                    f"Client {client.cid}: Training completed - "
                    + f"Accuracy: {client.g_fit_acc:.4f}, Loss: {client.g_fit_loss:.4f}"
                )
            )
        else:
            logger.debug(f"Client {client.cid}: Skipping training (not selected)")

        return client.get_parameters(config), client.x_train.shape[0], fit_response

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
        logger.debug(f"Client {client.cid}: Starting evaluation")

        # Analyze if the client is selected
        client.selected = Utils.is_select_by_server(
            client.cid, config["selected_by_server"].split(",")
        )

        # Set weights if selected
        if client.selected:
            logger.debug(f"Client {client.cid}: Setting parameters for evaluation")
            client.set_parameters(parameters)

        # Evaluate the model
        loss, acc = client.model.evaluate(client.x_test, client.y_test)
        client.g_eval_acc = np.mean(acc)
        client.g_eval_loss = np.mean(loss)
        logger.info(
            (
                f"Client {client.cid}: Evaluation completed - "
                + f"Accuracy: {client.g_eval_acc:.4f}, Loss: {client.g_eval_loss:.4f}"
            )
        )

        eval_resp = {
            "cid": client.cid,
            "acc": client.g_eval_acc,
            "loss": client.g_eval_loss,
            "participating_state": client.get_participating_state(),
        }

        return loss, client.x_test.shape[0], eval_resp
