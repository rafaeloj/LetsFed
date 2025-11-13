from typing import TYPE_CHECKING

import numpy as np
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)

from .....utils.logger import Logger
from .....utils.utils import Utils
from ...drivers.accuracy import AccuracyDriver
from ...drivers.driver import Driver
from ..base import TrainingStrategy
from ..structs import LetsFedTrainingStrategyConfig

if TYPE_CHECKING:
    from ...fl_client import FLClient

logger = Logger(__name__)


class LetsFedClient(TrainingStrategy):
    """
    Federated Learning strategy for the LetsFed framework.
    """

    def __init__(self, config: LetsFedTrainingStrategyConfig) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            client: The federated learning client instance.
        """
        super().__init__(config)
        self.add_drivers(self._get_drivers())
        logger.info("LetsFedClient training strategy initialized with drivers")

    def _get_drivers(self) -> list[Driver]:
        """
        Get the list of drivers for the MaxFL client.

        Returns:
            list[Driver]: List of driver instances.
        """
        drivers = [
            AccuracyDriver(),
        ]
        logger.debug(f"LetsFedClient initialized with {len(drivers)} drivers")
        return drivers

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
        logger.debug(
            f"Client {client.cid}: Starting LetsFed fit - "
            + f"Participating: {client.get_participating_state()}"
        )

        # Initialize fit response
        fit_response = {
            "cid": client.cid,
            "participating_state": client.get_participating_state(),
        }

        # Calculate and store model size
        model_size = sum([layer.nbytes for layer in parameters])
        client.model_size = model_size
        logger.debug(f"Client {client.cid}: Model size = {model_size / (1024**2):.2f} MB")

        # Analyze if the client is selected
        client.selected = Utils.is_select_by_server(
            client.cid, config["selected_by_server"].split(",")
        )
        logger.info(
            f"Client {client.cid}: Selected = {client.selected}, "
            + f"Participating = {client.get_participating_state()}"
        )

        # Setting global parameters for selected clients if they want to participate
        if client.selected:
            if client.get_participating_state():
                logger.info(f"Client {client.cid}: Accepting global model and starting training")
                client.set_parameters(parameters)

            history = client.model.fit(
                client.x_train, client.y_train, epochs=client.conf.client.epochs, verbose=0
            )
            client.g_fit_acc = np.mean(history.history["accuracy"])
            client.g_fit_loss = np.mean(history.history["loss"])
            logger.info(
                f"Client {client.cid}: Training completed - "
                + f"Acc: {client.g_fit_acc:.4f}, Loss: {client.g_fit_loss:.4f}"
            )
        else:
            logger.debug(f"Client {client.cid}: Skipping training (not selected)")

        return client.get_parameters(), client.x_train.shape[0], fit_response

    def _manager_client_state(self, client: "FLClient") -> None:
        """
        Manage the client's participation state.

        Args:
            client (FLClient): The federated learning client.
        """
        old_state = client.get_participating_state()
        if not self.willing:
            client.set_participating_state(False)
            logger.info(f"Client {client.cid}: Changed participation state to False (not willing)")

        if self.willing:
            client.set_participating_state(True)
            if not old_state:
                logger.info(f"Client {client.cid}: Changed participation state to True (willing)")

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
        logger.debug(f"Client {client.cid}: Starting LetsFed evaluation")

        # Analyze if the client is selected
        client.selected = Utils.is_select_by_server(
            str(client.cid), config["selected_by_server"].split(",")
        )

        # Analyzing if the client is selected by the server
        if client.selected:
            logger.debug(f"Client {client.cid}: Applying drivers for evaluation")
            # Set global model weights and apply drivers
            self.apply_drivers(client, parameters, config)
            self._manager_client_state(client)

            if client.get_participating_state():
                logger.debug(f"Client {client.cid}: Setting global parameters for evaluation")
                client.set_parameters(parameters)

        # Evaluate the model
        loss, acc = client.model.evaluate(client.x_test, client.y_test)
        client.g_eval_acc = np.mean(acc)
        client.g_eval_loss = np.mean(loss)
        eval_resp = {
            "cid": client.cid,
            "acc": client.g_eval_acc,
            "loss": client.g_eval_loss,
            "participating_state": client.get_participating_state(),
        }
        logger.info(
            f"Client {client.cid}: Evaluation completed - "
            + f"Acc: {client.g_eval_acc:.4f}, Loss: {client.g_eval_loss:.4f}, "
            + f"Participating: {client.get_participating_state()}"
        )

        return loss, client.x_test.shape[0], eval_resp
