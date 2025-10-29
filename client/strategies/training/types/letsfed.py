from typing import TYPE_CHECKING

import numpy as np
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)

from .....conf.structs import LetsFedTrainingStrategyConfig
from .....utils.utils import Utils
from ...drivers.accuracy import AccuracyDriver
from ...drivers.driver import Driver
from ..base import TrainingStrategy

if TYPE_CHECKING:
    from ...fl_client import FLClient


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

    def _get_drivers(self) -> list[Driver]:
        """
        Get the list of drivers for the MaxFL client.

        Returns:
            list[Driver]: List of driver instances.
        """
        drivers = [
            AccuracyDriver(),
        ]
        return drivers

    def fit(
        self, client: FLClient, parameters: NDArrays, config: Config
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
        # Initialize fit response
        fit_response = {
            "cid": client.cid,
            "participating_state": client.get_participating_state(),
        }

        # Calculate and store model size
        model_size = sum([layer.nbytes for layer in parameters])
        client.model_size = model_size

        # Analyze if the client is selected
        client.selected = Utils.is_select_by_server(
            client.cid, config["selected_by_server"].split(",")
        )

        # Setting global parameters for selected clients if they want to participate
        if client.selected:
            if client.get_participating_state():
                client.set_parameters(parameters)

            history = client.model.fit(
                client.x_train, client.y_train, epochs=client.conf.client.epochs, verbose=0
            )
            client.g_fit_acc = np.mean(history.history["accuracy"])
            client.g_fit_loss = np.mean(history.history["loss"])

        return client.get_parameters(), client.x_train.shape[0], fit_response

    def _manager_client_state(self, client: FLClient) -> None:
        """
        Manage the client's participation state.

        Args:
            client (FLClient): The federated learning client.
        """
        if not self.willing:
            client.set_participating_state(False)

        if self.willing:
            client.set_participating_state(True)

    def evaluate(
        self, client: FLClient, parameters: NDArrays, config: Config
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
        # Analyze if the client is selected
        client.selected = Utils.is_select_by_server(
            str(client.cid), config["selected_by_server"].split(",")
        )

        # Analyzing if the client is selected by the server
        if client.selected:
            # Set global model weights and apply drivers
            self.apply_drivers(client, parameters, config)
            self._manager_client_state(client)

            if client.get_participating_state():
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

        return loss, client.x_test.shape[0], eval_resp
