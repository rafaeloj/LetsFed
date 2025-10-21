from typing import TYPE_CHECKING

import numpy as np
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)

from .....utils.utils import Utils
from ...drivers.driver import Driver
from ...drivers.maxfl_pre_training import MaxFLPreTrainingDriver
from ...drivers.maxfl_qk import MaxFLQkDriver
from ..base import TrainingStrategy

if TYPE_CHECKING:
    from ...fl_client import FLClient


class MaxFLClient(TrainingStrategy):
    """
    MaxFL training strategy for federated learning clients.
    """

    def init(self, client: FLClient) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            client: The federated learning client instance.
        """
        pre_training = MaxFLPreTrainingDriver()
        pre_training.run(client, None, None)
        client.data_to_log["maxfl_threshold"] = client.maxfl_threshold
        client.add_drivers(self._get_drivers())

    def _get_drivers(self) -> list[Driver]:
        """
        Get the list of drivers for the MaxFL client.

        Returns:
            list[Driver]: List of driver instances.
        """
        drivers = [MaxFLQkDriver()]
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
            "participating_state": True,
        }

        # Calculate and store model size
        model_size = sum([layer.nbytes for layer in parameters])
        client.model_size = model_size

        # Evaluate local model before training (for case when the client is not selected)
        loss, acc = client.model.evaluate(client.x_train, client.y_train)
        client.maxfl_loss = np.mean(loss)
        client.g_fit_acc = np.mean(acc)
        client.g_fit_loss = np.mean(loss)

        # Apply drivers to compute qk
        client.apply_drivers(parameters=parameters, config=config)
        client.data_to_log["qk"] = client.qk
        fit_response["qk"] = client.qk

        # Analyze if the client is selected
        client.selected = Utils.is_select_by_server(
            client.cid, config["selected_by_server"].split(",")
        )

        if client.selected:
            client.set_parameters(parameters)
            history = client.model.fit(
                client.x_train, client.y_train, epochs=client.conf.client.epochs, verbose=0
            )
            client.g_fit_acc = np.mean(history.history["accuracy"])
            client.g_fit_loss = np.mean(history.history["loss"])

        return client.get_parameters(), client.x_train.shape[0], fit_response

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
            client.cid, config["selected_by_server"].split(",")
        )

        # Set weights if selected
        if client.selected:
            client.set_parameters(parameters)

        loss, acc = client.model.evaluate(client.x_test, client.y_test)
        client.g_eval_acc = np.mean(acc)
        client.g_eval_loss = np.mean(loss)
        eval_resp = {
            "cid": client.cid,
            "acc": client.g_eval_acc,
            "loss": client.g_eval_loss,
            "participating_state": client.get_participating_state(),
            "desired_state": client.get_participating_state(),
        }

        return loss, client.x_test.shape[0], eval_resp
