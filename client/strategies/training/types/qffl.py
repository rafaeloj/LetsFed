import copy
from typing import TYPE_CHECKING

import numpy as np
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)

from .....utils.utils import Utils
from ..base import TrainingStrategy

if TYPE_CHECKING:
    from ...fl_client import FLClient


class QFFLClient(TrainingStrategy):
    """
    Quantized Federated Learning (QFFL) client strategy.
    """

    def init(self, client: FLClient) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            client: The federated learning client instance.
        """
        pass

    def get_parameters(self, client: FLClient) -> NDArrays:
        """
        Get model parameters from the client.

        Args:
            client (FLClient): The federated learning client.

        Returns:
            NDArrays: The model parameters.
        """
        return client.model.get_weights()

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

        # Calculate model size
        model_size = sum([layer.nbytes for layer in parameters])
        client.model_size = model_size

        # Evaluate local model before training (for case when the client is not selected)
        loss, acc = client.model.evaluate(client.x_train, client.y_train)
        client.g_fit_acc = np.mean(acc)
        client.g_fit_loss = np.mean(loss)

        # Analyze if the client is selected
        client.selected = Utils.is_select_by_server(
            client.cid, config["selected_by_server"].split(",")
        )

        # Fitting model
        if client.selected:
            client.model.set_weights(parameters)
            prev_model = copy.deepcopy(client.model)
            history = client.model.fit(
                client.x_train, client.y_train, epochs=client.conf.client.epochs, verbose=0
            )
            client.g_fit_acc = np.mean(history.history["accuracy"])
            client.g_fit_loss = np.mean(history.history["loss"])

            # Calculate delta parameters
            delta_parameters = [
                curr - prev
                for curr, prev in zip(
                    client.model.get_weights(), prev_model.get_weights(), strict=True
                )
            ]
            delta_parameters = delta_parameters * (1 / client.conf.client.eta)  ## QFFL

            return delta_parameters, client.x_train.shape[0], fit_response

        return client.model.get_weights(), client.x_train.shape[0], fit_response

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
        client.selected = Utils.is_select_by_server(
            client.cid, config["selected_by_server"].split(",")
        )
        if client.selected:
            client.model.set_weights(parameters)

        loss, acc = client.model.evaluate(client.x_test, client.y_test)
        client.g_eval_acc = np.mean(acc)
        client.g_eval_loss = np.mean(loss)
        eval_resp = {
            "cid": client.cid,
            "acc": client.g_eval_acc,
            "loss": client.g_eval_loss,
            "participating_state": client.participating_state,
            "desired_state": client.participating_state,
        }

        return loss, client.x_test.shape[0], eval_resp
