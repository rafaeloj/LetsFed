import copy
import os
from typing import TYPE_CHECKING

import numpy as np
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)

from .....utils.utils import Utils
from ...drivers.accuracy import AccuracyDriver
from ...drivers.driver import Driver
from ..base import TrainingStrategy

if TYPE_CHECKING:
    from ...fl_client import FLClient

IDLE = int(os.environ.get("IDLE_STATE", "0"))
EXPLORING = int(os.environ.get("EXPLORING_STATE", "1"))


class LetsFedClient(TrainingStrategy):
    """
    Federated Learning strategy for the LetsFed framework.
    """

    def init(self, client: FLClient) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            client: The federated learning client instance.
        """
        client.desired_state = client.get_participating_state()
        client.state = IDLE
        client.rounds_intention = 1
        client.willing = True
        client.add_drivers(self._get_drivers())

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

    def _participating_fit(
        self, client: FLClient, parameters: NDArrays, config: Config
    ) -> NDArrays:
        """
        Train the model on the client data.

        Args:
            client (FLClient): The federated learning client.
            parameters (NDArrays): The model parameters.
            config (Config): Configuration dictionary.

        Returns:
            NDArrays: Updated model parameters.
        """
        if not client.get_participating_state():
            return self._non_participating_fit(client, parameters=parameters, config=config)

        client.set_parameters(parameters)
        history = client.model.fit(
            client.x_train, client.y_train, epochs=client.conf.client.epochs, verbose=0
        )
        client.g_fit_acc = np.mean(history.history["accuracy"])
        client.g_fit_loss = np.mean(history.history["loss"])
        new_parameters = client.get_parameters()

        return new_parameters

    def _non_participating_fit(
        self, client: FLClient, parameters: NDArrays, config: Config
    ) -> NDArrays:
        """
        Train the model on the client data.

        Args:
            client (FLClient): The federated learning client.
            parameters (NDArrays): The model parameters.
            config (Config): Configuration dictionary.

        Returns:
            NDArrays: Updated model parameters.
        """
        history = client.model.fit(
            client.x_train, client.y_train, epochs=client.conf.client.epochs, verbose=0
        )
        client.g_fit_acc = np.mean(history.history["accuracy"])
        client.g_fit_loss = np.mean(history.history["loss"])

        return parameters

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

        if client.selected:
            return (
                self._participating_fit(client=client, parameters=parameters, config=config),
                client.x_train.shape[0],
                fit_response,
            )

        return (
            self._non_participating_fit(client=client, parameters=parameters, config=config),
            client.x_train.shape[0],
            fit_response,
        )

    def _manager_client_state(self, client: FLClient) -> None:
        """
        Manage the client's participation state.

        Args:
            client (FLClient): The federated learning client.
        """
        if not client.willing and client.get_participating_state():
            client.set_participating_state(False)

        if client.willing and not client.get_participating_state():
            client.set_participating_state(True)

    def _participating_evaluate(
        self, client: FLClient, parameters: NDArrays, config: Config, eval_resp: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Evaluate the model on the client test data.

        Args:
            client (FLClient): The federated learning client.
            parameters (NDArrays): The model parameters.
            config (Config): Configuration dictionary.
            eval_resp (dict[str, Scalar]): Evaluation response dictionary.

        Returns:
            tuple[NDArrays, int, dict[str, Scalar]]: Loss, number of examples used for evaluation,
            and additional metrics.
        """
        if not client.get_participating_state():
            return self._non_participating_evaluate(client, parameters=parameters, config=config)

        # Evaluate the model (on a participant client)
        client.set_parameters(parameters)
        loss, acc = client.model.evaluate(client.x_test, client.y_test)
        client.g_eval_loss = np.mean(loss)
        client.g_eval_acc = np.mean(acc)

        eval_resp = {
            "acc": client.g_eval_acc,
            "loss": client.g_eval_loss,
            "r_intention": client.rounds_intention,
        }
        client.r_intention = client.rounds_intention

        return loss, client.x_test.shape[0], eval_resp

    def _non_participating_evaluate(
        self, client: FLClient, parameters: NDArrays, config: Config, eval_resp: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Evaluate the model on the client test data.

        Args:
            client (FLClient): The federated learning client.
            parameters (NDArrays): The model parameters.
            config (Config): Configuration dictionary.
            eval_resp (dict[str, Scalar]): Evaluation response dictionary.

        Returns:
            tuple[NDArrays, int, dict[str, Scalar]]: Loss, number of examples used for evaluation,
            and additional metrics.
        """
        loss, acc = client.model.evaluate(client.x_test, client.y_test)
        client.g_eval_acc = np.mean(loss)
        client.g_eval_loss = np.mean(acc)
        client.r_intention = client.rounds_intention

        eval_resp = {
            "acc": client.g_eval_acc,
            "loss": client.g_eval_loss,
            "r_intention": client.rounds_intention,
        }
        return loss, client.x_test.shape[0], eval_resp

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
        # Initialize variables
        eval_resp = {
            "cid": client.cid,
            "participating_state": client.get_participating_state(),
        }

        # Set global model weights and apply drivers
        client.g_model = copy.deepcopy(client.model)
        client.g_model.set_weights(parameters)
        client.apply_drivers(client, parameters, config)
        self._manager_client_state(client)

        # Update desired state
        client.desired_state = client.get_participating_state()
        eval_resp["desired_state"] = client.desired_state

        # Analyze if the client is selected
        client.selected = Utils.is_select_by_server(
            str(client.cid), config["selected_by_server"].split(",")
        )

        # Evaluate the model (on a participant client)
        if client.selected:
            loss, shape, eval_resp = self._participating_evaluate(
                client=client, parameters=parameters, config=config, eval_resp=eval_resp
            )
        # Evaluate the model (on a non-participant client)
        else:
            loss, shape, eval_resp = self._non_participating_evaluate(
                client=client, parameters=parameters, config=config, eval_resp=eval_resp
            )

        eval_resp["fit_acc"] = client.g_fit_acc

        return loss, shape, eval_resp
