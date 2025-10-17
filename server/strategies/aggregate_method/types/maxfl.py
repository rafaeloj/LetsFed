from functools import reduce
from typing import TYPE_CHECKING, Optional, Union

import keras
import numpy as np
from flwr.common import (
    EvaluateRes,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg

from .....dataset_manager.dataset_manager import DSManager
from .....model.model_manager import ModelManager
from .....utils.utils import Utils
from ...fl_server import FLServer
from ..base import AggregateMethod

if TYPE_CHECKING:
    from .. import FLServer


class MaxFL(AggregateMethod):
    def init(self, server: FLServer) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            server: The federated learning server instance.
        """
        server.model = self.load_model(server)

    def load_model(self, server: FLServer) -> keras.Model:
        """
        Method to load the model for federated learning

        Args:
            server: The federated learning server instance.

        Returns:
            The Keras model instance.
        """
        dm = DSManager(n_clients=server.conf.n_clients, conf=server.conf.db)
        train, validation, test = dm.load_locally(partition_id=int(0))
        keys = list(test.features.keys())
        self.x_train, self.y_train, self.x_validation, self.y_validation = (
            train[keys[0]],
            train[keys[1]],
            validation[keys[0]],
            validation[keys[1]],
        )
        self.x_test, self.y_test = test[keys[0]], test[keys[1]]

        mm = ModelManager(server.conf, input_shape=self.x_train.shape, path="app")

        return mm.get_model()

    def _aggregate(self, weights_to_aggregate: list[tuple[NDArrays, float]]) -> NDArrays:
        """
        Method to aggregate model weights from clients

        Args:
            weights_to_aggregate: List of tuples containing model weights and their corresponding
                weights.

        Returns:
            The aggregated model weights.
        """
        s = np.sum(np.array([qk for _, qk in weights_to_aggregate]))
        weighted_weights = [[layer * qk for layer in layers] for layers, qk in weights_to_aggregate]

        parameters = [reduce(np.add, w) / s for w in zip(*weighted_weights, strict=True)]

        return parameters

    def agg_fit(
        self,
        server: FLServer,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Method to call in aggregate fit step

        Args:
            server: The federated learning server instance.
            server_round: The current round number.
            results: List of tuples of ClientProxy and FitRes from clients.
            failures: List of failures that occurred during client training.

        Returns:
            A tuple containing the aggregated Parameters and a dictionary of Scalar metrics.
        """
        weights_results = []
        qk_s: float = 0.0
        for _, fit_res in results:
            cid = fit_res.metrics["cid"]
            qk = fit_res.metrics["qk"]
            if fit_res.metrics["participating_state"]:
                if Utils.is_select_by_server(cid, server.selected_clients):
                    weights_results.append((parameters_to_ndarrays(fit_res.parameters), qk))
                    qk_s += qk

        server.data_to_log["qk_s"] = qk_s / len(server.selected_clients)
        weights_avg = self._aggregate(weights_results)
        new_weights = [
            weight + (1 * weight_avg)
            for weight, weight_avg in zip(server.model.get_weights(), weights_avg, strict=True)
        ]
        server.model.set_weights(new_weights)

        return ndarrays_to_parameters(new_weights), {}

    def agg_eval(
        self,
        server: "FLServer",
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """
        Method to call in aggregate evaluate step

        Args:
            server: The federated learning server instance.
            server_round: The current round number.
            results: List of tuples of ClientProxy and EvaluateRes from clients.
            failures: List of failures that occurred during client evaluation.

        Returns:
            A tuple containing the aggregated loss (float) and a dictionary of Scalar metrics.
        """
        if not results:
            return None, {}

        loss_to_aggregated = []
        participating_clients = []
        non_participating_clients = []
        for _, eval_res in results:
            client_id = eval_res.metrics["cid"]
            if eval_res.metrics["participating_state"]:
                participating_clients.append(client_id)
                if Utils.is_select_by_server(client_id, server.selected_clients):
                    loss_to_aggregated.append((eval_res.loss, eval_res.num_examples))
            else:
                non_participating_clients.append(client_id)

        should_pass = len(loss_to_aggregated) <= 1
        if should_pass:
            return None, {}

        loss_aggregated = weighted_loss_avg(loss_to_aggregated)
        return loss_aggregated, {}
