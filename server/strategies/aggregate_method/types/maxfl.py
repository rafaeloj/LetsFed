from typing import TYPE_CHECKING, Optional, Union

from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from .....conf.structs import MaxFLAggregationMethodConfig
from .....utils.utils import Utils
from ...fl_server import FLServer
from ..base import AggregationMethod

if TYPE_CHECKING:
    from .. import FLServer


class MaxFL(AggregationMethod):
    """
    MaxFL aggregation strategy.
    """

    def __init__(self, config: MaxFLAggregationMethodConfig) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            config: The aggregation method configuration.
        """
        super().__init__(config)

    def _get_learning_rate(self, q_models_value: list[float]) -> float:
        """
        Method to get the learning rate based on q_models values.

        Args:
            q_models_value: List of q_model values from clients.

        Returns:
            The calculated learning rate.
        """
        return self.config.learning_rate / (sum(q_models_value) + self.config.epsilon)

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
        # Construct weights results and q_models values
        weights_results = []
        q_models_value: list[float] = []
        qk_s: float = 0.0
        for _, fit_res in results:
            cid = fit_res.metrics["cid"]
            qk = fit_res.metrics["qk"]
            if Utils.is_select_by_server(cid, server.selected_clients):
                if fit_res.metrics["participating_state"]:
                    weights_results.append((parameters_to_ndarrays(fit_res.parameters), qk))
                    qk_s += qk
                    q_models_value.append(fit_res.metrics["qk"])

        # Aggregate weights from selected and participating clients.
        server.data_to_log["qk_s"] = qk_s / len(server.selected_clients)
        weights_avg = aggregate(weights_results)
        learning_rate = self._get_learning_rate(q_models_value)
        new_weights = [
            weight - (learning_rate * (weight - weight_avg))  # Gradient descent
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
        # Check if there are any results
        if not results:
            return None, {}

        # Aggregate loss from selected and participating clients
        loss_to_aggregated = []
        for _, eval_res in results:
            client_id = eval_res.metrics["cid"]
            if Utils.is_select_by_server(client_id, server.selected_clients):
                if eval_res.metrics["participating_state"]:
                    loss_to_aggregated.append((eval_res.loss, eval_res.num_examples))

        should_pass = len(loss_to_aggregated) <= 1
        if should_pass:
            return None, {}

        loss_aggregated = weighted_loss_avg(loss_to_aggregated)
        return loss_aggregated, {}
