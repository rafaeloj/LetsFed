from logging import getLogger
from typing import TYPE_CHECKING, Any, Optional, Union

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

from .....utils.utils import Utils
from ..base import AggregationMethod
from ..structs import MaxFLAggregationMethodConfig

if TYPE_CHECKING:
    from ...fl_server import FLServer

logger = getLogger(__name__)


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
        logger.info(
            "MaxFL aggregation initialized - "
            + f"lr={config.learning_rate}, epsilon={config.epsilon}"
        )

    @staticmethod
    def params_from_json(params: dict[str, Any]) -> MaxFLAggregationMethodConfig:
        """
        Parse JSON parameters into MaxFLAggregationMethodConfig.

        Args:
            params: Dictionary of parameters from YAML configuration.

        Returns:
            MaxFLAggregationMethodConfig instance.
        """
        return MaxFLAggregationMethodConfig(
            epsilon=params.get("epsilon", 10.0),
            learning_rate=params.get("learning_rate", 0.01),
        )

    def _get_learning_rate(self, q_models_value: list[float]) -> float:
        """
        Method to get the learning rate based on q_models values.

        Args:
            q_models_value: List of q_model values from clients.

        Returns:
            The calculated learning rate.
        """
        lr = self.config.learning_rate / (sum(q_models_value) + self.config.epsilon)
        logger.debug(f"Computed learning rate: {lr:.6f} (sum_qk={sum(q_models_value):.4f})")
        return lr

    def agg_fit(
        self,
        server: "FLServer",
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
        logger.info(
            f"Round {server_round}: MaxFL aggregating fit results from "
            + f"{len(results)} clients ({len(failures)} failures)"
        )

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
                    logger.debug(f"Round {server_round}: Client {cid} - qk={qk:.4f}, participating")
                else:
                    logger.debug(
                        f"Round {server_round}: Client {cid} - qk={qk:.4f}, not participating"
                    )  # noqa: E501

        if not weights_results:
            logger.warning(f"Round {server_round}: No participating clients for aggregation")
            return None, {}

        # Aggregate weights from selected and participating clients.
        avg_qk = qk_s / len(server.selected_clients)
        server.data_to_log["qk_s"] = avg_qk
        logger.info(
            f"Round {server_round}: Aggregating {len(weights_results)} client models "
            + f"(avg_qk={avg_qk:.4f})"
        )

        weights_avg = aggregate(weights_results)
        learning_rate = self._get_learning_rate(q_models_value)

        new_weights = [
            weight - (learning_rate * (weight - weight_avg))  # Gradient descent
            for weight, weight_avg in zip(server.model.get_weights(), weights_avg, strict=True)
        ]
        server.model.set_weights(new_weights)

        logger.info(
            f"Round {server_round}: MaxFL aggregation completed with lr={learning_rate:.6f}"
        )

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
            A tuple containing the aggregated loss and a dictionary of Scalar metrics.
        """
        logger.debug(
            f"Round {server_round}: MaxFL aggregating evaluation results from "
            + f"{len(results)} clients"
        )

        if not results:
            logger.warning(f"Round {server_round}: No evaluation results to aggregate")
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
