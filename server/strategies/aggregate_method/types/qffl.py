from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg

from .....utils.logger import Logger
from .....utils.utils import Utils
from ..base import AggregationMethod
from ..structs import QFFLAggregationMethodConfig

if TYPE_CHECKING:
    from ...fl_server import FLServer

logger = Logger(__name__)


class QFFL(AggregationMethod):
    """
    q-Fair Federated Learning (q-FFL) aggregation strategy.

    Aggregates delta parameters from clients and applies them to the global model.
    Uses loss^q weighting to achieve fairness across clients with different data distributions.
    """

    def __init__(self, config: QFFLAggregationMethodConfig) -> None:
        """
        Initialize QFFL aggregation method.

        Args:
            config: The aggregation method configuration with q parameter.
        """
        super().__init__(config)
        logger.info(f"QFFL aggregation initialized with q={config.q}")

    @staticmethod
    def params_from_json(params: dict[str, Any]) -> QFFLAggregationMethodConfig:
        """
        Parse JSON parameters into QFFLAggregationMethodConfig.

        Args:
            params: Dictionary of parameters from YAML configuration.

        Returns:
            QFFLAggregationMethodConfig instance.
        """
        return QFFLAggregationMethodConfig(
            q=params.get("q", 0.0),
        )

    def agg_fit(
        self,
        server: "FLServer",
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Aggregate delta parameters from QFFL clients.

        QFFL clients return delta parameters (curr_params - prev_params) * (1/eta).
        This method:
        1. Collects deltas from participating clients
        2. Weights them by loss^q for fairness
        3. Applies aggregated delta to global model

        Args:
            server: The federated learning server instance.
            server_round: The current round number.
            results: List of tuples of ClientProxy and FitRes from clients.
            failures: List of failures that occurred during client training.

        Returns:
            A tuple containing the updated Parameters and a dictionary of Scalar metrics.
        """
        logger.info(
            f"Round {server_round}: QFFL aggregating fit results from "
            + f"{len(results)} clients ({len(failures)} failures)"
        )

        if not results:
            logger.warning(f"Round {server_round}: No results to aggregate")
            return None, {}

        # Sort results by client ID for reproducibility
        sorted_results = sorted(results, key=lambda x: int(x[1].metrics.get("cid", 0)))
        cid_order = [r[1].metrics.get("cid") for r in sorted_results]
        logger.debug(f"Round {server_round}: Processing clients in order: {cid_order}")

        # Collect deltas and losses from participating clients
        deltas = []
        losses = []
        num_examples = []
        participating_clients = []

        for _, fit_res in sorted_results:
            cid = fit_res.metrics["cid"]
            if Utils.is_select_by_server(cid, server.selected_clients):
                if fit_res.metrics["participating_state"]:
                    delta = parameters_to_ndarrays(fit_res.parameters)
                    loss = fit_res.metrics.get("loss", 1.0)  # Training loss
                    n_examples = fit_res.num_examples

                    deltas.append(delta)
                    losses.append(loss)
                    num_examples.append(n_examples)
                    participating_clients.append(cid)

                    logger.debug(
                        f"Round {server_round}: Client {cid} - "
                        + f"loss={loss:.4f}, n_examples={n_examples}"
                    )

        if not deltas:
            logger.warning(f"Round {server_round}: No participating clients to aggregate")
            return None, {}

        logger.info(
            f"Round {server_round}: Aggregating deltas from {len(deltas)} clients "
            + f"(clients: {participating_clients})"
        )

        # Compute q-FFL weights: weight_i = loss_i^q
        # Higher q -> more weight to clients with higher loss (more fairness)
        # q=0 -> uniform weighting (standard FedAvg)
        q = self.config.q
        if q == 0:
            # Uniform weighting
            weights = np.ones(len(losses))
        else:
            # q-FFL weighting
            weights = np.array([loss**q for loss in losses])

        # Normalize weights
        weights = weights / weights.sum()

        logger.debug(
            f"Round {server_round}: q-FFL weights (q={q}): " + f"{[f'{w:.4f}' for w in weights]}"
        )

        # Aggregate deltas using weighted average
        aggregated_delta = []
        for layer_idx in range(len(deltas[0])):
            layer_deltas = [delta[layer_idx] for delta in deltas]
            weighted_layer_delta = sum(
                w * delta for w, delta in zip(weights, layer_deltas, strict=True)
            )
            aggregated_delta.append(weighted_layer_delta)

        # Apply aggregated delta to global model
        current_params = server.model.get_weights()
        new_params = [
            param + delta for param, delta in zip(current_params, aggregated_delta, strict=True)
        ]

        server.model.set_weights(new_params)

        # Log statistics
        avg_loss = np.mean(losses)
        logger.info(
            f"Round {server_round}: QFFL aggregation completed "
            + f"(avg_loss={avg_loss:.4f}, q={q})"
        )

        return ndarrays_to_parameters(new_params), {}

    def agg_eval(
        self,
        server: "FLServer",
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients.

        Args:
            server: The federated learning server instance.
            server_round: The current round number.
            results: List of tuples of ClientProxy and EvaluateRes from clients.
            failures: List of failures that occurred during client evaluation.

        Returns:
            A tuple containing the aggregated loss and a dictionary of Scalar metrics.
        """
        logger.debug(
            f"Round {server_round}: QFFL aggregating evaluation results from "
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

        if len(loss_to_aggregated) <= 1:
            return None, {}

        loss_aggregated = weighted_loss_avg(loss_to_aggregated)
        logger.debug(f"Round {server_round}: Aggregated loss = {loss_aggregated:.4f}")
        return loss_aggregated, {}
