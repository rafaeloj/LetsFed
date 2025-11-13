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

from .....utils.logger import Logger
from .....utils.utils import Utils
from ..base import AggregationMethod
from ..structs import FedAvgAggregationMethodConfig

if TYPE_CHECKING:
    from ...fl_server import FLServer

logger = Logger(__name__)


class FedAVG(AggregationMethod):
    """
    Federated Averaging (FedAvg) aggregation strategy.
    """

    def __init__(self, config: FedAvgAggregationMethodConfig) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            config: The aggregation method configuration.
        """
        super().__init__(config)

    def agg_fit(
        self,
        server: "FLServer",
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
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
        logger.debug(f"FedAvg aggregation: Processing {len(results)} training results")

        # Check if there are any results
        if not results:
            logger.warning("FedAvg aggregation: No results to aggregate")
            return None, {}

        # Aggregate weights from selected and participating clients
        weights_results = []
        for _, fit_res in results:
            cid = fit_res.metrics["cid"]
            if Utils.is_select_by_server(cid, server.selected_clients):
                if fit_res.metrics["participating_state"]:
                    weights_results.append(
                        (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                    )

        if len(weights_results) == 0:
            logger.warning("FedAvg aggregation: No participating clients to aggregate")
            return None, {}

        logger.debug(f"FedAvg aggregation: Aggregating {len(weights_results)} client models")
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        return parameters_aggregated, {}

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
