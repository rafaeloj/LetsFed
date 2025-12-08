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

from .....utils.logger import Logger
from .....utils.utils import Utils
from ..base import AggregationMethod
from ..structs import LetsFedAggregationMethodConfig

if TYPE_CHECKING:
    from ...fl_server import FLServer

logger = Logger(__name__)


class LetsFed(AggregationMethod):
    """
    LetsFed aggregation strategy.
    """

    def __init__(self, config: LetsFedAggregationMethodConfig) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            config: The aggregation method configuration.
        """
        super().__init__(config)

    @staticmethod
    def params_from_json(params: dict[str, Any]) -> LetsFedAggregationMethodConfig:
        """
        Create config from dictionary (from YAML).

        Args:
            params: Dictionary of parameters from YAML

        Returns:
            LetsFedAggregationMethodConfig instance
        """
        # LetsFed has no extra parameters
        return LetsFedAggregationMethodConfig()

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
        logger.debug(f"LetsFed aggregation: Processing {len(results)} training results")

        # Check if there are any results
        if not results:
            logger.warning("LetsFed aggregation: No results to aggregate")
            return None, {}

        # Sort results by client ID for reproducibility
        # This ensures deterministic aggregation order regardless of when clients finish training
        sorted_results = sorted(results, key=lambda x: int(x[1].metrics.get("cid", 0)))
        cid_order = [r[1].metrics.get("cid") for r in sorted_results]
        logger.debug(f"LetsFed aggregation: Processing clients in order: {cid_order}")

        # Aggregate weights from selected and participating clients
        weights_results = []
        for _, fit_res in sorted_results:
            cid = fit_res.metrics["cid"]
            # interest_metric = fit_res.metrics["interest_metric"]
            if Utils.is_select_by_server(cid, server.selected_clients):
                if fit_res.metrics["participating_state"]:
                    logger.debug(
                        f"LetsFed aggregation: Client {cid} participating in round {server_round}"
                    )
                    weights_results.append(
                        (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                    )
                else:
                    logger.debug(
                        f"LetsFed aggregation: Client {cid} not participating in round {server_round}"  # noqa: E501
                    )
                    weights_results.append(
                        (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                    )

        if len(weights_results) == 0:
            logger.warning("LetsFed aggregation: No participating clients to aggregate")
            return None, {}

        logger.debug(f"LetsFed aggregation: Aggregating {len(weights_results)} client models")
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

        # Sort results by client ID for reproducibility
        sorted_results = sorted(results, key=lambda x: int(x[1].metrics.get("cid", 0)))

        # Aggregate loss from selected and participating clients
        loss_to_aggregated = []
        for _, eval_res in sorted_results:
            client_id = eval_res.metrics["cid"]
            if Utils.is_select_by_server(client_id, server.selected_clients):
                if eval_res.metrics["participating_state"]:
                    logger.debug(
                        f"LetsFed aggregation: Client {client_id} participating in round {server_round}"  # noqa: E501
                    )
                    loss_to_aggregated.append((eval_res.loss, eval_res.num_examples))
                else:
                    logger.debug(
                        f"LetsFed aggregation: Client {client_id} not participating in round {server_round}"  # noqa: E501
                    )
                    loss_to_aggregated.append((eval_res.loss, eval_res.num_examples))

        should_pass = len(loss_to_aggregated) <= 1
        if should_pass:
            return None, {}

        loss_aggregated = weighted_loss_avg(loss_to_aggregated)
        return loss_aggregated, {}
