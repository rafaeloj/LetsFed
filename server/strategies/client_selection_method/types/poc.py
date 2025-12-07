import math
from typing import TYPE_CHECKING, Any

from .....utils.logger import Logger
from ..base import ClientSelectionMethod
from ..structs import PoCSelectionMethodConfig

if TYPE_CHECKING:
    from ...fl_server import FLServer

logger = Logger(__name__)


class POC(ClientSelectionMethod):
    """
    POC client selection strategy.
    Clients are selected based on their performance and the overall client performance.
    Clients with performance below the average are prioritized for selection.
    """

    def __init__(self, config: PoCSelectionMethodConfig) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            config: The selection method configuration.
        """
        super().__init__(config)

    @staticmethod
    def params_from_json(params: dict[str, Any]) -> PoCSelectionMethodConfig:
        """
        Parse JSON parameters into PoCSelectionMethodConfig.

        Args:
            params: Dictionary of parameters from YAML configuration.

        Returns:
            PoCSelectionMethodConfig instance.
        """
        return PoCSelectionMethodConfig(perc_of_clients=params.get("perc_of_clients", 0.3))

    def select(
        self,
        server: "FLServer",
        server_round: int,
        list_of_clients: list[str],
    ) -> list[str]:
        """
        Select clients for the current round.

        Args:
            server: The federated learning server instance.
            server_round: The current round number.
            list_of_clients: List of available client IDs.

        Returns:
            A list of selected client IDs.
        """
        logger.debug(f"Round {server_round}: POC selecting from {len(list_of_clients)} clients")
        if server_round == 1:
            logger.info(f"Round {server_round}: POC selecting all clients")
            return list_of_clients

        # Get accuracy metric from clients_metrics
        # Try to use 'accuracy' first, fallback to other metrics if needed
        if "accuracy" in server.clients_metrics:
            metric_key = "accuracy"
        else:
            raise ValueError("POC selection requires 'accuracy' metric in clients_metrics")

        lc: list[tuple[str, float]] = [
            (cid, server.clients_metrics[metric_key][int(cid)]) for cid in list_of_clients
        ]
        lc.sort(key=lambda x: x[1])
        selected_clients = []

        # Get average accuracy from clients_metrics_avg
        avg_metric = server.clients_metrics_avg.get(metric_key)

        for cid, metric_value in lc:
            if metric_value <= avg_metric:
                selected_clients.append(cid)

        clients2select = math.ceil(float(len(list_of_clients)) * float(self.config.perc_of_clients))

        logger.info(
            f"Round {server_round}: POC selected {len(selected_clients[:clients2select])}/"
            + f"{len(list_of_clients)} clients"
        )

        return selected_clients[:clients2select]
