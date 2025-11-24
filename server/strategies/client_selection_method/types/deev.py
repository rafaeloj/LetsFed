from math import ceil
from typing import TYPE_CHECKING, Any

from .....utils.logger import Logger
from ..base import ClientSelectionMethod
from ..structs import DeevSelectionMethodConfig

if TYPE_CHECKING:
    from ...fl_server import FLServer

logger = Logger(__name__)


class DEEV(ClientSelectionMethod):
    """
    DEEV client selection strategy.
    Clients are selected based on their performance and the overall client performance.
    Clients with performance below the average are prioritized for selection.

    Additionally, a decay factor is applied to gradually reduce the number of selected clients
    over rounds.
    """

    def __init__(self, config: DeevSelectionMethodConfig) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            config: The selection method configuration.
        """
        super().__init__(config)

    @staticmethod
    def params_from_json(params: dict[str, Any]) -> DeevSelectionMethodConfig:
        """
        Parse JSON parameters into DeevSelectionMethodConfig.

        Args:
            params: Dictionary of parameters from YAML configuration.

        Returns:
            DeevSelectionMethodConfig instance.
        """
        return DeevSelectionMethodConfig(decay=params.get("decay", 0.95))

    def select(
        self,
        server: "FLServer",
        server_round: int,
        list_of_clients: list[str],
    ) -> list[str]:
        """
        Select clients based on DEEV strategy.

        Args:
            server: The federated learning server instance.
            server_round: The current round number.
            list_of_clients: List of available client IDs.

        Returns:
            A list of selected client IDs.
        """
        logger.debug(f"Round {server_round}: DEEV selecting from {len(list_of_clients)} clients")
        if server_round == 1:
            logger.info(f"Round {server_round}: DEEV selecting all clients")
            return list_of_clients

        # Select clients with accuracy below average
        selected_clients: list[str] = []
        lc: list[tuple[str, float]] = [
            (cid, server.clients_acc[int(cid)]) for cid in list_of_clients
        ]
        lc.sort(key=lambda x: x[1])
        for cid, acc in lc:
            if acc < server.clients_acc_avg:
                selected_clients.append(cid)

        if self.config.decay > 0.0:
            the_chosen_ones = len(selected_clients) * (1 - self.config.decay) ** int(server_round)
            selected_clients = selected_clients[: ceil(the_chosen_ones)]

        logger.info(
            f"Round {server_round}: DEEV selected {len(selected_clients)}/"
            + f"{len(list_of_clients)} clients"
        )

        return selected_clients
