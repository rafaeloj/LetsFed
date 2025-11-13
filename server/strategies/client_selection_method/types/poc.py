from typing import TYPE_CHECKING

from ..base import ClientSelectionMethod
from ..structs import PoCSelectionMethodConfig

if TYPE_CHECKING:
    from ...fl_server import FLServer


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
        if server_round == 1:
            return list_of_clients

        lc: list[tuple[str, float]] = [
            (cid, server.clients_acc[int(cid)]) for cid in list_of_clients
        ]
        lc.sort(key=lambda x: x[1])
        selected_clients = []
        for cid, acc in lc:
            if acc < server.clients_acc_avg:
                selected_clients.append(cid)

        clients2select = int(float(len(list_of_clients)) * float(self.config.perc_of_clients))

        return selected_clients[:clients2select]
