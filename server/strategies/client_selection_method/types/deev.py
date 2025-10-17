from math import ceil
from typing import TYPE_CHECKING

from ..base import ClientSelectionMethod

if TYPE_CHECKING:
    from ...fl_server import FLServer


class DEEV(ClientSelectionMethod):
    """
    DEEV client selection strategy.
    Clients are selected based on their performance and the overall client performance.
    Clients with performance below the average are prioritized for selection.

    Additionally, a decay factor is applied to gradually reduce the number of selected clients
    over rounds.
    """

    def init(self, server: FLServer) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            server: The federated learning server instance.
        """
        pass

    def select(
        self,
        server: FLServer,
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
        if server_round == 1:
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

        if server.conf.server.selection_method.decay > 0.0:
            the_chosen_ones = len(selected_clients) * (
                1 - server.conf.server.selection_method.decay
            ) ** int(server_round)
            selected_clients = selected_clients[: ceil(the_chosen_ones)]

        return selected_clients
