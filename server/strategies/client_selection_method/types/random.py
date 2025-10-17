import random
from typing import TYPE_CHECKING, List

from ..base import ClientSelectionMethod

if TYPE_CHECKING:
    from ...fl_server import FLServer


class RandomSelection(ClientSelectionMethod):
    """
    Random client selection strategy.
    Clients are selected randomly in each round based on a specified percentage.
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
        list_of_clients: List[str],
    ) -> List[str]:
        """
        Select clients based on Random strategy.

        Args:
            server: The federated learning server instance.
            server_round: The current round number.
            list_of_clients: List of available client IDs.

        Returns:
            A list of selected client IDs.
        """
        if server_round == 1:
            return list_of_clients

        perc = int(len(list_of_clients) * server.conf.server.selection_method.perc_of_clients)
        selected_clients = random.sample(list_of_clients, perc)

        return selected_clients
