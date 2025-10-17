from typing import TYPE_CHECKING

import numpy as np

from ..base import ClientSelectionMethod

if TYPE_CHECKING:
    from ...fl_server import FLServer


class LetsFedSelection(ClientSelectionMethod):
    """
    LetsFed client selection strategy.
    Clients are selected based on their participation status.
    Non-participating clients are selected using one strategy,
    while participating clients are selected using another strategy.
    """

    def init(self, server: FLServer) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            server: The federated learning server instance.
        """
        pass

    def _select_participating_clients(
        self, server: FLServer, server_round: int, list_of_clients: list[str]
    ) -> list[str]:
        """
        Select participating clients based on their performance.

        Args:
            server: The federated learning server instance.
            server_round: The current round number.
            list_of_clients: List of available client IDs.

        Returns:
            A list of selected client IDs.
        """
        return self.participating.select(
            server=server, server_round=server_round, list_of_clients=list_of_clients
        )

    def _select_non_participating_clients(
        self, server: FLServer, server_round: int, list_of_clients: list[str]
    ) -> list[str]:
        """
        Select non-participating clients based on their performance.

        Args:
            server: The federated learning server instance.
            server_round: The current round number.
            list_of_clients: List of available client IDs.

        Returns:
            A list of selected client IDs.
        """
        return self.non_participating.select(
            server=server, server_round=server_round, list_of_clients=list_of_clients
        )

    def select(self, server: FLServer, server_round: int, list_of_clients: list[str]) -> list[str]:
        """
        Select clients based on LetsFed strategy.

        Args:
            server: The federated learning server instance.
            server_round: The current round number.
            list_of_clients: List of available client IDs.

        Returns:
            A list of selected client IDs.
        """
        non_participating_clients = np.where(not server.client_participating_state)[0].astype(str)
        participating_clients = np.where(server.client_participating_state)[0].astype(str)

        non_participating_clients_selected = self._select_non_participating_clients(
            server=server,
            server_round=server_round,
            list_of_clients=non_participating_clients.tolist(),
        )

        participating_clients_selected = self._select_participating_clients(
            server=server, server_round=server_round, list_of_clients=participating_clients.tolist()
        )

        return non_participating_clients_selected + participating_clients_selected
