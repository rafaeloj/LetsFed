from .base import ClientSelectionMethod
from typing import TYPE_CHECKING, List
import numpy as np
if TYPE_CHECKING:
    from .. import FLServer

class LetsFedSelection(ClientSelectionMethod):
    def __init__(self, participating: ClientSelectionMethod, non_participating: ClientSelectionMethod):
        self.participating = participating
        self.non_participating = non_participating

    def select(
        self,
        server: 'FLServer',
        server_round: int,
        list_of_clients: List[str]
    ) -> List[str]:
        non_participating_clients = np.where(server.client_participating_state == False)[0].astype(str)
        participating_clients = np.where(server.client_participating_state == True)[0].astype(str)

        non_participating_clients_selected = self.non_participating.select(
            server = server,
            server_round = server_round,
            list_of_clients = non_participating_clients.tolist(),
        )

        participating_clients_selected = self.participating.select(
            server = server,
            server_round = server_round,
            list_of_clients = participating_clients.tolist()
        )

        return non_participating_clients_selected + participating_clients_selected