from .base import ClientSelectionMethod
from typing import TYPE_CHECKING, List
import random as rnd
if TYPE_CHECKING:
    from .. import FLServer

class RandomSelection(ClientSelectionMethod):
    def select(
        self,
        server: 'FLServer',
        server_round: int,
        list_of_clients: List[str],
    ) -> List[str]:
        if server_round == 1:
            return list_of_clients
        perc = int(len(list_of_clients)*server.conf.server.selection.perc_of_clients)
        selected_clients = rnd.sample(list_of_clients, perc)
        return selected_clients