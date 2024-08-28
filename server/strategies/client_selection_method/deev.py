from .base import ClientSelectionMethod
from typing import TYPE_CHECKING, List, Tuple
from math import ceil
import numpy as np
if TYPE_CHECKING:
    from .. import FLServer

class DEEV(ClientSelectionMethod):
    def select(
        self,
        server: 'FLServer',
        server_round: int,
        list_of_clients: List[str],
    ) -> List[str]:
        if server_round == 1:
            return list_of_clients
        selected_clients: List[str] = []
        # Cria alista de clientes com acurácia e ordena por acurácia
        lc: List[Tuple[str, float]] = [(cid, server.clients_acc[int(cid)]) for cid in list_of_clients]
        lc.sort(key = lambda x: x[1])
        for cid, acc in lc:
            if acc < server.clients_acc_avg:
                selected_clients.append(cid)

        if server.conf.server.selection.decay > 0.0:
            the_chosen_ones  = len(selected_clients) * (1 - server.conf.server.selection.decay)**int(server_round)
            selected_clients = selected_clients[ : ceil(the_chosen_ones)]
        return selected_clients
        