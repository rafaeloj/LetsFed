from .base import ClientSelectionMethod
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from .. import FLServer

class POC(ClientSelectionMethod):
    def select(
        self,
        server: 'FLServer',
        server_round: int,
        list_of_clients: List[str],
    ) -> List[str]:
        if server_round == 0:
            return list_of_clients

        lc: List[Tuple[str, float]] = [(cid, server.clients_acc[int(cid)]) for cid in list_of_clients]
        lc.sort(key = lambda x: x[1])
        selected_clients = []
        for cid, acc in lc:
            if acc < server.clients_acc_avg:
                selected_clients.append(cid)

        clients2select        = int(float(len(list_of_clients)) * float(server.conf.server.selection.perc_of_clients))
        selected_clients: List[str] = [
            cid for cid, _ in lc[:clients2select]
        ]

        return selected_clients