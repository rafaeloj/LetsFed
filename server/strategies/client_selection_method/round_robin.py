from .base import ClientSelectionMethod
from typing import TYPE_CHECKING, List
import numpy as np

if TYPE_CHECKING:
    from server.strategies import FLServer

class RoundRobinSelection(ClientSelectionMethod):
    def select(
        self,
        server: 'FLServer',
        server_round: int,
        list_of_clients: List[str],
    ) -> List[str]:
        if server_round == 1:
            server.how_many_time_selected = np.zeros(server.conf.n_clients)
            return list_of_clients

        clients_cid_int = [int(cid) for cid in list_of_clients]

        # Pega a quantidade de clientes que querem participar dentro do contador
        how_many_time_selected_client = server.how_many_time_selected[clients_cid_int]

        # Aqui basicamente eu to pegando um array de indices ordenados pelos valorres do how_many_time_selected_client
        sort_cids = np.argsort(how_many_time_selected_client)
        
        # Pegando o top menos chamados
        top_values_of_cid  = sort_cids[:int(len(how_many_time_selected_client)*server.conf.server.selection.perc_of_clients)]

        # Update score
        for cid_value_index in top_values_of_cid:
            server.how_many_time_selected[
                cid_value_index
            ] += 1
        
        # To pegando indice dos clientes que foram selecionados
        top_clients = [str(cid) for cid in top_values_of_cid]
        return top_clients