from typing import TYPE_CHECKING, List

import numpy as np

from .....conf.structs import SelectionMethodConfig
from ..base import ClientSelectionMethod

if TYPE_CHECKING:
    from ...fl_server import FLServer


class RoundRobinSelection(ClientSelectionMethod):
    """
    Round Robin client selection strategy.
    Each client is selected in a cyclic order to ensure fair participation.

    This strategy keeps track of how many times each client has been selected
    and selects clients that have been selected the least number of times in each round.

    This strategy requires the server to maintain a count of selections for each client.
    """

    def __init__(self, config: SelectionMethodConfig) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            config: The selection method configuration.
        """
        super().__init__(config)
        self.how_many_time_selected = np.zeros(config.n_clients)

    def select(
        self,
        server: FLServer,
        server_round: int,
        list_of_clients: List[str],
    ) -> List[str]:
        """
        Select clients based on Round Robin strategy.

        Args:
            server: The federated learning server instance.
            server_round: The current round number.
            list_of_clients: List of available client IDs.

        Returns:
            A list of selected client IDs.
        """
        if server_round == 1:
            return list_of_clients

        clients_cid_int = [int(cid) for cid in list_of_clients]

        # Get how many times each client has been selected
        how_many_time_selected_client = self.how_many_time_selected[clients_cid_int]

        # Here I basically get an array of indices sorted by the values
        # of how_many_time_selected_client
        sort_cids = np.argsort(how_many_time_selected_client)

        # Get the top least called
        top_values_of_cid = sort_cids[
            : int(len(how_many_time_selected_client) * self.config.perc_of_clients)
        ]

        # Update scores
        for cid_value_index in top_values_of_cid:
            self.how_many_time_selected[cid_value_index] += 1

        # Get the indices of the selected clients
        top_clients = [str(cid) for cid in top_values_of_cid]
        return top_clients
