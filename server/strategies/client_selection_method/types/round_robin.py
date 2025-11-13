from typing import TYPE_CHECKING, List

import numpy as np

from ....utils.logger import Logger
from ..base import ClientSelectionMethod
from ..structs import SelectionMethodConfig

if TYPE_CHECKING:
    from ...fl_server import FLServer

logger = Logger(__name__)


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
        logger.info(f"RoundRobinSelection initialized for {config.n_clients} clients")

    def select(
        self,
        server: "FLServer",
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
        logger.debug(
            f"Round {server_round}: RoundRobin selecting from {len(list_of_clients)} clients"
        )

        if server_round == 1:
            logger.info(
                f"Round {server_round}: First round - selecting all {len(list_of_clients)} clients"
            )
            return list_of_clients

        clients_cid_int = [int(cid) for cid in list_of_clients]

        # Get how many times each client has been selected
        how_many_time_selected_client = self.how_many_time_selected[clients_cid_int]

        # Here I basically get an array of indices sorted by the values
        # of how_many_time_selected_client
        sort_cids = np.argsort(how_many_time_selected_client)

        # Get the top least called
        num_to_select = int(len(how_many_time_selected_client) * self.config.perc_of_clients)
        top_values_of_cid = sort_cids[:num_to_select]

        logger.debug(
            f"Round {server_round}: Selecting {num_to_select} least-selected clients "
            + f"({self.config.perc_of_clients * 100:.1f}% of {len(list_of_clients)})"
        )

        # Update scores
        for cid_value_index in top_values_of_cid:
            self.how_many_time_selected[cid_value_index] += 1

        # Get the indices of the selected clients
        top_clients = [str(cid) for cid in top_values_of_cid]

        logger.info(
            f"Round {server_round}: RoundRobin selected {len(top_clients)} clients - "
            + f"CIDs: {', '.join(top_clients[:5])}{'...' if len(top_clients) > 5 else ''}"
        )

        return top_clients
