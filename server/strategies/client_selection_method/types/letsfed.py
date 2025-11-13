from typing import TYPE_CHECKING

import numpy as np

from .....utils.logger import Logger
from ..base import ClientSelectionMethod
from ..structs import LetsFedSelectionMethodConfig

if TYPE_CHECKING:
    from ...fl_server import FLServer

logger = Logger(__name__)


class LetsFedSelection(ClientSelectionMethod):
    """
    LetsFed client selection strategy.
    Clients are selected based on their participation status.
    Non-participating clients are selected using one strategy,
    while participating clients are selected using another strategy.
    """

    def __init__(self, config: LetsFedSelectionMethodConfig) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            config: The selection method configuration.
        """
        super().__init__(config)

        # Import here to avoid circular dependency
        from ..factory import ClientSelectionFactory

        self.participating_selection_method = ClientSelectionFactory.create(
            config.participating_selection_method
        )
        self.non_participating_selection_method = ClientSelectionFactory.create(
            config.non_participating_selection_method
        )

        logger.info(
            "LetsFedSelection initialized - "
            + f"Participating: {config.participating_selection_method.__class__.__name__}, "
            + f"Non-participating: {config.non_participating_selection_method.__class__.__name__}"
        )

    def _select_participating_clients(
        self, server: "FLServer", server_round: int, list_of_clients: list[str]
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
        logger.debug(
            f"Round {server_round}: Selecting from {len(list_of_clients)} participating clients"
        )
        selected = self.participating_selection_method.select(
            server=server, server_round=server_round, list_of_clients=list_of_clients
        )
        logger.info(f"Round {server_round}: Selected {len(selected)} participating clients")
        return selected

    def _select_non_participating_clients(
        self, server: "FLServer", server_round: int, list_of_clients: list[str]
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
        logger.debug(
            f"Round {server_round}: Selecting from {len(list_of_clients)} "
            + "non-participating clients"
        )
        selected = self.non_participating_selection_method.select(
            server=server, server_round=server_round, list_of_clients=list_of_clients
        )
        logger.info(f"Round {server_round}: Selected {len(selected)} non-participating clients")
        return selected

    def select(
        self, server: "FLServer", server_round: int, list_of_clients: list[str]
    ) -> list[str]:
        """
        Select clients based on LetsFed strategy.

        Args:
            server: The federated learning server instance.
            server_round: The current round number.
            list_of_clients: List of available client IDs.

        Returns:
            A list of selected client IDs.
        """
        logger.debug(f"Round {server_round}: Starting LetsFed client selection")

        non_participating_clients = np.where(not server.client_participating_state)[0].astype(str)
        participating_clients = np.where(server.client_participating_state)[0].astype(str)

        logger.info(
            f"Round {server_round}: {len(participating_clients)} participating, "
            + f"{len(non_participating_clients)} non-participating clients"
        )

        non_participating_clients_selected = self._select_non_participating_clients(
            server=server,
            server_round=server_round,
            list_of_clients=non_participating_clients.tolist(),
        )

        participating_clients_selected = self._select_participating_clients(
            server=server, server_round=server_round, list_of_clients=participating_clients.tolist()
        )

        server.data_to_log["participating_clients_selected"] = participating_clients_selected
        server.data_to_log["non_participating_clients_selected"] = (
            non_participating_clients_selected
        )

        total_selected = non_participating_clients_selected + participating_clients_selected
        logger.info(
            f"Round {server_round}: LetsFed selected {len(total_selected)} total clients "
            + f"({len(participating_clients_selected)} participating + "
            + f"{len(non_participating_clients_selected)} non-participating)"
        )

        return total_selected
