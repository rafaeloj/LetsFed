from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..fl_server import FLServer


class ClientSelectionMethod(ABC):
    """
    Abstract base class for client selection methods.
    """

    @abstractmethod
    def init(self, server: FLServer) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            server: The federated learning server instance.
        """
        ...

    @abstractmethod
    def select(
        self,
        server: FLServer,
        server_round: int,
        list_of_clients: list[str],
    ) -> list[str]:
        """
        Select clients for the current round.

        Args:
            server: The federated learning server instance.
            server_round: The current round number.
            list_of_clients: List of available client IDs.

        Returns:
            A list of selected client IDs.
        """
        ...
