from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .structs import SelectionMethodConfig

if TYPE_CHECKING:
    from ..fl_server import FLServer


class ClientSelectionMethod(ABC):
    """
    Abstract base class for client selection methods.
    """

    def __init__(self, config: SelectionMethodConfig) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            config: The selection method configuration.
        """
        self.config = config

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
