import math
import random
from typing import TYPE_CHECKING, List

from .....utils.logger import Logger
from ..base import ClientSelectionMethod
from ..structs import RandomSelectionMethodConfig

if TYPE_CHECKING:
    from ...fl_server import FLServer

logger = Logger(__name__)


class RandomSelection(ClientSelectionMethod):
    """
    Random client selection strategy.
    Clients are selected randomly in each round based on a specified percentage.
    """

    def __init__(self, config: RandomSelectionMethodConfig) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            config: The selection method configuration.
        """
        super().__init__(config)

    def select(
        self,
        server: "FLServer",
        server_round: int,
        list_of_clients: List[str],
    ) -> List[str]:
        """
        Select clients based on Random strategy.

        Args:
            server: The federated learning server instance.
            server_round: The current round number.
            list_of_clients: List of available client IDs.

        Returns:
            A list of selected client IDs.
        """
        if server_round == 1:
            logger.debug(
                f"Random selection (Round 1): Selecting all {len(list_of_clients)} clients"
            )
            return list_of_clients

        perc = math.ceil(len(list_of_clients) * self.config.perc_of_clients)
        selected_clients = random.sample(list_of_clients, perc)

        logger.debug(
            f"Random selection: {len(selected_clients)}/{len(list_of_clients)} clients "
            + f"({self.config.perc_of_clients * 100:.0f}%)"
        )

        return selected_clients
