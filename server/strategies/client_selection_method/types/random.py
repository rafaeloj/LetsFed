import math
import random
from typing import TYPE_CHECKING, Any

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

    @staticmethod
    def params_from_json(params: dict[str, Any]) -> RandomSelectionMethodConfig:
        """
        Parse JSON parameters into RandomSelectionMethodConfig.

        Args:
            params: Dictionary of parameters from YAML configuration.

        Returns:
            RandomSelectionMethodConfig instance.
        """
        return RandomSelectionMethodConfig(perc_of_clients=params.get("perc_of_clients", 0.3))

    def select(
        self,
        server: "FLServer",
        server_round: int,
        list_of_clients: list[str],
    ) -> list[str]:
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

        # Create a deterministic Random instance for this specific round
        # This ensures reproducibility regardless of other random operations
        round_rng = random.Random(server.conf.seed + server_round)  # noqa: S311

        perc = math.ceil(len(list_of_clients) * self.config.perc_of_clients)
        selected_clients = round_rng.sample(list_of_clients, perc)

        logger.debug(
            f"Random selection (Round {server_round}): "
            + f"{len(selected_clients)}/{len(list_of_clients)} clients "
            + f"({self.config.perc_of_clients * 100:.0f}%) "
            + f"seed={server.conf.seed + server_round}"
        )

        return selected_clients
