import os
from random import randint
from typing import TYPE_CHECKING

from flwr.common import (
    Config,
    NDArrays,
)

from .driver import Driver

if TYPE_CHECKING:
    from ..fl_client import FLClient

IDLE = int(os.environ.get("IDLE_STATE", "0"))
EXPLORING = int(os.environ.get("EXPLORING_STATE", "1"))


class CuriosityDriver(Driver):
    """
    Driver for curiosity-based client selection.
    """

    def run(self, client: FLClient, parameters: NDArrays, config: Config) -> None:
        """
        Run the driver with the given client, parameters, and config.

        Args:
            client (FLClient): The federated learning client.
            parameters (NDArrays): The model parameters.
            config (Config): Configuration dictionary.
        """
        if not client.selected:
            state = True if client.state == EXPLORING else False

        elif self.on_exploration(client=client):
            state = self.explore(client=client)

        elif client.participating_state:
            state = self.start_exploration(client=client)

        else:
            state = False

        client.curiosity = state

    def explore(self, client: FLClient) -> bool:
        """
        Explore the environment.
        """
        client.rounds_intention -= 1
        if not self.on_exploration(client=client):
            self.set_idle(client=client)
            return False
        return True

    def on_exploration(self, client: FLClient) -> bool:
        """
        Check if the client is currently exploring.
        """
        return client.rounds_intention > 0 and client.state == EXPLORING

    def start_exploration(self, client: FLClient) -> bool:
        """
        Start the exploration phase for the client.

        Args:
            client (FLClient): The federated learning client.
        """
        if client.rounds_intention == 0:
            client.rounds_intention = randint(1, int(client.conf.rounds) + 1)  # noqa: S311

        self.set_exploring(client=client)
        return True

    def set_exploring(self, client: FLClient) -> None:
        """
        Set the client state to exploring.
        """
        client.state = EXPLORING

    def set_idle(self, client: FLClient) -> None:
        """
        Set the client state to idle.
        """
        client.rounds_intention = 0
        client.state = IDLE
