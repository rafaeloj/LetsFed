import os
from random import randint
from typing import TYPE_CHECKING

from flwr.common import (
    Config,
    NDArrays,
)

from .context import DriverContext
from .driver import Driver

if TYPE_CHECKING:
    from ..fl_client import FLClient

IDLE = int(os.environ.get("IDLE_STATE", "0"))
EXPLORING = int(os.environ.get("EXPLORING_STATE", "1"))


class CuriosityDriver(Driver):
    """
    (NOT TESTED)
    Driver for curiosity-based client selection.

    This driver manages the exploration and idle states of clients,
    implementing a curiosity-driven participation strategy.

    Modifies:
        - curiosity: Boolean indicating client's curiosity state
        - state: Client's current state (IDLE or EXPLORING)
        - rounds_intention: Number of rounds the client intends to participate
    """

    def run(
        self, client: FLClient, parameters: NDArrays, config: Config, context: DriverContext
    ) -> None:
        """
        Run the driver with the given client, parameters, and config.

        Args:
            client (FLClient): The federated learning client.
            parameters (NDArrays): The model parameters.
            config (Config): Configuration dictionary.
            context (DriverContext): Context for storing driver results.
        """
        if not client.selected:
            state = True if client.state == EXPLORING else False

        elif self.on_exploration(client=client):
            state = self.explore(client=client, context=context)

        elif client.get_participating_state():
            state = self.start_exploration(client=client, context=context)

        else:
            state = False

        context.set("curiosity", state)

    def explore(self, client: FLClient, context: DriverContext) -> bool:
        """
        Explore the environment.
        """
        client.rounds_intention -= 1
        if not self.on_exploration(client=client):
            self.set_idle(client=client, context=context)
            return False
        return True

    def on_exploration(self, client: FLClient) -> bool:
        """
        Check if the client is currently exploring.
        """
        return client.rounds_intention > 0 and client.state == EXPLORING

    def start_exploration(self, client: FLClient, context: DriverContext) -> bool:
        """
        Start the exploration phase for the client.

        Args:
            client (FLClient): The federated learning client.
            context (DriverContext): Context for storing driver results.
        """
        if client.rounds_intention == 0:
            rounds_intention = randint(1, int(client.conf.rounds) + 1)  # noqa: S311
            context.set("rounds_intention", rounds_intention)

        self.set_exploring(client=client, context=context)
        return True

    def set_exploring(self, client: FLClient, context: DriverContext) -> None:
        """
        Set the client state to exploring.
        """
        context.set("state", EXPLORING)

    def set_idle(self, client: FLClient, context: DriverContext) -> None:
        """
        Set the client state to idle.
        """
        context.set("rounds_intention", 0)
        context.set("state", IDLE)
