"""
Normal Parameters Strategy.

This strategy shares all model parameters between server and clients.
This is the standard federated learning approach where the entire model
is synchronized.
"""

from typing import TYPE_CHECKING, Any

from flwr.common import NDArrays

from .....utils.logger import Logger
from ..base import ParametersStrategy
from ..structs import NormalParametersStrategyConfig

if TYPE_CHECKING:
    from ...fl_client import FLClient

logger = Logger(__name__)


class NormalParametersStrategy(ParametersStrategy):
    """
    Normal parameters strategy - shares all parameters.

    In this strategy:
    - Server sends all global model parameters to clients
    - Clients update their entire model with received parameters
    - Clients send all their model parameters back to server after training
    - Server aggregates all client parameters to create new global model
    """

    def __init__(self, config: NormalParametersStrategyConfig) -> None:
        """
        Initialize normal parameters strategy.

        Args:
            config: The parameters strategy configuration.
        """
        super().__init__(config)
        logger.info("NormalParametersStrategy initialized - sharing all parameters")

    @staticmethod
    def params_from_json(params: dict[str, Any]) -> NormalParametersStrategyConfig:
        """
        Parse JSON parameters into NormalParametersStrategyConfig.

        Args:
            params: Dictionary of parameters from YAML configuration.

        Returns:
            NormalParametersStrategyConfig instance.
        """
        # No parameters needed for normal strategy
        return NormalParametersStrategyConfig()

    def get_parameters(self, client: "FLClient") -> NDArrays:
        """
        Get all model parameters from client.

        Returns all weights from the client's model to send to the server.

        Args:
            client: The federated learning client instance.

        Returns:
            All model parameters as NDArrays.
        """
        parameters = client.model.get_weights()
        logger.debug(
            f"Client {client.cid}: Getting all parameters ({len(parameters)} layers/tensors)"
        )
        return parameters

    def set_parameters(self, client: "FLClient", parameters: NDArrays) -> None:
        """
        Set all model parameters in client.

        Updates the entire client model with parameters received from server.

        Args:
            client: The federated learning client instance.
            parameters: All model parameters received from server.
        """
        logger.debug(
            f"Client {client.cid}: Setting all parameters ({len(parameters)} layers/tensors)"
        )
        client.model.set_weights(parameters)
