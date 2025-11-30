"""
Normal Parameters Strategy (Server).

This strategy shares all model parameters between server and clients.
"""

from typing import TYPE_CHECKING, Any

from flwr.common import NDArrays

from .....utils.logger import Logger
from ..base import ParametersStrategy
from ..structs import NormalParametersStrategyConfig

if TYPE_CHECKING:
    from ...fl_server import FLServer

logger = Logger(__name__)


class NormalParametersStrategy(ParametersStrategy):
    """Normal parameters strategy (server) - shares all parameters."""

    def __init__(self, config: NormalParametersStrategyConfig) -> None:
        super().__init__(config)
        logger.info("Server NormalParametersStrategy initialized - sharing all parameters")

    @staticmethod
    def params_from_json(params: dict[str, Any]) -> NormalParametersStrategyConfig:
        """
        Parse JSON parameters into NormalParametersStrategyConfig.

        Args:
            params: Dictionary of parameters from YAML configuration (not used for normal strategy).

        Returns:
            NormalParametersStrategyConfig instance.
        """
        return NormalParametersStrategyConfig()

    def get_parameters(self, server: "FLServer", parameters: NDArrays) -> NDArrays:
        """
        Get model parameters to send to client.

        This method determines which parameters from the server's global parameters
        should be sent to the client for aggregation.

        Args:
            server: The federated learning server instance.
            parameters: Full global model parameters.

        Returns:
            Model parameters as NDArrays (may be full or partial parameters).
        """
        logger.debug(f"Server: Sending all parameters ({len(parameters)} layers)")
        return parameters

    def set_parameters(self, server: "FLServer", parameters: NDArrays) -> None:
        """
        Set model parameters received from clients aggregation.

        This method determines how to apply the aggregated parameters.

        Args:
            server: The federated learning server instance.
            parameters: Model parameters received from clients.
        """
        logger.debug(f"Server: Setting parameters ({len(parameters)} layers)")
        server.model.set_weights(parameters)
