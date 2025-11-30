"""
Layer-Wise Parameters Strategy (Server).

Shares only the first K layers between server and clients.
"""

from typing import TYPE_CHECKING, Any

from flwr.common import NDArrays

from .....utils.logger import Logger
from ..base import ParametersStrategy
from ..structs import LayerWiseParametersStrategyConfig

if TYPE_CHECKING:
    from ...fl_server import FLServer

logger = Logger(__name__)


class LayerWiseParametersStrategy(ParametersStrategy):
    """Layer-wise parameters strategy (server) - shares only first K layers."""

    def __init__(self, config: LayerWiseParametersStrategyConfig) -> None:
        super().__init__(config)
        self.num_shared_layers = config.num_shared_layers
        msg = "".join(
            [
                "Server LayerWiseParametersStrategy initialized - ",
                f"sharing first {self.num_shared_layers} layers",
            ]
        )
        logger.info(msg)

    @staticmethod
    def params_from_json(params: dict[str, Any]) -> LayerWiseParametersStrategyConfig:
        """
        Parse JSON parameters into LayerWiseParametersStrategyConfig.

        Args:
            params: Dictionary of parameters from YAML configuration.
                Expected key: 'num_shared_layers' (int, default: 5).

        Returns:
            LayerWiseParametersStrategyConfig instance.
        """
        num_shared_layers = params.get("num_shared_layers", 5)
        return LayerWiseParametersStrategyConfig(num_shared_layers=num_shared_layers)

    def get_parameters(self, server: "FLServer", parameters: NDArrays) -> NDArrays:
        """
        Get model parameters to send to clients.

        For layerwise strategy, only the first K layers (shared layers) are sent
        to clients for training.

        Args:
            server: The federated learning server instance.
            parameters: Full global model parameters.

        Returns:
            First K layers of global parameters to send to clients.
        """
        total_layers = len(parameters)
        num_to_send = min(self.num_shared_layers, total_layers)
        shared_params = parameters[:num_to_send]

        msg = f"Server: Sending shared parameters ({num_to_send}/{total_layers} layers)"
        logger.debug(msg)
        return shared_params

    def set_parameters(self, server: "FLServer", parameters: NDArrays) -> None:
        """
        Set model parameters received from clients aggregation.

        For layerwise strategy, only the first K layers are received from aggregation.
        This method updates only those shared layers in the server model, keeping
        the personalized layers (K+1 onwards) unchanged.

        Args:
            server: The federated learning server instance.
            parameters: Aggregated shared parameters (first K layers) from clients.
        """
        # Get current full model weights
        current_weights = server.model.get_weights()

        # Combine: aggregated shared layers + unchanged personalized layers
        num_shared = len(parameters)
        new_weights = list(parameters) + current_weights[num_shared:]

        msg = "".join(
            [
                "Server: Setting parameters - ",
                f"updated {num_shared} shared layers, ",
                f"kept {len(current_weights) - num_shared} personalized layers unchanged",
            ]
        )
        logger.debug(msg)

        # Update model with combined weights
        server.model.set_weights(new_weights)
