"""
Layer-Wise Parameters Strategy.

This strategy shares only the first K layers between server and clients,
allowing for model personalization while reducing communication costs.
"""

from typing import TYPE_CHECKING, Any

from flwr.common import NDArrays

from .....utils.logger import Logger
from ..base import ParametersStrategy
from ..structs import LayerWiseParametersStrategyConfig

if TYPE_CHECKING:
    from ...fl_client import FLClient

logger = Logger(__name__)


class LayerWiseParametersStrategy(ParametersStrategy):
    """
    Layer-wise parameters strategy - shares only first K layers.

    In this strategy:
    - Server sends only the first K layers to clients (shared/global layers)
    - Clients update only the first K layers with received parameters
    - Remaining layers are personalized and not shared
    - Clients send only the first K layers back to server after training
    - Server aggregates only the first K layers from all clients

    This approach:
    - Reduces communication costs (fewer parameters transmitted)
    - Allows model personalization (last layers adapt to local data)
    - Maintains global knowledge in shared layers
    """

    def __init__(self, config: LayerWiseParametersStrategyConfig) -> None:
        """
        Initialize layer-wise parameters strategy.

        Args:
            config: The parameters strategy configuration.
        """
        super().__init__(config)
        self.num_shared_layers = config.num_shared_layers
        msg = "".join(
            [
                "LayerWiseParametersStrategy initialized - ",
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

        Returns:
            LayerWiseParametersStrategyConfig instance.
        """
        num_shared_layers = params.get("num_shared_layers", 5)
        return LayerWiseParametersStrategyConfig(num_shared_layers=num_shared_layers)

    def get_parameters(self, client: "FLClient") -> NDArrays:
        """
        Get only the first K layers' parameters from client.

        Returns only the parameters of the first K layers (shared layers)
        to send to the server. The personalized layers are kept local.

        Args:
            client: The federated learning client instance.

        Returns:
            First K layers' parameters as NDArrays.
        """
        all_parameters = client.model.get_weights()
        total_layers = len(all_parameters)

        # Determine how many layers to share (minimum of configured and actual)
        num_to_share = min(self.num_shared_layers, total_layers)

        # Get only the first K layers
        shared_parameters = all_parameters[:num_to_share]

        msg = "".join(
            [
                f"Client {client.cid}: Getting shared parameters ",
                f"({num_to_share}/{total_layers} layers)",
            ]
        )
        logger.debug(msg)

        return shared_parameters

    def set_parameters(self, client: "FLClient", parameters: NDArrays) -> None:
        """
        Set only the first K layers' parameters in client.

        Updates only the first K layers of the client model with parameters
        received from server. The personalized layers remain unchanged.

        Args:
            client: The federated learning client instance.
            parameters: First K layers' parameters received from server.
        """
        # Get current model parameters
        current_parameters = client.model.get_weights()
        total_layers = len(current_parameters)
        num_shared = len(parameters)

        msg = "".join(
            [
                f"Client {client.cid}: Setting shared parameters ",
                f"({num_shared}/{total_layers} layers)",
            ]
        )
        logger.debug(msg)

        # Create new parameter list: shared parameters + personal parameters
        new_parameters = list(parameters) + current_parameters[num_shared:]

        # Update model with combined parameters
        client.model.set_weights(new_parameters)

        msg = "".join(
            [
                f"Client {client.cid}: Model updated - ",
                f"shared layers: {num_shared}, personalized layers: {total_layers - num_shared}",
            ]
        )
        logger.debug(msg)
