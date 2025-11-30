"""
Base class for parameters strategies.

This module defines the abstract interface for different parameter
sharing strategies between server and clients.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from flwr.common import NDArrays

from .structs import ParametersStrategyConfig

if TYPE_CHECKING:
    from ..fl_client import FLClient


class ParametersStrategy(ABC):
    """
    Abstract base class for parameters sharing strategies.

    This class defines the interface for different approaches to
    sharing model parameters between server and clients.
    """

    def __init__(self, config: ParametersStrategyConfig) -> None:
        """
        Initialize parameters strategy.

        Args:
            config: The parameters strategy configuration.
        """
        self.config = config

    @staticmethod
    @abstractmethod
    def params_from_json(params: dict[str, Any]) -> Any:  # noqa: ANN401
        """
        Parse JSON parameters into the strategy-specific config.

        Args:
            params: Dictionary of parameters from YAML configuration.

        Returns:
            Strategy-specific configuration dataclass instance.
        """
        ...

    @abstractmethod
    def get_parameters(self, client: "FLClient") -> NDArrays:
        """
        Get model parameters to send to server.

        This method determines which parameters from the client's model
        should be sent to the server for aggregation.

        Args:
            client: The federated learning client instance.

        Returns:
            Model parameters as NDArrays (may be full or partial parameters).
        """
        ...

    @abstractmethod
    def set_parameters(self, client: "FLClient", parameters: NDArrays) -> None:
        """
        Set model parameters received from server.

        This method determines how to apply the parameters received
        from the server to the client's model.

        Args:
            client: The federated learning client instance.
            parameters: Model parameters received from server.
        """
        ...
