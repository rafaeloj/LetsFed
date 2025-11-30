"""
Base class for parameters strategies (Server).

This module defines the abstract interface for different parameter
sharing strategies on the server side.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from flwr.common import NDArrays

from .structs import ParametersStrategyConfig

if TYPE_CHECKING:
    from ..fl_server import FLServer


class ParametersStrategy(ABC):
    """
    Abstract base class for parameters sharing strategies (server-side).

    This class defines the interface for different approaches to
    sharing model parameters between server and clients from the server's perspective.
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
        ...

    @abstractmethod
    def set_parameters(self, server: "FLServer", parameters: NDArrays) -> None:
        """
        Set model parameters received from clients aggregation.

        This method determines how to apply the aggregated parameters.

        Args:
            server: The federated learning server instance.
            parameters: Model parameters received from clients.
        """
        ...
