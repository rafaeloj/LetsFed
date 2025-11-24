from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union

from flwr.common import EvaluateRes, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

from .structs import AggregationMethodConfig

if TYPE_CHECKING:
    from ..fl_server import FLServer


class AggregationMethod(ABC):
    """
    Abstract base class for aggregation methods.
    """

    def __init__(self, config: AggregationMethodConfig) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            config: The aggregation method configuration.
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
    def agg_fit(
        self,
        server: "FLServer",
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Method to call in aggregate fit step

        Args:
            server: The federated learning server instance.
            server_round: The current round number.
            results: List of tuples of ClientProxy and FitRes from clients.
            failures: List of failures that occurred during client training.

        Returns:
            A tuple containing the aggregated Parameters and a dictionary of Scalar metrics.
        """
        ...

    @abstractmethod
    def agg_eval(
        self,
        server: "FLServer",
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """
        Method to call in aggregate evaluate step

        Args:
            server: The federated learning server instance.
            server_round: The current round number.
            results: List of tuples of ClientProxy and EvaluateRes from clients.
            failures: List of failures that occurred during client evaluation.

        Returns:
            A tuple containing the aggregated loss (float) and a dictionary of Scalar metrics.
        """
        ...
