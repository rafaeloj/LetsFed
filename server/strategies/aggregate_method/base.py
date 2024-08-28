from abc import ABC, abstractmethod
from flwr.common import Scalar, Parameters, FitRes, EvaluateRes
from typing import Tuple, Dict, TYPE_CHECKING, Optional, List, Union
from flwr.server.client_proxy import ClientProxy

if TYPE_CHECKING:
    from server.strategies import FLServer

class AggregateMethod(ABC):
    @abstractmethod
    def agg_fit(
        self,
        server: 'FLServer',
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """ Method to call in aggregate fit step """

    @abstractmethod
    def agg_eval(
        self,
        server: 'FLServer',
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """ Method to call in aggregate evaluate step """

    @abstractmethod
    def init(server: 'FLServer'):
        """
            Method to initialize parameters of specific solution
        """