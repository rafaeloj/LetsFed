from .base import AggregateMethod
from typing import TYPE_CHECKING, Dict, Tuple, List, Optional, Union
from flwr.common import Parameters, FitRes, EvaluateRes, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy.aggregate import aggregate ,weighted_loss_avg
from flwr.server.client_proxy import ClientProxy
from utils import is_select_by_server

if TYPE_CHECKING:
    from .. import FLServer

class FedAVG(AggregateMethod):
    def agg_fit(
        self,
        server: 'FLServer',
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}
        weights_results = []
        for _, fit_res in results:
            cid = fit_res.metrics['cid']
            if fit_res.metrics['participating_state']:
                if is_select_by_server(cid, server.selected_clients):
                    weights_results.append((
                        parameters_to_ndarrays(fit_res.parameters),
                        fit_res.num_examples    
                    ))

        if len(weights_results) == 0:
            return None, {}

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        return parameters_aggregated, {}
    def agg_eval(
        self,
        server: 'FLServer',
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ):
        if not results:
            return None, {}

        loss_to_aggregated        = []
        participating_clients     = []
        non_participating_clients = []
        for _, eval_res in results:
            client_id       = eval_res.metrics['cid']
            if eval_res.metrics['participating_state']:
                participating_clients.append(client_id)
                if is_select_by_server(client_id, server.selected_clients):
                    loss_to_aggregated.append((eval_res.loss, eval_res.num_examples))
            else:
                non_participating_clients.append(client_id)
        should_pass = len(loss_to_aggregated) <= 1
        if should_pass:
            return None, {}

        loss_aggregated = weighted_loss_avg(loss_to_aggregated)
        return loss_aggregated, {}

    def init(self, server: 'FLServer'):
        return super().init()