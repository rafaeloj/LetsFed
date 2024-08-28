from utils.dataset_manager import DSManager
from .base import AggregateMethod
from typing import TYPE_CHECKING, Dict, Tuple, List, Optional, Union
from flwr.common import Parameters, FitRes, EvaluateRes, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, NDArrays
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.server.client_proxy import ClientProxy
from utils import is_select_by_server
from flwr_datasets.utils import divide_dataset
import numpy as np
from functools import reduce
from utils import ModelManager

if TYPE_CHECKING:
    from .. import FLServer

class MaxFL(AggregateMethod):
    def init(self, server: 'FLServer'):
        server.model = self.load_model(server)
    def load_model(self, server: 'FLServer'):
        dm = DSManager(n_clients=server.conf.n_clients, conf=server.conf.db)
        train, test = dm.load_locally(partition_id=int(0))
        keys = list(test.features.keys())
        train, validation = divide_dataset(dataset=train, division=[0.8, 0.2])
        self.x_train, self.y_train, self.x_validation, self.y_validation = train[keys[0]], train[keys[1]], validation[keys[0]], validation[keys[1]]
        self.x_test, self.y_test = test[keys[0]], test[keys[1]]
        mm = ModelManager(
            server.conf,
            input_shape = self.x_train.shape,
            path='app'
        )
        
        return mm.get_model()

    def agg_fit(
        self,
        server: 'FLServer',
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        weights_results = []
        qk_s: float = 0.0
        for _, fit_res in results:
            cid = fit_res.metrics['cid']
            qk = fit_res.metrics['qk']
            if fit_res.metrics['participating_state']:
                if is_select_by_server(cid, server.selected_clients):
                    weights_results.append((
                        parameters_to_ndarrays(fit_res.parameters),
                        qk 
                    ))
                    qk_s += qk
        server.data_to_log['qk_s'] = qk_s / len(server.selected_clients)
        weights_avg = self.aggregate(weights_results)
        new_weights = [
            w + (1*l) for w, l in zip(server.model.get_weights(), weights_avg)
        ]
        server.model.set_weights(new_weights)
        return ndarrays_to_parameters(new_weights), {}

    def aggregate(self, weights_to_aggregate: List[Tuple[NDArrays,float]]) -> NDArrays:
        s = np.sum(np.array([qk for _, qk in weights_to_aggregate]))
        weighted_weights = [
            [layer * qk for layer in layers] for layers, qk in weights_to_aggregate
        ]

        parameters = [
            reduce(np.add, w) / s
            for w in zip(*weighted_weights)
        ]
        return parameters


    def agg_eval(
        self,
        server: 'FLServer',
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
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