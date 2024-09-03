from typing import Dict, List, Optional, Tuple, Union
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from .client_selection_method import ClientSelectionMethod
from .aggregate_method import AggregateMethod
from conf import Environment
from utils import my_logger
import keras
import numpy as np
class FLServer(Strategy):
    def __init__(
        self,
        client_selection: ClientSelectionMethod,
        aggregate_method: AggregateMethod,
        conf            : Environment,
    ):
        self.client_selection   : ClientSelectionMethod = client_selection 
        self.aggregate_method   : AggregateMethod = aggregate_method
        self.conf               : Environment = conf
        self.list_of_clients    : List[str] = [str(x) for x in range(conf.n_clients)]
        self.selected_clients   : List[str] = []
        self.current_round      : int = 0
        self.clients_acc_avg    : float = 0.0
        self.clients_acc = np.zeros(conf.n_clients)
        self.client_participating_state = np.ones(conf.n_clients, dtype=bool)
        self.clients_loss = np.zeros(conf.n_clients)
        self.model: keras.Model = None
        self.aggregate_method.init(self)
        self.data_to_log = {}

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return None

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        return None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy | FitIns]]:
        self.current_round = server_round
        clients_cids = self.client_selection.select(
            server = self,
            server_round = server_round,
            list_of_clients = self.list_of_clients
        )
        self.selected_clients = clients_cids
        config = {
            'rounds': server_round,
            'selected_by_server': ','.join(clients_cids),
        }
        fit_ins = FitIns(parameters, config)

        clients = client_manager.sample(
            num_clients=self.conf.n_clients, min_num_clients=self.conf.n_clients
        )

        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy | FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str,Scalar]]:
        parameters, config = self.aggregate_method.agg_fit(
            server = self, 
            server_round = server_round,
            results = results,
            failures = failures)
        return parameters, config

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy | EvaluateIns]]:
        config = {
            'rounds': server_round,
            'selected_by_server': ','.join(self.selected_clients),
        }

        evaluate_ins = EvaluateIns(parameters = parameters, config = config)

        clients = client_manager.sample(
            num_clients=self.conf.n_clients, min_num_clients=self.conf.n_clients
        )
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        loss, config = self.aggregate_method.agg_eval(self, server_round, results, failures)
        self.__collect_clients_data(results)
        my_logger.log(
            '/s-data.csv',
            data = self.get_log_data(server_round),
        )
        return loss, config
    
    def get_log_data(self, server_round):
        return {
            'rounds': server_round,
            'acc': self.clients_acc_avg,
            'loss': np.mean(self.clients_loss),
            'model_type': self.conf.model_type.lower(),
            'n_selected': len(self.selected_clients),
            'selection': f"[{';'.join(self.selected_clients)}]",
            'dataset': self.conf.db.dataset,
            'threshold': self.conf.client.threshold,
            'init_clients': self.conf.init_clients,
            'participating_state':  f"[{';'.join([str(state) for state in self.client_participating_state])}]",
            'number_of_participating': np.count_nonzero(self.client_participating_state == True),
            'number_of_non_participating': np.count_nonzero(self.client_participating_state == False),
            'training_method': self.conf.client.training_strategy,
            'aggregation': f"{self.conf.server.aggregation.method}-default",
            'selection': self.conf.server.selection.method,
            **self.data_to_log,
        }

    def __collect_clients_data(self, results: List[Tuple[ClientProxy, EvaluateRes]]):
        for _, client in results:
            cid = int(client.metrics['cid'])
            acc = client.metrics['acc'],
            participating_state = client.metrics['participating_state']
            loss = client.loss
            self.clients_acc[cid]  = acc[0]  ## Não consegui encontrar onde isso vira uma tupla...
            self.clients_loss[cid] = loss
            self.client_participating_state[cid] = participating_state
        self.clients_acc_avg: float = np.mean(self.clients_acc)

