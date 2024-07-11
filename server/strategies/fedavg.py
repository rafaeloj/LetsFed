import flwr as fl
from flwr.common import FitIns, EvaluateIns, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate ,weighted_loss_avg
from utils.logger import my_logger
from utils import is_select_by_server
import random

class FedAvg(fl.server.strategy.FedAvg):
    def __init__(self,
            n_clients,
            rounds,
            perc,
            epoch: int,
            dirichlet_alpha: float,
            non_iid: bool,
            dataset: str,
            threshold: float,
            model_type: str,
            init_clients: float,
            config_test: str,
        ):
        self.n_clients = n_clients
        self.rounds = rounds
        self.list_of_clients = [str(x) for x in range(n_clients)]
        self.selected_clients = []
        self.exploration = perc
        self.epoch = epoch
        self.dirichlet_alpha = dirichlet_alpha
        self.non_iid = non_iid
        self.dataset = dataset
        self.threshold = threshold
        self.model_type = model_type
        self.init_clients = init_clients
        self.config_test = config_test

        super().__init__(fraction_fit = 1, min_available_clients = n_clients, min_fit_clients = n_clients, min_evaluate_clients = n_clients)

    def _exploration_clients(self, server_round):
        if server_round == 1:
            return self.list_of_clients
        # Realiza o sorteio para enviar o modelo aos clientes que não querem participar
        perc = int(len(self.list_of_clients)*self.exploration)
        explored_clients = random.sample(self.list_of_clients, perc)
        return explored_clients

    def configure_fit(self, server_round, parameters, client_manager):
        self.selected_clients = self._exploration_clients(server_round)

        config = {
            'rounds'          : server_round,
			"selected_by_server" : ','.join(self.selected_clients),
        }

        fit_ins = FitIns(parameters, config)
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        weights_results = []
        for _, fit_res in results:
            cid = fit_res.metrics['cid']
            if fit_res.metrics['dynamic_engagement']: # Na teoria não participantes não responderam o chamado
                if is_select_by_server(str(cid), self.selected_clients):
                    weights_results.append((
                        parameters_to_ndarrays(fit_res.parameters),
                        fit_res.num_examples    
                    ))

        if len(weights_results) == 0:
            return self._last_fit, {}

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        self._last_fit = parameters_aggregated
        return parameters_aggregated, {}
            
    def configure_evaluate(self, server_round, parameters, client_manager: ClientManager):
        config = {
            'rounds': server_round,
			"selected_by_server" : ','.join(self.selected_clients),
        }
        evaluate_ins = EvaluateIns(parameters, config)

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [ (client, evaluate_ins) for client in clients ]
    
    def aggregate_evaluate(self, server_round, results, failures):
        loss_to_aggregated = []
        c_engaged          = []
        c_not_engaged      = []
        for _, eval_res in results:
            client_id       = eval_res.metrics['cid']
            if eval_res.metrics['dynamic_engagement']:
                c_engaged.append(client_id)
                if is_select_by_server(str(client_id), self.selected_clients):
                    loss_to_aggregated.append((eval_res.loss, eval_res.num_examples))
            else:
                c_not_engaged.append(client_id)
    
        should_pass = (len(loss_to_aggregated) == 0)
        
        # Update status
        self.clients_intentions      = list(range(self.n_clients))
        for cid in c_engaged + c_not_engaged:
            self.clients_intentions[int(cid)] = True if cid in c_engaged else False
        
        my_logger.log(
            '/s-data.csv',
            data = {
                'rounds': server_round,
                'strategy': 'avg',
                'model_type': self.model_type.lower(),
                'select_client_method': 'random',
                'select_client_method_to_engaged': 'random',
                'n_selected': len(self.selected_clients),
                'n_engaged': len(c_engaged),
                'n_not_engaged': len(c_not_engaged),
                'selection': '|'.join([f"{str(client)}:{self.clients_intentions[int(client)]}" for client in self.selected_clients]),
                'r_intetion': None,
                'r_robin': None,
                'skip_round': should_pass,
                'local_epochs': self.epoch,
                'dirichlet_alpha': self.dirichlet_alpha,
                'non_iid': self.non_iid,
                'dataset': self.dataset.lower(),
                'exploitation': None,
                'exploration': self.exploration,
                'decay': None,
                'threshold': self.threshold,
                'init_clients': self.init_clients,
                'config_test': self.config_test,
                'forget_clients': None,
            },
        )

        if should_pass:
            return self._last_eval, {}
        loss_aggregated = weighted_loss_avg(loss_to_aggregated)
        self._last_eval = loss_aggregated
        return loss_aggregated, {}