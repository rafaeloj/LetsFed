import flwr as fl
from flwr.common import FitIns, EvaluateIns, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate ,weighted_loss_avg
from utils.logger import my_logger
from utils import is_select_by_server
import random

class FedAvg(fl.server.strategy.FedAvg):
    def __init__(self, config):
        self.n_clients        = config['clients']
        self.rounds           = config['rounds']
        self.list_of_clients  = [str(x) for x in range(self.n_clients)]
        self.selected_clients = []
        self.exploration      = config['exploration']
        self.epochs            = config['epochs']
        self.dirichlet_alpha  = config['dirichlet_alpha']
        self.dataset          = config['dataset']
        self.threshold        = config['threshold']
        self.model_type       = config['model_type']
        self.init_clients     = config['init_clients']
        self.tid              = config['tid']

        super().__init__(
            fraction_fit          = 1,
            min_available_clients = self.n_clients,
            min_fit_clients       = self.n_clients,
            min_evaluate_clients  = self.n_clients
        )

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
            if bool(fit_res.metrics['participating_state']):
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
        participating_clients          = []
        non_participating_clients      = []
        for _, eval_res in results:
            client_id       = eval_res.metrics['cid']
            if eval_res.metrics['participating_state']:
                participating_clients.append(client_id)
                if is_select_by_server(str(client_id), self.selected_clients):
                    loss_to_aggregated.append((eval_res.loss, eval_res.num_examples))
            else:
                non_participating_clients.append(client_id)
    
        should_pass = (len(loss_to_aggregated) == 0)
        
        # Update status
        self.clients_intentions      = list(range(self.n_clients))
        for cid in participating_clients + non_participating_clients:
            self.clients_intentions[int(cid)] = True if cid in participating_clients else False
        
        my_logger.log(
            '/s-data.csv',
            data = {
                'rounds': server_round,
                'strategy': 'avg',
                'model_type': self.model_type.lower(),
                'select_client_method': 'random',
                'select_client_method_to_engaged': 'random',
                'n_selected': len(self.selected_clients),
                'n_participating_clients': len(participating_clients),
                'n_non_participating_clients': len(non_participating_clients),
                'selection': '|'.join([f"{str(client)}:{self.clients_intentions[int(client)]}" for client in self.selected_clients]),
                'r_intetion': None,
                'r_robin': None,
                'skip_round': should_pass,
                'local_epochs': self.epochs,
                'dirichlet_alpha': self.dirichlet_alpha,
                'dataset': self.dataset.lower(),
                'exploitation': None,
                'exploration': self.exploration,
                'decay': None,
                'threshold': self.threshold,
                'init_clients': self.init_clients,
                'tid': self.tid,
                'forget_clients': None,
            },
        )

        if should_pass:
            return self._last_eval, {}
        loss_aggregated = weighted_loss_avg(loss_to_aggregated)
        self._last_eval = loss_aggregated
        return loss_aggregated, {}