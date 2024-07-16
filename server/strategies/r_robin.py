import flwr as fl
from flwr.common import EvaluateIns, FitIns, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate ,weighted_loss_avg
from utils.logger import my_logger
from utils.select_by_server import is_select_by_server
import numpy as np

class FedRR(fl.server.strategy.FedAvg):
    def __init__(
        self,
        n_clients:        int,
        rounds:           int,
        epoch: int,
        dirichlet_alpha: float,
        non_iid: bool,
        dataset: str,
        threshold: float,
        model_type:       str,
        perc_of_clients:  float,
        init_clients: float,
        config_test: str,
    ):
        self.n_clients          = n_clients
        self.rounds             = rounds
        self.list_of_clients    = [cid for cid in range(n_clients)]
        self.list_of_accuracies = []
        self.selected_clients   = []
        self.perc_of_clients    = perc_of_clients
        self.init_clients = init_clients
        self.epoch = epoch
        self.dirichlet_alpha = dirichlet_alpha
        self.non_iid = non_iid
        self.dataset = dataset
        self.threshold = threshold
        self.model_type = model_type
        self.config_test = config_test
        self.how_many_time_selected = np.full(n_clients, 0)

        super().__init__(fraction_fit = 1, min_available_clients = n_clients, min_fit_clients = n_clients, min_evaluate_clients = n_clients)

    def __select_clients(self, server_round):
        if server_round <= 1:
            self.selected_clients = [str(x) for x in range(self.n_clients)]
            return
        # Aqui basicamente eu to pegando um array de indices ordenados pelos valorres do how_many_time_selected
        sort_indexes = np.argsort(self.how_many_time_selected)
        
        # Pegando o top menos chamados
        # top_values_of_cid  = sort_indexes[:int(len(how_many_time_selected_client_engaged_index)*self.least_select_factor)]
        top_values_of_cid  = sort_indexes[:int(len(self.how_many_time_selected)*self.perc_of_clients)]

        # Update score
        for cid_value_index in top_values_of_cid:
            self.how_many_time_selected[cid_value_index] += 1
        
        # To pegando o cid dos clientes que foram selecionados
        top_clients = [str(self.how_many_time_selected[cid_index]) for cid_index in top_values_of_cid]
        self.selected_clients = top_clients

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        self.__select_clients(server_round)
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        self.clients_last_round = self.selected_clients

        config = {
			"selected_by_server" : ','.join(self.selected_clients),
            'rounds'          : server_round,
        }

        fit_ins = FitIns(parameters, config)
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round: int, results, failures):
        rr_parameters = self._rr_aggregated_fit(results = results, server_round=server_round)
        return rr_parameters, {}

    def _rr_aggregated_fit(self, results, server_round):
        weights_results = []
        for _, fit_res in results:
            client_id         = str(fit_res.metrics['cid'])
            if is_select_by_server(client_id, ','.join(self.selected_clients)) and fit_res.metrics['dynamic_engagement']:
                weights_results.append((
                    parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples
                ))

        if len(weights_results) == 0:
            return self._last_fit
        
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results)) ## FedAvg
        self._last_fit = parameters_aggregated
        return parameters_aggregated

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        config = {
            'rounds': server_round,
			"selected_by_server" : ','.join(self.selected_clients),
        }

        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

		# Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
			client_manager.num_available()
		)
        clients = client_manager.sample(
			num_clients=sample_size, min_num_clients=min_num_clients
		)

        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(self, server_round: int, results, failures):
        rr_loss_aggregated = self._rr_aggregate_evaluate(results = results, server_round=server_round)

        return rr_loss_aggregated, {}

    def _rr_aggregate_evaluate(self, results, server_round):
        loss_to_aggregate       = []
        c_engaged               = []
        c_not_engaged           = []

        for _, eval_res in results:
            client_id       = str(eval_res.metrics['fit_acc'])
            if bool(eval_res.metrics['dynamic_engagement']):
                c_engaged.append(client_id)
                if is_select_by_server(client_id, ','.join(self.selected_clients)):
                    loss_to_aggregate.append((eval_res.num_examples, eval_res.loss))
            else:
                c_not_engaged.append(client_id)

        # Update status
        self.clients_intentions      = list(range(self.n_clients))
        for cid in c_engaged + c_not_engaged:
            self.clients_intentions[int(cid)] = True if cid in c_engaged else False

        should_pass = len(loss_to_aggregate) == 0
        my_logger.log(
            '/s-data.csv',
            data = {
                'rounds': server_round,
                'strategy': "r_robin",
                'model_type': self.model_type.lower(),
                'select_client_method': None,
                'select_client_method_to_engaged': None,
                'n_selected': len(self.selected_clients),
                'n_engaged': len(c_engaged),
                'n_not_engaged': len(c_not_engaged),
                'selection': '|'.join([f"{str(client)}:{self.clients_intentions[int(client)]}" for client in self.selected_clients]),
                'r_intetion': None,
                'r_robin': '|'.join([str(x) for x in self.how_many_time_selected.tolist()]),
                'skip_round': should_pass,
                'local_epochs': self.epoch,
                'dirichlet_alpha': self.dirichlet_alpha,
                'non_iid': self.non_iid,
                'dataset': self.dataset.lower(),
                'exploitation': None,
                'exploration': self.perc_of_clients,
                'decay': None,
                'threshold': self.threshold,
                'init_clients': self.init_clients,
                'config_test': self.config_test,
                'forget_clients': None,
            },
        )
        if should_pass:
            return self._last_eval
    
        loss_aggregated = weighted_loss_avg(loss_to_aggregate)
        self._last_eval = loss_aggregated

        return loss_aggregated, {}
