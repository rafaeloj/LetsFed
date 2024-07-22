import flwr as fl
from flwr.common import EvaluateIns, FitIns, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate ,weighted_loss_avg
import numpy as np
from utils import my_logger
import math

class FedDEEV(fl.server.strategy.FedAvg):
    def __init__(
        self,
        n_clients:        int,
        epoch: int,
        dirichlet_alpha: float,
        non_iid: bool,
        dataset: str,
        threshold: float,
        rounds:           int,
        fraction_clients: float,
        perc_of_clients:  float,
        decay:            float,
        model_type:       str,
        init_clients: float,
        config_test: str,
    ):
        self.n_clients          = n_clients
        self.frac_clients       = fraction_clients
        self.rounds             = rounds
        self.list_of_clients    = []
        self.list_of_accuracies = []
        self.selected_clients   = []
        self.init_clients = init_clients

        self.epoch = epoch
        self.dirichlet_alpha = dirichlet_alpha
        self.non_iid = non_iid
        self.dataset = dataset
        self.threshold = threshold
        self.model_type = model_type
        self.config_test = config_test
        ## DEEV
        self.perc_of_clients    = perc_of_clients
        self.decay_factor       = decay

        super().__init__(fraction_fit = self.frac_clients, min_available_clients = n_clients, min_fit_clients = n_clients, min_evaluate_clients = n_clients)

    def __select_clients(self, server_round):
        if server_round <= 1:
            self.selected_clients = [str(x) for x in range(self.n_clients)]
            return

        self.selected_clients = []
        for idx_accuracy in range(len(self.list_of_accuracies)):
            if self.list_of_accuracies[idx_accuracy] < self.average_accuracy:
                self.selected_clients.append(self.list_of_clients[idx_accuracy])

        if self.decay_factor > 0:
            the_chosen_ones  = len(self.selected_clients) * (1 - self.decay_factor)**int(server_round)
            self.selected_clients = self.selected_clients[ : math.ceil(the_chosen_ones)]

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        self.__select_clients(server_round=server_round)
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
        deev_parameters = self._deev_aggregated_fit(results = results, server_round=server_round)
        return deev_parameters, {}

    def _deev_aggregated_fit(self, results, server_round):
        weights_results = []
        for _, fit_res in results:
            client_id = str(fit_res.metrics['cid'])
            if self._is_seleceted_by_server(client_id) and fit_res.metrics['dynamic_engagement']:
                weights_results.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

        if len(weights_results) == 0:
            return self._last_fit

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results)) ## AVG
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
        deev_loss_aggregated = self._deev_aggregate_evaluate(results = results, server_round=server_round)
        return deev_loss_aggregated, {}

    def _deev_aggregate_evaluate(self, results, server_round):
        self.list_of_clients    = []
        local_list_clients      = []
        self.list_of_accuracies = []
        accs                    = []
        c_engaged               = []
        c_not_engaged           = []
        loss_to_aggregate       = []

        for _, eval_res in results:
            client_id       = str(eval_res.metrics['cid'])
            # client_accuracy = float(eval_res.metrics['fit_acc'])
            client_accuracy = float(eval_res.metrics['acc'])
            accs.append(client_accuracy)
            local_list_clients.append((client_id, client_accuracy))
            if bool(eval_res.metrics['dynamic_engagement']):
                c_engaged.append(client_id)
                if self._is_seleceted_by_server(client_id):
                    loss_to_aggregate.append((eval_res.num_examples, eval_res.loss))
            else:
                c_not_engaged.append(client_id)

        local_list_clients.sort(key=lambda x: x[1])
        accs.sort()

        self.list_of_clients    = [str(client[0]) for client in local_list_clients]
        self.list_of_accuracies = [float(client[1]) for client in local_list_clients]

        self.average_accuracy   = np.mean(accs)

        # Update status
        self.clients_intentions      = list(range(self.n_clients))
        for cid in c_engaged + c_not_engaged:
            self.clients_intentions[int(cid)] = True if cid in c_engaged else False
        
        should_pass = len(loss_to_aggregate) == 0
        my_logger.log(
            '/s-data.csv',
            data = {
                'rounds': server_round,
                'strategy': 'deev',
                'model_type': self.model_type,
                'select_client_method': None,
                'select_client_method_to_engaged': None,
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
                'exploration': None,
                'decay': self.decay_factor,
                'threshold': self.threshold,
                'init_clients': self.init_clients,
                'config_test': self.config_test,
                'forget_clients': None,
            }
        )
        if should_pass:
            return self._last_eval
        
        loss_aggregated = weighted_loss_avg(loss_to_aggregate)
        self._last_eval = loss_aggregated


        return loss_aggregated, {}

    def _is_seleceted_by_server(self, cid: str):
        return str(cid) in [x for x in self.selected_clients]