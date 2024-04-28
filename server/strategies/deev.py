import flwr as fl
from flwr.common import EvaluateIns, FitIns, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate ,weighted_loss_avg
import numpy as np
import math

class FedDEEV(fl.server.strategy.FedAvg):
    def __init__(
        self,
        n_clients:        int,
        rounds:           int,
        fraction_clients: float = 1.0,
        perc_of_clients:  float = 0.5,
        decay:            float = 0,
    ):
        self.n_clients          = n_clients
        self.frac_clients       = fraction_clients
        self.rounds             = rounds
        self.list_of_clients    = []
        self.list_of_accuracies = []
        self.selected_clients   = []
        
        ## DEEV
        self.perc_of_clients    = perc_of_clients
        self.decay_factor       = decay
        super().__init__(fraction_fit = self.frac_clients, min_available_clients = n_clients, min_fit_clients = n_clients, min_evaluate_clients = n_clients)

    def __select_clients(self, server_round):
        """
            VALIDAR
        """
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
			"selected_clients" : ','.join(self.selected_clients),
            'round'          : server_round,
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
            if int(server_round) <= 1:
                weights_results.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))
            else:
                if client_id in self.selected_clients:
                    weights_results.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results)) ## AVG
        return parameters_aggregated

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        if self.fraction_evaluate == 0.0:
            return []
        config = {
            'round': server_round,
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

        for response in results:
            client_id       = response[1].metrics['cid']
            client_accuracy = float(response[1].metrics['acc'])
            accs.append(client_accuracy)
            local_list_clients.append((client_id, client_accuracy))

        local_list_clients.sort(key=lambda x: x[1])
        accs.sort()

        self.list_of_clients    = [str(client[0]) for client in local_list_clients]
        self.list_of_accuracies = [float(client[1]) for client in local_list_clients]

        self.average_accuracy   = np.mean(accs)
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        return loss_aggregated, {}

