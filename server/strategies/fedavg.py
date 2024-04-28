import flwr as fl
from flwr.common import FitIns, EvaluateIns, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate ,weighted_loss_avg
from flwr.common.logger import logger
from logging import INFO
import numpy as np

class FedAvg(fl.server.strategy.FedAvg):
    def __init__(self, n_clients, rounds):
        self.n_clients = n_clients
        self.rounds = rounds
        super().__init__(fraction_fit = 1, min_available_clients = n_clients, min_fit_clients = n_clients, min_evaluate_clients = n_clients)

    def configure_fit(self, server_round, parameters, client_manager):
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        config = {
            'round'          : server_round,
        }

        fit_ins = FitIns(parameters, config)
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        print(clients[0])
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(self, server_round, results, failures):
        weights_results = [
            (
                parameters_to_ndarrays(fit_res.parameters),
                fit_res.num_examples
            ) for _, fit_res in results
        ]

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        return parameters_aggregated, {}
    
    def configure_evaluate(self, server_round, parameters, client_manager: ClientManager):
        config = {
            'round': server_round,
        }
        evaluate_ins = EvaluateIns(parameters, config)

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [ (client, evaluate_ins) for client in clients ]
    
    def aggregae_evaluate(self, server_round, results, failures):
        loss_to_aggregated = [
            (
                evaluate_res.num_examples,
                evaluate_res.loss,
            ) for evaluate_res in results
        ]
        sum_acc = np.sum(
            np.array(
                [ eval_res.metrics['accuracy'] for eval_res in results]
            )
        )
        sum_examples = np.sum(
            np.array(
                [r.num_examples for _, r in results]
            )
        )
        avg_acc =  sum_acc / sum_examples

        loss_aggregated = weighted_loss_avg(loss_to_aggregated)
        with open('logs/s-fedavg-evaluate.csv', 'a') as filename:
            filename.write(f'{server_round},{avg_acc},{loss_aggregated}')
        return loss_aggregated, {}