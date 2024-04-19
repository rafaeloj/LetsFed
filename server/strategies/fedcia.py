from typing import Dict, List, Tuple
import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate ,weighted_loss_avg
from flwr.common.logger import logger
from logging import INFO
import numpy as np
import pickle
import base64

class FedCia(fl.server.strategy.FedAvg):
    def __init__(
        self,
        n_clients: int,
        rounds:     int,
    ):
        self.n_clients = n_clients
        self.rounds = rounds
        self.cia_parameters = []
        print(n_clients)
        super().__init__(fraction_fit=1, min_available_clients=n_clients, min_fit_clients=n_clients, min_evaluate_clients=n_clients)

    def __select_clients(self):
        self.selected_clients = [str(cid) for cid in range(self.n_clients)]
    
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        # self.__select_clients()
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        config = {
            'round'          : server_round,
            'cia_parameters' : self._encode_message(self.cia_parameters),
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
        fed_parameters = self._fed_aggregate_fit(results = results)
        cia_parameters = self._cia_aggregated_fit(results = results)
        
        self.cia_parameters = parameters_to_ndarrays(cia_parameters)
        return fed_parameters, {}

    def _fed_aggregate_fit(self, results):
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results)) ## FedAvg

        return parameters_aggregated

    def _cia_aggregated_fit(self, results):
        weights_of_participating_clients = []
        
        for _, fit_res in results:
            metrics = fit_res.metrics
            if metrics['dynamic_engagement']: # Only participate clients
                logger.log(INFO, f"CIA: Fitted Client {metrics['cid']}")
                cia_parameters = self._decode_message(metrics['cia_parameters'])
                weights_of_participating_clients.append((cia_parameters, fit_res.num_examples))
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_of_participating_clients)) ## FedAvg

        return parameters_aggregated

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        config = {
            'round': server_round,
            'cia_parameters': self._encode_message(self.cia_parameters),
        }

        evaluate_ins = EvaluateIns(parameters, config)
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        return [ (client, evaluate_ins) for client in clients ]

    def aggregate_evaluate(self, server_round: int, results, failures):
        fed_loss_aggregated = self._fed_aggregate_evaluate(results = results)
        cia_loss_aggregated = self._cia_aggregate_evaluate(results = results)

        return fed_loss_aggregated, {'cia_loss_aggregated': cia_loss_aggregated}
    
    def _fed_aggregate_evaluate(self, results):
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        return loss_aggregated

    def _cia_aggregate_evaluate(self, results):
        cia_parameters = []
        for _, evaluate_res in results:
            metrics = evaluate_res.metrics
            if metrics['dynamic_engagement']:
                logger.log(INFO, f"CIA: Client {metrics['cid']} evaluated")
                cia_parameters.append((evaluate_res.num_examples, metrics['cia_loss']))
        
        loss_aggregated = weighted_loss_avg(cia_parameters)
        return loss_aggregated

    def _decode_message(self, message):
        decoded_message = base64.b64decode(message)
        return pickle.loads(decoded_message)
    
    def _encode_message(self, message):
        cia_parameters_serialized = pickle.dumps(message)
        cia_parameters_bytes = base64.b64encode(cia_parameters_serialized)
        return cia_parameters_bytes
