import flwr as fl
from flwr.common import EvaluateIns, FitIns, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate ,weighted_loss_avg
from flwr.common.logger import logger
from utils.select_by_server import is_select_by_server
from utils.logger import my_logger
from logging import INFO
from typing import Dict
from .drivers import Driver, SelectionDriver
import numpy as np
class FedCIA(fl.server.strategy.FedAvg):
    def __init__(
        self,
        n_clients: int,
        rounds:     int,
        engaged_clients: list,
        select_client_method: str,
        exploitation: float,
        exploration: float,
        least_select_factor: float,
        decay: float,
        solution: str,
        epoch: int,
        dirichlet_alpha: float,
        no_iid: bool,
        dataset: float,
        threshold: float,
    ):
        self.n_clients = n_clients
        self.rounds = rounds
        super().__init__(fraction_fit=1, min_available_clients=n_clients, min_fit_clients=n_clients, min_evaluate_clients=n_clients)
        self.engaged_clients      = [(cid, True) for cid in engaged_clients]
        self.not_engaged_clients  = [(cid, False) for cid in range(n_clients) if cid not in engaged_clients]
        self.clients_intentions      = list(range(self.n_clients))
        self.solution = solution
        for client_info in self.engaged_clients + self.not_engaged_clients:
            self.clients_intentions[client_info[0]] = client_info[1]
        self.select_client_method = select_client_method
        self.least_select_factor  = least_select_factor
        self.decay_factor         = decay
        # Select
        self.selected_clients = []
        self.r_intetions          = [0 for _ in range(n_clients)]
        # random select
        self.exploitation = exploitation ## Ta errado o termo mas não consegui pensar em outro nome
        self.exploration = exploration
        
        
        self.epoch = epoch
        self.dirichlet_alpha = dirichlet_alpha
        self.no_iid = no_iid
        self.dataset = dataset
        self.decay = decay
        self.threshold = threshold
        # least select
        self.how_many_time_selected = np.array([0 for _ in range(n_clients)])
        self.how_many_time_selected_not_engaged = np.array([0 for _ in range(n_clients)])
        self._init_client_config()
        
        self.behaviors: Dict[str, Driver]                         = self.set_behaviors()

    def set_behaviors(self):
        return {
            'selection_driver': SelectionDriver(self.select_client_method)
        }

    def _init_client_config(self):
        self.manager_client_rounds = [0 for i in range(self.n_clients)]


    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        config = {
            'round': server_round,
        }

        self.behaviors['selection_driver'].run(self, parameters = parameters, config = config)

        config['selected_by_server'] = ','.join([str(client[0]) for client in self.selected_clients])
        logger.log(INFO, f"SELECT CLIENTS: {config['selected_by_server']}")
        fit_ins = FitIns(parameters, config)

        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round: int, results, failures):
        cia_parameters = self._cia_aggregated_fit(results = results)
        return cia_parameters, {}

    def _cia_aggregated_fit(self, results):
        weights_of_participating_clients = []
        for _, fit_res in results:
            metrics = fit_res.metrics
            cid = fit_res.metrics['cid']
            if metrics['dynamic_engagement']:
                if is_select_by_server(str(cid), [str(c[0]) for c in self.selected_clients]):
                    weights_of_participating_clients.append((
                        parameters_to_ndarrays(fit_res.parameters),
                        fit_res.num_examples
                    ))

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_of_participating_clients)) ## FedAvg
        return parameters_aggregated

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        config = {
            'round': server_round,
            'selected_by_server': ','.join([str(client[0]) for client in self.selected_clients])
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
        self._collect_client_informations(results = results, server_round = server_round)
        cia_loss_aggregated = self._cia_aggregate_evaluate(results = results)
        my_logger.log(
            '/s-data.csv',
            header = [
                'round',
                'solution',
                'method',
                'n_selected',
                'n_engaged',
                'n_not_engaged',
                'selection',
                'r_intetion',
                'r_robin',
                'skip_round',
                'epoch',
                'dirichlet_alpha',
                'no_iid',
                'dataset',
                'exploitation',
                'exploration',
                'least_select_factor',
                'decay',
                'threshold',
            ],
            data = [
                server_round,
                self.solution,
                self.select_client_method,
                len(self.selected_clients),
                len(self.engaged_clients),
                len(self.not_engaged_clients),
                '|'.join([f"{str(client[0])}:{self.clients_intentions[client[0]]}" for client in self.selected_clients]),
                '|'.join([str(x) for x in self.r_intetions.tolist()]),
                '|'.join([str(x) for x in self.how_many_time_selected.tolist()]),
                False,
                self.epoch,
                self.dirichlet_alpha,
                self.no_iid,
                self.dataset,
                self.exploitation,
                self.exploration,
                self.least_select_factor,
                self.decay,
                self.threshold,
            ]

        )
        return cia_loss_aggregated, {}

    def _collect_client_informations(self, results, server_round):
        clients_info = self._extract_client_info(results = results)
        clients_info.sort(key=lambda client: int(client['cid']))
        self._update_client_infos(
            clients_info = clients_info,
            server_round = server_round
        )
        self.engaged_clients_acc_avg = np.mean(np.array([client_info[1] for client_info in self.engaged_clients_acc]))
        self.not_engaged_clients_acc_avg = np.mean(np.array([client_info[1] for client_info in self.not_engaged_clients_acc]))
        
    def _extract_client_info(self, results):
        return  [
            {
                'cid': int(eval_res.metrics['cid']),
                'want': eval_res.metrics['want'],
                'loss': eval_res.loss,
                'acc': eval_res.metrics['acc'],
                'dynamic_engagement': eval_res.metrics['dynamic_engagement'],
                'r_intention': eval_res.metrics['r_intention']
            }
            for _, eval_res in results
        ]

    def _cia_aggregate_evaluate(self, results):
        cia_parameters = []
        for _, evaluate_res in results:
            metrics = evaluate_res.metrics
            cid = str(metrics['cid'])
            if metrics['dynamic_engagement']:
                if is_select_by_server(str(cid), [str(c[0]) for c in self.selected_clients]):
                    cia_parameters.append((
                        evaluate_res.num_examples,
                        evaluate_res.loss,
                    ))
        loss_aggregated = weighted_loss_avg(cia_parameters)
        return loss_aggregated

    def _update_client_infos(self, clients_info, server_round):
        """ Sim poderia fazer em 1 for, mas não quero - SUPERA """
        self.clients_intentions      = list(range(self.n_clients))
        for client_info in clients_info:
            self.clients_intentions[int(client_info['cid'])] = client_info['want']
        self.engaged_clients         = [(client_info['cid'], client_info['want'], client_info['dynamic_engagement']) for client_info in clients_info if client_info['want']]
        self.engaged_clients_acc     = [(client_info['cid'], client_info['acc']) for client_info in clients_info if client_info['want']]
        self.not_engaged_clients     = [(client_info['cid'], client_info['want'], client_info['dynamic_engagement']) for client_info in clients_info if not client_info['want']]
        self.not_engaged_clients_acc = [(client_info['cid'], client_info['acc']) for client_info in clients_info if not client_info['want']]
        self.r_intetions             = np.array([client_info['r_intention'] for client_info in clients_info])
        # atualizar
        for client_info in self.engaged_clients:
            self.manager_client_rounds[int(client_info[0])] += 1
