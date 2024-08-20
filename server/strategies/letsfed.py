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
class LetsFed(fl.server.strategy.FedAvg):
    def __init__(
        self,
        config: Dict,
    ):
        self.n_clients = config['clients']
        self.rounds    = config['rounds']
        super().__init__(
            fraction_fit = 1,
            min_available_clients = self.n_clients,
            min_fit_clients = self.n_clients,
            min_evaluate_clients = self.n_clients
        )
        
        self.participating_clients      = [(cid, True) for cid in config['participating']]
        self.non_participating_clients  = [(cid, False) for cid in range(self.n_clients) if cid not in config['participating']]
        self.clients_intentions   = list(range(self.n_clients))
        self.solution             = config['strategy']
        self.init_clients         = config['init_clients']
        for client_info in self.participating_clients + self.non_participating_clients:
            self.clients_intentions[client_info[0]] = client_info[1]
        
        self.p_method         = config['p_method']
        self.np_method        = config['np_method']

        # Select
        self.selected_clients = []
        self.r_intetions      = np.full(self.n_clients, 0)
        
        # random select
        self.exploitation = config['exploitation']
        self.exploration  = config['exploration']
        self.decay        = config['decay']
        
        self.model_type      = config['model_type']
        self.epochs          = config['epochs']
        self.dirichlet_alpha = config['dirichlet_alpha']
        self.dataset         = config['dataset']
        self.threshold       = config['threshold']
        self.tid             = config['tid']
        # least select
        self.how_many_time_selected             = np.full(self.n_clients, 0)
        self.how_many_time_selected_non_participating = np.full(self.n_clients, 0)
        self._init_client_config()

        

        self.behaviors: Dict[str, Driver] = self.set_behaviors()

        self.forget_clients = np.full(self.n_clients, int(self.rounds*0.15))



    def set_behaviors(self):
        return {
            'selection_driver': SelectionDriver(self.np_method)
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
            'rounds': server_round,
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
            if metrics['participating_state']:
                if is_select_by_server(str(cid), [str(c[0]) for c in self.selected_clients]):
                    weights_of_participating_clients.append((
                        parameters_to_ndarrays(fit_res.parameters),
                        fit_res.num_examples
                    ))

        if len(weights_of_participating_clients) == 0:
            return self._last_fit
        
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_of_participating_clients)) ## FedAvg
        self._last_fit = parameters_aggregated
        return parameters_aggregated

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        config = {
            'rounds': server_round,
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
        cia_loss_aggregated = self._cia_aggregate_evaluate(results = results, server_round = server_round)

        return cia_loss_aggregated, {}

    def _collect_client_informations(self, results, server_round):
        clients_info = self._extract_client_info(results = results)
        clients_info.sort(key=lambda client: int(client['cid']))
        self._update_client_infos(
            clients_info = clients_info,
            server_round = server_round
        )
        self.participating_clients_acc_avg = np.mean(np.array([client_info[1] for client_info in self.participating_clients_acc]))
        self.non_participating_clients_acc_avg = np.mean(np.array([client_info[1] for client_info in self.non_participating_clients_acc]))
        
    def _extract_client_info(self, results):
        return  [
            {
                'cid': int(eval_res.metrics['cid']),
                'desired_state': eval_res.metrics['desired_state'],
                'loss': eval_res.loss,
                'acc': eval_res.metrics['acc'],
                'fit_acc': eval_res.metrics['fit_acc'],
                'participating_state': eval_res.metrics['participating_state'],
                'r_intention': eval_res.metrics['r_intention']
            }
            for _, eval_res in results
        ]

    def _cia_aggregate_evaluate(self, results, server_round):
        cia_parameters = []
        participating_clients          = []
        non_participating_clients      = []
        for _, evaluate_res in results:
            metrics = evaluate_res.metrics
            cid = str(metrics['cid'])
            if bool(metrics['participating_state']):
                participating_clients.append(cid)
                if is_select_by_server(str(cid), [str(c[0]) for c in self.selected_clients]):
                    cia_parameters.append((
                        evaluate_res.num_examples,
                        evaluate_res.loss,
                    ))
            else:
                non_participating_clients.append(cid)
        should_pass = len(cia_parameters) == 0
        
        my_logger.log(
            '/s-data.csv',
            data = {
                'rounds': server_round,
                'strategy': self.solution.lower(),
                'model_type': self.model_type.lower(),
                'np_method': self.np_method.lower(),
                'p_method': self.p_method.lower(),
                'n_selected': len(self.selected_clients),
                'n_participating_clients': len(participating_clients),
                'n_non_participating_clients': len(non_participating_clients),
                'selection': '|'.join([f"{str(client[0])}:{client[2]}" for client in self.participating_clients+self.non_participating_clients if client[0] in [c[0] for c in self.selected_clients]]),
                'r_intetion': '|'.join([str(x) for x in self.r_intetions.tolist()]),
                'r_robin': '|'.join([str(x) for x in self.how_many_time_selected.tolist()]),
                'skip_round': should_pass,
                'local_epochs': self.epochs,
                'dirichlet_alpha': self.dirichlet_alpha,
                'dataset': self.dataset.lower(),
                'exploitation': self.exploitation,
                'exploration': self.exploration,
                'threshold': self.threshold,
                'decay': self.decay,
                'init_clients': self.init_clients,
                'tid': self.tid,
                'forget_clients': "|".join(f'{str(client)}' for client in self.forget_clients),
            },
        )

        if should_pass:
            return self._last_eval

        loss_aggregated = weighted_loss_avg(cia_parameters)
        self._last_eval = loss_aggregated
        return loss_aggregated

    def _update_client_infos(self, clients_info, server_round):
        """ Sim poderia fazer em 1 for, mas não quero - SUPERA """
        self.clients_intentions      = list(range(self.n_clients))
        for client_info in clients_info:
            self.clients_intentions[int(client_info['cid'])] = client_info['desired_state']
        self.participating_clients         = [(client_info['cid'], client_info['desired_state'], client_info['participating_state']) for client_info in clients_info if client_info['desired_state']]
        self.participating_clients_acc     = [(client_info['cid'], client_info['fit_acc']) for client_info in clients_info if client_info['desired_state']]
        self.non_participating_clients     = [(client_info['cid'], client_info['desired_state'], client_info['participating_state']) for client_info in clients_info if not client_info['desired_state']]
        self.non_participating_clients_acc = [(client_info['cid'], client_info['fit_acc']) for client_info in clients_info if not client_info['desired_state']]
        self.r_intetions             = np.array([client_info['r_intention'] for client_info in clients_info])
        # atualizar
        for client_info in self.participating_clients:
            self.manager_client_rounds[int(client_info[0])] += 1
