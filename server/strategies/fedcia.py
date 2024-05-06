import flwr as fl
from flwr.common import EvaluateIns, FitIns, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate ,weighted_loss_avg
from flwr.common.logger import logger
from logging import INFO
import random
import numpy as np
import math
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
        log_foulder: str,
    ):
        self.n_clients = n_clients
        self.rounds = rounds
        super().__init__(fraction_fit=1, min_available_clients=n_clients, min_fit_clients=n_clients, min_evaluate_clients=n_clients)
        self.n_engaged_clients    = len(engaged_clients)
        self.engaged_clients      = [(cid, True) for cid in engaged_clients]
        self.not_engaged_clients  = [(cid, False) for cid in range(n_clients) if cid not in engaged_clients]
        self.select_client_method = select_client_method
        self.least_select_factor  = least_select_factor
        self.decay_factor         = decay
        self.log_foulder          = log_foulder
        # Select
        self.current_selection = []

        # random select
        self.exploitation = exploitation ## Ta errado o termo mas não consegui pensar em outro nome
        self.exploration = exploration

        # worst performance select
        self.clients_loss = [] # Only engaged clients

        # best performance select
        self.clients_acc = [] # Only engaged clients

        # least select
        self.how_many_time_selected = np.array([0 for _ in range(n_clients)])

    def _select_clients(self, server_round):
        if server_round == 1:
            return self.engaged_clients
        
        engaged_clients_cid = [int(c[0]) for c in self.engaged_clients]
        # Pega a quantidade de clientes que querem participar dentro do contador
        how_many_time_selected_client_engaged_index = self.how_many_time_selected[engaged_clients_cid]

        # Aqui basicamente eu to pegando um array de indices ordenados pelos valorres do how_many_time_selected_client_engaged_index
        sort_indexes = np.argsort(how_many_time_selected_client_engaged_index)
        
        # Pegando o top menos chamados
        top_values_of_cid  = sort_indexes[:int(len(how_many_time_selected_client_engaged_index)*self.least_select_factor)]

        # Update score
        for cid_value_index in top_values_of_cid:
            self.how_many_time_selected[
                engaged_clients_cid[
                    cid_value_index
                ]
            ] += 1
        # To pegand o indice dos clientes que foram selecionados
        top_clients = [(engaged_clients_cid[cid], None) for cid in top_values_of_cid]
        # return top_clients
        if self.select_client_method == 'default':
            client_to_send_model = top_clients + self._exploration_clients_dynamic(server_round=server_round)
        if self.select_client_method == 'default_1':
            client_to_send_model = top_clients + self._exploration_clients()

        return client_to_send_model

    def _random_select(self, server_round):
        if server_round == 1:
            return self.engaged_clients
        # Sorteia N clientes da lista de clientes interessados
        perc = int(len(self.engaged_clients)*0.5)
        select_clients = random.sample(self.engaged_clients, perc)
        
        client_to_send_model = select_clients + self._exploration_clients()
        return client_to_send_model

    def _exploration_clients_dynamic(self, server_round):
        selected_clients = []
        for idx_accuracy in range(len(self.not_engaged_clients_acc)):
            if self.not_engaged_clients_acc[idx_accuracy][1] < self.not_engaged_clients_acc_avg:
                selected_clients.append(self.not_engaged_clients[idx_accuracy])

        if self.decay_factor > 0.0:
            the_chosen_ones  = len(selected_clients) * (1 - self.decay_factor)**int(server_round)
            selected_clients = selected_clients[ : math.ceil(the_chosen_ones)]

        return selected_clients

    def _worst_performance(self, server_round):
        """
            Com conjunto só dos que querem participar?
            Deve ter a exploração junto?
            Atualmente: feito somente com os engaged
        """
        if server_round == 1:
            return self.engaged_clients
        clients2select = int(float(len(self.engaged_clients)) * 0.5)
        select_clients = self.clients_loss[:clients2select]
        return select_clients + self._exploration_clients()
    def _best_performance(self, server_round):
        """
            Com conjunto só dos que querem participar?
            Deve ter a exploração junto?

            Atualmente: Feito somente com os engaged
        """
        if server_round == 1:
            return self.engaged_clients
        clients2select = int(float(len(self.engaged_clients)) * 0.5)
        select_clients = self.clients_acc[:clients2select]

        client_to_send_model = select_clients + self._exploration_clients()
        return client_to_send_model
    
    def _least_selected(self, server_round):
        if server_round == 1:
            return self.engaged_clients
        
        engaged_clients_cid = [c[0] for c in self.engaged_clients]
        
        # Pega a quantidade de clientes que querem participar dentro do contador
        how_many_time_selected_client_engaged_index = self.how_many_time_selected[engaged_clients_cid]

        # Aqui basicamente eu to pegando um array de indices ordenados pelos valorres do how_many_time_selected_client_engaged_index
        sort_indexes = np.argsort(how_many_time_selected_client_engaged_index)
        
        # Pegando o top menos chamados
        top_values_of_cid  = sort_indexes[:int(len(how_many_time_selected_client_engaged_index)*self.least_select_factor)]

        # Update score
        for cid_value_index in top_values_of_cid:
            self.how_many_time_selected[
                engaged_clients_cid[
                    cid_value_index
                ]
            ] += 1
        # To pegand o indice dos clientes que foram selecionados
        top_clients = [(engaged_clients_cid[cid], None) for cid in top_values_of_cid]
        # return top_clients
    
        client_to_send_model = top_clients + self._exploration_clients()
        return client_to_send_model

    def _deev_select(self, server_round):
        if server_round == 1:
            return self.engaged_clients

        selected_clients = []
        for idx_accuracy in range(len(self.clients_acc)):
            if self.clients_acc[idx_accuracy][1] < self.engaged_clients_acc_avg:
                selected_clients.append(self.engaged_clients[idx_accuracy])

        if self.decay_factor > 0.0:
            the_chosen_ones  = len(selected_clients) * (1 - self.decay_factor)**int(server_round)
            new_selected_clients = selected_clients[ : math.ceil(the_chosen_ones)]

        return new_selected_clients + self._exploration_clients()

    def _exploration_clients(self):
        # Realiza o sorteio para enviar o modelo aos clientes que não querem participar
        perc = int(len(self.not_engaged_clients)*self.exploration)
        explored_clients = random.sample(self.not_engaged_clients, perc)
        return explored_clients

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        self.current_selection = self._select_clients(server_round)

        with open(f'logs{self.log_foulder}/s-clients-selected.csv', 'a') as filename:
            filename.write(f"{';'.join([str(client[0]) for client in self.current_selection])},{server_round}\n")

        config = {
            'round' : server_round,
            'selected_by_server': ','.join([str(client[0]) for client in self.current_selection])
        }
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
                if self._is_seleceted_by_server(cid):
                    weights_of_participating_clients.append((
                        parameters_to_ndarrays(fit_res.parameters),
                        fit_res.num_examples
                    ))

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_of_participating_clients)) ## FedAvg
        return parameters_aggregated

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        config = {
            'round': server_round,
            'selected_by_server': ','.join([str(client[0]) for client in self.current_selection])
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
        self.clients_loss = []
        self.clients_acc  = []

        self.engaged_clients = []
        self.engaged_clients_acc = []
        self.not_engaged_clients = []
        self.not_engaged_clients_acc = []

        loss_to_aggregate = []
        clients_info = []

        c_engaged               = []
        c_not_engaged           = []
        for _, eval_res in results:
            self.clients_loss.append((int(eval_res.metrics['cid']), eval_res.loss))
            self.clients_acc.append((int(eval_res.metrics['cid']), eval_res.metrics['acc']))
 
            if eval_res.metrics['dynamic_engagement']:
                c_engaged.append(eval_res.metrics['cid'])
                self.engaged_clients.append((eval_res.metrics['cid'], eval_res.metrics['want']))
                self.engaged_clients_acc.append((eval_res.metrics['cid'], eval_res.metrics['acc']))
                if self._is_seleceted_by_server(str(eval_res.metrics['cid'])):
                    loss_to_aggregate.append((eval_res.num_examples, eval_res.loss))
            else:
                self.not_engaged_clients.append((eval_res.metrics['cid'], eval_res.metrics['want']))
                self.not_engaged_clients_acc.append((eval_res.metrics['cid'], eval_res.metrics['acc']))
                c_not_engaged.append(eval_res.metrics['cid'])
            clients_info.append(
                (int(eval_res.metrics['cid']), eval_res.metrics['want'], eval_res.loss, eval_res.metrics['acc'])
            )

        with open(f'logs{self.log_foulder}/s-infos.csv', 'a') as filename:
            # 'round', 'selected', 'engaged', 'not_engaged'
            filename.write(f"{server_round},{len(self.current_selection)},{len(self.engaged_clients)},{len(self.not_engaged_clients)}\n")

        clients_info.sort(key=lambda x: x[0])
        
        # self._update_status_of_clients(clients_info)
        self.engaged_clients_acc_avg = np.mean(np.array([client_info[1] for client_info in self.engaged_clients_acc]))
        self.not_engaged_clients_acc_avg = np.mean(np.array([client_info[1] for client_info in self.not_engaged_clients_acc]))
        
        cia_loss_aggregated = self._cia_aggregate_evaluate(results = results)
        
        # if self.select_client_method == 'least_selected':
        self._save_counter(server_round)

        return cia_loss_aggregated, {}

    def _cia_aggregate_evaluate(self, results):
        cia_parameters = []
        for _, evaluate_res in results:
            metrics = evaluate_res.metrics
            cid = str(metrics['cid'])
            if metrics['dynamic_engagement']:
                if self._is_seleceted_by_server(cid):
                    cia_parameters.append((
                        evaluate_res.num_examples,
                        evaluate_res.loss,
                    ))
        loss_aggregated = weighted_loss_avg(cia_parameters)
        return loss_aggregated

    def _save_counter(self, server_round):
        with open(f'logs{self.log_foulder}/s-infos-counter.csv', 'a') as filename:
            # 'round', 'selected', 'engaged', 'not_engaged'
            filename.write(f"{server_round},{';'.join([str(x) for x in self.how_many_time_selected.tolist()])}\n")

    def _is_seleceted_by_server(self, cid: str):
        return str(cid) in [str(c[0]) for c in self.current_selection]