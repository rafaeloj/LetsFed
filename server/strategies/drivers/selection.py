from .driver import Driver
import random
import math
import numpy as np
class SelectionDriver(Driver):
    def __init__(self, selection_method):
        self.selection = selection_method

    def run(self, server, parameters, config):
        self._select_clients(server=server, server_round=config['rounds'])

    def _select_clients(self, server, server_round):
        """Menos selecionados com exploração randomica ou utilizando o deev"""
        if server_round == 1:
            server.selected_clients = server.participating_clients + server.non_participating_clients
            return

        non_participating = []
        participating = []
        if server.p_method.lower() == 'random':
            participating = self._random_select(server=server, server_round=server_round)
        elif server.p_method.lower() == 'r_robin':
            participating = self._r_robin_participating(server=server)
        elif server.p_method.lower() == 'deev':
            participating = self._deev_participating(server=server, server_round=server_round)
        elif server.p_method.lower() == 'poc':
            participating = self._poc_participating(server=server)

        if server.np_method.lower() == 'deev':
            non_participating = self._deev_non_participating(server = server, server_round = server_round)
        elif server.np_method.lower() == 'random':
            non_participating = self._exploration_clients(server=server)
        elif server.np_method.lower() == 'r_robin':
            non_participating = self._r_robin_non_participating(server = server)
        elif server.np_method.lower() == 'deev-invert':
            non_participating = self._deev_invert_non_participating(server = server, server_round = server_round)

        # real_non_participating = [client for client in non_participating if server.forget_clients[client[0]] > 0]
        # for client in real_non_participating:
        #     if not client[1]: # se não interessado
        #         server.forget_clients[client[0]] -= 1
        # for client in participating:
        #     server.forget_clients[client[0]] = int(server.rounds*0.15)

        server.selected_clients = participating + non_participating
    
    def _r_robin_non_participating(self, server):
        non_participating_clients_cid = [int(c[0]) for c in server.non_participating_clients]
        # Pega a quantidade de clientes que querem participar dentro do contador
        how_many_time_selected_client_non_participating_index = server.how_many_time_selected[non_participating_clients_cid]

        # Aqui basicamente eu to pegando um array de indices ordenados pelos valorres do how_many_time_selected_client_non_participating_index
        sort_indexes = np.argsort(how_many_time_selected_client_non_participating_index)
        
        # Pegando o top menos chamados
        # top_values_of_cid  = sort_indexes[:int(len(how_many_time_selected_client_non_participating_index)*server.least_select_factor)]
        top_values_of_cid  = sort_indexes[:int(len(how_many_time_selected_client_non_participating_index)*server.exploitation)]

        # Update score
        for cid_value_index in top_values_of_cid:
            server.how_many_time_selected_non_participating[
                non_participating_clients_cid[
                    cid_value_index
                ]
            ] += 1
        
        # To pegand o indice dos clientes que foram selecionados
        top_clients = [(non_participating_clients_cid[cid], None) for cid in top_values_of_cid]
        return top_clients

    def _r_robin_participating(self,server):
        participating_clients_cid = [int(c[0]) for c in server.participating_clients]
        # Pega a quantidade de clientes que querem participar dentro do contador
        how_many_time_selected_client_participating_index = server.how_many_time_selected[participating_clients_cid]

        # Aqui basicamente eu to pegando um array de indices ordenados pelos valorres do how_many_time_selected_client_participating_index
        sort_indexes = np.argsort(how_many_time_selected_client_participating_index)
        
        # Pegando o top menos chamados
        # top_values_of_cid  = sort_indexes[:int(len(how_many_time_selected_client_participating_index)*server.least_select_factor)]
        top_values_of_cid  = sort_indexes[:int(len(how_many_time_selected_client_participating_index)*server.exploitation)]

        # Update score
        for cid_value_index in top_values_of_cid:
            server.how_many_time_selected[
                participating_clients_cid[
                    cid_value_index
                ]
            ] += 1
        
        # To pegand o indice dos clientes que foram selecionados
        top_clients = [(participating_clients_cid[cid], None) for cid in top_values_of_cid]
        return top_clients

    def _deev_participating(self, server, server_round):
        selected_clients = []
        for idx_accuracy in range(len(server.participating_clients_acc)):
            if server.participating_clients_acc[idx_accuracy][1] < server.participating_clients_acc_avg:
                selected_clients.append(server.participating_clients[idx_accuracy])

        if server.decay > 0.0:
            the_chosen_ones  = len(selected_clients) * (1 - server.decay)**int(server_round)
            selected_clients = selected_clients[ : math.ceil(the_chosen_ones)]

        return selected_clients

    def _poc_participating(self, server):
        selected_clients = []
        order_list = sorted(server.participating_clients_acc, key=lambda t: t[1])
        cids = [cid for cid, acc in order_list]
        clients2select        = int(len(cids) * server.exploitation)
        for cid, acc in order_list:
            selected_clients.append((cid, server.clients_intentions[cid]))
        # selected_clients = server.participating_clients[:clients2select]
        return selected_clients[:clients2select]

    def _deev_invert_non_participating(self, server, server_round):
        selected_clients = []
        for idx_accuracy in range(len(server.non_participating_clients_acc)):
            if server.non_participating_clients_acc[idx_accuracy][1] > server.non_participating_clients_acc_avg:
                selected_clients.append(server.non_participating_clients[idx_accuracy])

        if server.decay > 0.0:
            the_chosen_ones  = len(selected_clients) * (1 - server.decay)**int(server_round)
            selected_clients = selected_clients[ : math.ceil(the_chosen_ones)]

        # return selected_clients
        return selected_clients[:int(len(selected_clients)*server.exploitation)]

    def _exploration_clients(self, server):
        # Realiza o sorteio para enviar o modelo aos clientes que não querem participar
        if len(server.non_participating_clients) == 1:
            return server.non_participating_clients
        perc = int(len(server.non_participating_clients)*server.exploitation)
        explored_clients = random.sample(server.non_participating_clients, perc)
        return explored_clients

    def _deev_non_participating(self, server, server_round):
        selected_clients = []
        for idx_accuracy in range(len(server.non_participating_clients_acc)):
            if server.non_participating_clients_acc[idx_accuracy][1] < server.non_participating_clients_acc_avg:
                selected_clients.append(server.non_participating_clients[idx_accuracy])

        if server.decay > 0.0:
            the_chosen_ones  = len(selected_clients) * (1 - server.decay)**int(server_round)
            selected_clients = selected_clients[ : math.ceil(the_chosen_ones)]

        return selected_clients[:int(len(selected_clients)*server.exploitation)]
        # return selected_clients

    def _random_select(self, server, server_round):
        # Realiza o sorteio para enviar o modelo aos clientes que não querem participar
        if len(server.participating_clients) == 1:
            return server.participating_clients
        perc = int(len(server.participating_clients)*server.exploration)
        selected_clients = random.sample(server.participating_clients, perc)
        return selected_clients