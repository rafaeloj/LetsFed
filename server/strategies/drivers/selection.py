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
            server.selected_clients = server.engaged_clients + server.not_engaged_clients
            return
        
        not_engaged = []
        engaged = self._poc_engaged(server=server)

        if server.select_client_method == 'default':
            not_engaged = self._deev_not_engaged(server = server, server_round = server_round)
        if server.select_client_method == 'default_1':
            not_engaged = self._exploration_clients(server=server)
        if server.select_client_method == 'r_robin':
            not_engaged = self._r_robin_not_engaged(server = server)

        server.selected_clients = engaged + not_engaged
    
    def _r_robin_not_engaged(self, server):
        not_engaged_clients_cid = [int(c[0]) for c in server.not_engaged_clients]
        # Pega a quantidade de clientes que querem participar dentro do contador
        how_many_time_selected_client_not_engaged_index = server.how_many_time_selected[not_engaged_clients_cid]

        # Aqui basicamente eu to pegando um array de indices ordenados pelos valorres do how_many_time_selected_client_not_engaged_index
        sort_indexes = np.argsort(how_many_time_selected_client_not_engaged_index)
        
        # Pegando o top menos chamados
        # top_values_of_cid  = sort_indexes[:int(len(how_many_time_selected_client_not_engaged_index)*server.least_select_factor)]
        top_values_of_cid  = sort_indexes[:int(len(how_many_time_selected_client_not_engaged_index)*server.exploration)]

        # Update score
        for cid_value_index in top_values_of_cid:
            server.how_many_time_selected_not_engaged[
                not_engaged_clients_cid[
                    cid_value_index
                ]
            ] += 1
        
        # To pegand o indice dos clientes que foram selecionados
        top_clients = [(not_engaged_clients_cid[cid], None) for cid in top_values_of_cid]
        return top_clients

    def _r_robin_engaged(self,server):
        engaged_clients_cid = [int(c[0]) for c in server.engaged_clients]
        # Pega a quantidade de clientes que querem participar dentro do contador
        how_many_time_selected_client_engaged_index = server.how_many_time_selected[engaged_clients_cid]

        # Aqui basicamente eu to pegando um array de indices ordenados pelos valorres do how_many_time_selected_client_engaged_index
        sort_indexes = np.argsort(how_many_time_selected_client_engaged_index)
        
        # Pegando o top menos chamados
        # top_values_of_cid  = sort_indexes[:int(len(how_many_time_selected_client_engaged_index)*server.least_select_factor)]
        top_values_of_cid  = sort_indexes[:int(len(how_many_time_selected_client_engaged_index)*0.30)]

        # Update score
        for cid_value_index in top_values_of_cid:
            server.how_many_time_selected[
                engaged_clients_cid[
                    cid_value_index
                ]
            ] += 1
        
        # To pegand o indice dos clientes que foram selecionados
        top_clients = [(engaged_clients_cid[cid], None) for cid in top_values_of_cid]
        return top_clients

    def _deev_engaged(self, server, server_round):
        selected_clients = []
        for idx_accuracy in range(len(server.engaged_clients_acc)):
            if server.engaged_clients_acc[idx_accuracy][1] < server.engaged_clients_acc_avg:
                selected_clients.append(server.engaged_clients[idx_accuracy])

        if server.decay_factor > 0.0:
            the_chosen_ones  = len(selected_clients) * (1 - server.decay_factor)**int(server_round)
            selected_clients = selected_clients[ : math.ceil(the_chosen_ones)]

        return selected_clients

    def _poc_engaged(self, server):
        selected_clients = []
        order_list = sorted(server.engaged_clients_acc, key=lambda t: t[1])
        cids = [cid for cid, acc in order_list]
        clients2select        = int(len(cids) * 0.33)
        for cid, acc in order_list:
            selected_clients.append((cid, server.clients_intentions[cid]))
        # selected_clients = server.engaged_clients[:clients2select]
        return selected_clients[:clients2select]

    def _deev_invert_not_engaged(self, server, server_round):
        selected_clients = []
        for idx_accuracy in range(len(server.not_engaged_clients_acc)):
            if server.not_engaged_clients_acc[idx_accuracy][1] > server.not_engaged_clients_acc_avg:
                selected_clients.append(server.not_engaged_clients[idx_accuracy])

        if server.decay_factor > 0.0:
            the_chosen_ones  = len(selected_clients) * (1 - server.decay_factor)**int(server_round)
            selected_clients = selected_clients[ : math.ceil(the_chosen_ones)]

        return selected_clients

    def _exploration_clients(self, server):
        # Realiza o sorteio para enviar o modelo aos clientes que não querem participar
        if len(server.not_engaged_clients) == 1:
            return server.not_engaged_clients
        perc = int(len(server.not_engaged_clients)*server.exploration)
        explored_clients = random.sample(server.not_engaged_clients, perc)
        return explored_clients

    def _deev_not_engaged(self, server, server_round):
        selected_clients = []
        for idx_accuracy in range(len(server.not_engaged_clients_acc)):
            if server.not_engaged_clients_acc[idx_accuracy][1] < server.not_engaged_clients_acc_avg:
                selected_clients.append(server.not_engaged_clients[idx_accuracy])

        if server.decay_factor > 0.0:
            the_chosen_ones  = len(selected_clients) * (1 - server.decay_factor)**int(server_round)
            selected_clients = selected_clients[ : math.ceil(the_chosen_ones)]

        # return selected_clients[:int(len(selected_clients)*server.exploration)]
        return selected_clients

    def _random_select(self, server, server_round):
        if server_round == 1:
            return server.engaged_clients
        # Sorteia N clientes da lista de clientes interessados
        perc = int(len(server.engaged_clients)*server.exploration)
        select_clients = random.sample(server.engaged_clients, perc)
        
        client_to_send_model = select_clients + server._exploration_clients()
        return client_to_send_model
    
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
