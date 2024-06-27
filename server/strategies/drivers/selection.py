from .driver import Driver
import random
import math
import numpy as np
class SelectionDriver(Driver):
    def __init__(self, selection_method):
        self.selection = selection_method

    def run(self, server, parameters, config):
        self._select_clients(server=server, server_round=config['round'])

    def _select_clients(self, server, server_round):
        """Menos selecionados com exploração randomica ou utilizando o deev"""
        if server_round == 1:
            server.current_selection = server.engaged_clients + server.not_engaged_clients
            return
        
        engaged_clients_cid = [int(c[0]) for c in server.engaged_clients]
        # Pega a quantidade de clientes que querem participar dentro do contador
        how_many_time_selected_client_engaged_index = server.how_many_time_selected[engaged_clients_cid]

        # Aqui basicamente eu to pegando um array de indices ordenados pelos valorres do how_many_time_selected_client_engaged_index
        sort_indexes = np.argsort(how_many_time_selected_client_engaged_index)
        
        # Pegando o top menos chamados
        top_values_of_cid  = sort_indexes[:int(len(how_many_time_selected_client_engaged_index)*server.least_select_factor)]

        # Update score
        for cid_value_index in top_values_of_cid:
            server.how_many_time_selected[
                engaged_clients_cid[
                    cid_value_index
                ]
            ] += 1
        # To pegand o indice dos clientes que foram selecionados
        top_clients = [(engaged_clients_cid[cid], None) for cid in top_values_of_cid]
        # return top_clients
        if server.select_client_method == 'default':
            client_to_send_model = top_clients + self._exploration_clients_dynamic(server=server,server_round=server_round)
        if server.select_client_method == 'default_1':
            client_to_send_model = top_clients + self._exploration_clients(server=server)


        # if not 1 in [info[0] for info in client_to_send_model]:
        #     client_to_send_model = client_to_send_model + [(1, True)]
        # if not 2 in [info[0] for info in client_to_send_model]:
        #     client_to_send_model = client_to_send_model + [(2, True)]


        if len(client_to_send_model) < server.n_clients *0.3:
            faltantes = (server.n_clients *0.3) - len(client_to_send_model)
            not_select = [
                i for i in range(server.n_clients) if i not in [int(info[0]) for info in client_to_send_model]
            ]
            clientes_faltantes = random.sample(not_select, faltantes)
            conjunto_faltante = [(c, True) for c in clientes_faltantes]
            client_to_send_model = client_to_send_model + conjunto_faltante

        server.current_selection = client_to_send_model

    def _exploration_clients(self, server):
        # Realiza o sorteio para enviar o modelo aos clientes que não querem participar
        if len(server.not_engaged_clients) == 1:
            return server.not_engaged_clients
        perc = int(len(server.not_engaged_clients)*server.exploration)
        explored_clients = random.sample(server.not_engaged_clients, perc)
        return explored_clients

    def _exploration_clients_dynamic(self, server, server_round):
        selected_clients = []
        for idx_accuracy in range(len(server.not_engaged_clients_acc)):
            if server.not_engaged_clients_acc[idx_accuracy][1] < server.not_engaged_clients_acc_avg:
                selected_clients.append(server.not_engaged_clients[idx_accuracy])

        if server.decay_factor > 0.0:
            the_chosen_ones  = len(selected_clients) * (1 - server.decay_factor)**int(server_round)
            selected_clients = selected_clients[ : math.ceil(the_chosen_ones)]

        return selected_clients

    def _random_select(self, server, server_round):
        if server_round == 1:
            return server.engaged_clients
        # Sorteia N clientes da lista de clientes interessados
        perc = int(len(server.engaged_clients)*0.5)
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
