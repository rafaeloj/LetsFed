import flwr as fl
from flwr.common import EvaluateIns, FitIns, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate ,weighted_loss_avg
from utils.logger import my_logger

class FedPOC(fl.server.strategy.FedAvg):
    def __init__(
        self,
        n_clients:        int,
        rounds:           int,
        epoch: int,
        dirichlet_alpha: float,
        no_iid: bool,
        dataset: str,
        threshold: float,
        fraction_clients: float = 1.0,
        perc_of_clients:  float = 0.5,
    ):
        self.n_clients          = n_clients
        self.frac_clients       = fraction_clients
        self.rounds             = rounds
        self.list_of_clients    = []
        self.list_of_accuracies = []
        self.selected_clients   = []
        self.perc_of_clients    = perc_of_clients

        self.epoch = epoch
        self.dirichlet_alpha = dirichlet_alpha
        self.no_iid = no_iid
        self.dataset = dataset
        self.threshold = threshold

        super().__init__(fraction_fit = self.frac_clients, min_available_clients = n_clients, min_fit_clients = n_clients, min_evaluate_clients = n_clients)

    def __select_clients(self, server_round):
        if server_round <= 1:
            self.selected_clients = [str(x) for x in range(self.n_clients)]
            return
        self.selected_clients = []
        clients2select        = int(float(self.n_clients) * float(self.perc_of_clients))
        self.selected_clients = self.list_of_clients[:clients2select]

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        self.__select_clients(server_round)
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        self.clients_last_round = self.selected_clients

        config = {
			"selected_by_server" : ','.join(self.selected_clients),
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
        poc_parameters = self._poc_aggregated_fit(results = results, server_round=server_round)
        return poc_parameters, {}

    def _poc_aggregated_fit(self, results, server_round):
        weights_results = []
        for _, fit_res in results:
            client_id         = str(fit_res.metrics['cid'])
            if self._is_seleceted_by_server(client_id) and fit_res.metrics['dynamic_engagement']:
                weights_results.append((
                    parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples
                ))

        if len(weights_results) == 0:
            return self._last_fit
        
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results)) ## FedAvg
        self._last_fit = parameters_aggregated
        return parameters_aggregated


    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        if self.fraction_evaluate == 0.0:
            return []
        config = {
            'round': server_round,
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
        poc_loss_aggregated = self._poc_aggregate_evaluate(results = results, server_round=server_round)

        return poc_loss_aggregated, {}

    def _poc_aggregate_evaluate(self, results, server_round):
        local_list_clients      = []
        self.list_of_clients    = []
        accs                    = []
        loss_to_aggregate       = []
        c_engaged               = []
        c_not_engaged           = []

        for _, eval_res in results:
            client_id       = str(eval_res.metrics['cid'])
            client_accuracy = float(eval_res.metrics['acc'])
            accs.append(client_accuracy)
            local_list_clients.append((client_id, client_accuracy))
            if eval_res.metrics['dynamic_engagement']:
                c_engaged.append(client_id)
                if self._is_seleceted_by_server(client_id):
                    loss_to_aggregate.append((eval_res.num_examples, eval_res.loss))
            else:
                c_not_engaged.append(client_id)

        local_list_clients.sort(key=lambda x: x[1])

        self.list_of_clients    = [str(client[0]) for client in local_list_clients]

        # Update status
        self.clients_intentions      = list(range(self.n_clients))
        for cid in c_engaged + c_not_engaged:
            self.clients_intentions[int(cid)] = True if cid in c_engaged else False

        should_pass = len(loss_to_aggregate) == 0
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
                "POC",
                None,
                len(self.selected_clients),
                len(c_engaged),
                len(c_not_engaged),
                '|'.join([f"{str(client)}:{self.clients_intentions[int(client)]}" for client in self.selected_clients]),
                None,
                None,
                should_pass,
                self.epoch,
                self.dirichlet_alpha,
                self.no_iid,
                self.dataset,
                None,
                None,
                None,
                None,
                self.threshold,
            ]
        )
        if should_pass:
            return self._last_eval
    
        loss_aggregated = weighted_loss_avg(loss_to_aggregate)
        self._last_eval = loss_aggregated

        return loss_aggregated, {}
    
    def _is_seleceted_by_server(self, cid: str):
        return str(cid) in self.selected_clients