from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from ...conf.structs import Environment
from ...utils.logger import Logger
from .aggregate_method.base import AggregateMethod
from .client_selection_method.base import ClientSelectionMethod

logger = Logger(__name__)


class FLServer(Strategy):
    """
    Base class for federated learning servers.

    All server implementations should inherit from this class.
    It provides common functionality and enforces implementation of
    key federated learning server methods.
    """

    def __init__(
        self,
        client_selection: ClientSelectionMethod,
        aggregate_method: AggregateMethod,
        conf: Environment,
    ) -> None:
        """
        Initialize federated server.

        Args:
            config: Environment configuration
            client_selection: Client selection strategy
            aggregate_method: Model aggregation strategy
        """
        super().__init__()

        self.client_selection: ClientSelectionMethod = client_selection
        self.aggregate_method: AggregateMethod = aggregate_method
        self.conf: Environment = conf

        self.list_of_clients: List[str] = [str(x) for x in range(conf.n_clients)]
        self.selected_clients: List[str] = []
        self.client_participating_state = np.ones(conf.n_clients, dtype=bool)
        self.current_round: int = 0
        self.clients_acc_avg: float = 0.0
        self.clients_loss_avg: float = 0.0
        self.clients_acc = np.zeros(conf.n_clients)
        self.clients_loss = np.zeros(conf.n_clients)
        self.model: keras.Model = None
        self.aggregate_method.init(self)
        self.data_to_log = {}

        # Logging
        self.metrics_history: Dict[str, list] = {
            "round": [],
            "avg_accuracy": [],
            "avg_loss": [],
        }

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """
        Initialize model parameters using an initialization function.

        Args:
            client_manager: ClientManager to sample clients from for initialization.
        """
        return None

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluate model parameters using an evaluation function.

        Args:
            server_round: Current server round.
            parameters: Model parameters to evaluate.
        """
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy | FitIns]]:
        """Configure the next round of training.

        Args:
            server_round: Current server round.
            parameters: Model parameters to train.
            client_manager: ClientManager to sample clients from.

        Returns:
            A list of tuples containing ClientProxy and FitIns.
        """
        self.current_round = server_round
        clients_cids = self.client_selection.select(
            server=self, server_round=server_round, list_of_clients=self.list_of_clients
        )
        self.selected_clients = clients_cids
        config = {
            "rounds": server_round,
            "selected_by_server": ",".join(clients_cids),
        }
        fit_ins = FitIns(parameters, config)

        clients = client_manager.sample(
            num_clients=self.conf.n_clients, min_num_clients=self.conf.n_clients
        )

        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy | FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate training results using an aggregation function.

        Args:
            server_round: Current server round.
            results: List of (ClientProxy, FitRes) tuples, where FitRes contains the
                results from the client training.
            failures: List of failures that occurred during client training.

        Returns:
            A tuple of (aggregated_parameters, aggregated_metrics) where
            aggregated_parameters is the aggregated model parameters and
            aggregated_metrics is a dictionary of aggregated metrics.
        """
        parameters, config = self.aggregate_method.agg_fit(
            server=self, server_round=server_round, results=results, failures=failures
        )
        return parameters, config

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy | EvaluateIns]]:
        """
        Configure the next round of evaluation.

        Args:
            server_round: Current server round.
            parameters: Model parameters to evaluate.
            client_manager: ClientManager to sample clients from.

        Returns:
            A list of tuples containing ClientProxy and EvaluateIns.
        """
        config = {
            "rounds": server_round,
            "selected_by_server": ",".join(self.selected_clients),
        }

        evaluate_ins = EvaluateIns(parameters=parameters, config=config)

        clients = client_manager.sample(
            num_clients=self.conf.n_clients, min_num_clients=self.conf.n_clients
        )
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results using an aggregation function.

        Args:
            server_round: Current server round.
            results: List of (ClientProxy, EvaluateRes) tuples, where EvaluateRes
                contains the results from the client evaluation.
            failures: List of failures that occurred during client evaluation.

        Returns:
            A tuple of (aggregated_loss, aggregated_metrics) where aggregated_loss is
            the aggregated loss across all clients and aggregated_metrics is a dictionary
            of aggregated metrics.
        """
        loss, config = self.aggregate_method.agg_eval(self, server_round, results, failures)
        self._collect_clients_data(results)
        logger.log(
            "/s-data.csv",
            data=self.get_log_data(server_round),
        )

        return loss, config

    def get_log_data(self, server_round: int) -> Dict[str, Union[int, float, str]]:
        """
        Get log data for the current server round.

        Args:
            server_round: Current server round.

        Returns:
            Dictionary of log data.
        """
        return {
            "rounds": server_round,
            "acc": self.clients_acc_avg,
            "loss": np.mean(self.clients_loss),
            "model_type": self.conf.model_type.lower(),
            "n_selected": len(self.selected_clients),
            "selection": f"[{';'.join(self.selected_clients)}]",
            "dataset": self.conf.db.dataset,
            "threshold": self.conf.client.threshold,
            "init_clients": self.conf.init_clients,
            "participating_state": f"[{';'.join([str(state) for state in self.client_participating_state])}]",  # noqa: E501
            "number_of_participating": np.count_nonzero(self.client_participating_state),
            "number_of_non_participating": np.count_nonzero(not self.client_participating_state),
            "training_method": self.conf.client.training_strategy,
            "aggregation_method": f"{self.conf.server.aggregation.method}-default",
            "selection_method": self.conf.server.selection.method,
            **self.data_to_log,
        }

    def _collect_clients_data(self, results: List[Tuple[ClientProxy, EvaluateRes]]) -> None:
        """
        Collect data from clients after evaluation.

        Args:
            results: List of (ClientProxy, EvaluateRes) tuples from client evaluations.
        """
        for _, client in results:
            cid = int(client.metrics["cid"])
            acc = (client.metrics["acc"],)
            participating_state = client.metrics["participating_state"]
            loss = client.loss
            self.clients_acc[cid] = acc[0]  ## Não consegui encontrar onde isso vira uma tupla...
            self.clients_loss[cid] = loss
            self.client_participating_state[cid] = participating_state

        self.clients_acc_avg: float = np.mean(self.clients_acc)
