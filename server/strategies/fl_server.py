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

        # Initialize server parameters
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
        self.data_to_log = {}

        # CID Mapping: Bidirectional mapping between Flower UUIDs and numeric CIDs
        # uuid_to_cid: Maps ClientProxy.cid (UUID string) -> numeric CID (string)
        # cid_to_uuid: Maps numeric CID (string) -> ClientProxy.cid (UUID string)
        self.uuid_to_cid: Dict[str, str] = {}
        self.cid_to_uuid: Dict[str, str] = {}

        # Initialize strategies
        self.aggregate_method.init(self)
        self.client_selection.init(self)

    def _update_cid_mapping(self, proxy: ClientProxy, numeric_cid: str) -> None:
        """
        Update the bidirectional mapping between Flower UUIDs and numeric CIDs.

        This method is called whenever we receive metrics from a client containing
        their numeric CID. It ensures we can always translate between:
        - ClientProxy.cid (UUID) <-> numeric CID used in our data structures

        Args:
            proxy: ClientProxy object with UUID
            numeric_cid: Numeric CID from client metrics (e.g., "0", "1", ...)
        """
        uuid = str(proxy.cid)

        # Update both mappings
        self.uuid_to_cid[uuid] = numeric_cid
        self.cid_to_uuid[numeric_cid] = uuid

    def _get_numeric_cid(self, proxy: ClientProxy) -> Optional[str]:
        """
        Get the numeric CID for a ClientProxy.

        Args:
            proxy: ClientProxy object

        Returns:
            Numeric CID as string, or None if not yet mapped
        """
        return self.uuid_to_cid.get(str(proxy.cid))

    def _get_proxy_uuid(self, numeric_cid: str) -> Optional[str]:
        """
        Get the UUID for a numeric CID.

        Args:
            numeric_cid: Numeric CID as string

        Returns:
            UUID as string, or None if not yet mapped
        """
        return self.cid_to_uuid.get(numeric_cid)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        Args:
            server_round: Current server round.
            parameters: Model parameters to train.
            client_manager: ClientManager to sample clients from.

        Returns:
            A list of tuples containing ClientProxy and FitIns.
        """
        self.current_round = server_round

        # Step 1: Get ALL available clients from ClientManager
        all_available_clients = client_manager.sample(
            num_clients=self.conf.n_clients,
            min_num_clients=1,  # Accept at least 1 client (flexible)
        )

        # Step 2: Build list of available numeric CIDs
        # For first round, UUID mapping might not exist yet, so we attempt to use
        # existing mappings and fall back to all clients if no mapping exists
        available_numeric_cids = []
        for proxy in all_available_clients:
            numeric_cid = self._get_numeric_cid(proxy)
            if numeric_cid is not None:
                available_numeric_cids.append(numeric_cid)

        # If no mappings exist yet (first round), allow all numeric CIDs
        if not available_numeric_cids:
            available_numeric_cids = self.list_of_clients

        # Step 3: Use client selection strategy to choose from AVAILABLE numeric CIDs
        selected_numeric_cids = self.client_selection.select(
            server=self,
            server_round=server_round,
            list_of_clients=available_numeric_cids,
        )
        self.selected_clients = selected_numeric_cids

        # Step 4: Map selected numeric CIDs back to ClientProxy objects
        # Build a UUID->Proxy mapping for quick lookup
        uuid_to_proxy = {str(proxy.cid): proxy for proxy in all_available_clients}

        selected_proxies = []
        for numeric_cid in selected_numeric_cids:
            uuid = self._get_proxy_uuid(numeric_cid)
            if uuid and uuid in uuid_to_proxy:
                selected_proxies.append(uuid_to_proxy[uuid])

        # If no proxies found (first round), use all available proxies
        if not selected_proxies:
            selected_proxies = all_available_clients

        # Step 5: Create FitIns with configuration
        config = {
            "rounds": server_round,
            "selected_by_server": ",".join(selected_numeric_cids),
        }
        fit_ins = FitIns(parameters, config)

        return [(proxy, fit_ins) for proxy in selected_proxies]

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
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
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

        # Get all available clients
        all_available_clients = client_manager.sample(
            num_clients=self.conf.n_clients,
            min_num_clients=1,  # Accept at least 1 client
        )

        # Filter to get only the clients that were selected for training, and that are available
        # We evaluate the same clients that trained in this round
        # Build UUID->Proxy mapping for quick lookup
        uuid_to_proxy = {str(proxy.cid): proxy for proxy in all_available_clients}

        selected_proxies = []
        for numeric_cid in self.selected_clients:
            uuid = self._get_proxy_uuid(numeric_cid)
            if uuid and uuid in uuid_to_proxy:
                selected_proxies.append(uuid_to_proxy[uuid])

        # If no proxies found (first round before mappings are established),
        # use all available proxies
        if not selected_proxies:
            selected_proxies = all_available_clients

        return [(proxy, evaluate_ins) for proxy in selected_proxies]

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

        This method:
        1. Extracts numeric CID from client metrics
        2. Updates UUID <-> CID mapping for future rounds
        3. Stores client metrics in data structures indexed by numeric CID

        Args:
            results: List of (ClientProxy, EvaluateRes) tuples from client evaluations.
        """
        for proxy, eval_res in results:
            # Extract numeric CID from client metrics
            numeric_cid = str(int(eval_res.metrics["cid"]))

            # Update the bidirectional mapping
            self._update_cid_mapping(proxy, numeric_cid)

            # Extract metrics
            acc = eval_res.metrics["acc"]
            participating_state = eval_res.metrics["participating_state"]
            loss = eval_res.loss

            # Store in data structures indexed by numeric CID
            cid_idx = int(numeric_cid)
            self.clients_acc[cid_idx] = acc
            self.clients_loss[cid_idx] = loss
            self.client_participating_state[cid_idx] = participating_state

        self.clients_acc_avg: float = np.mean(self.clients_acc)
