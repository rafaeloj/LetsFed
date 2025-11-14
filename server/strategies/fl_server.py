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
from ...dataset_manager.dataset_manager import DSManager
from ...model.model_manager import ModelManager
from ...utils.logger import Logger
from .aggregate_method.base import AggregationMethod
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
        aggregate_method: AggregationMethod,
        conf: Environment,
    ) -> None:
        """
        Initialize federated server.

        Args:
            config: Environment configuration
            client_selection: Client selection strategy
            aggregate_method: Model aggregation strategy
        """
        logger.info("Initializing FL Server")
        logger.info(f"Configuration: {conf.n_clients} clients, {conf.rounds} rounds")
        logger.info(f"Aggregation: {conf.server.aggregation_method.name}")
        logger.info(f"Selection: {conf.server.selection_method.name}")
        logger.info(f"Training strategy: {conf.client.training_strategy.name}")

        super().__init__()

        # Initialize server parameters
        self.client_selection: ClientSelectionMethod = client_selection
        self.aggregate_method: AggregationMethod = aggregate_method
        self.conf: Environment = conf

        self.list_of_clients: List[str] = [str(x) for x in range(conf.n_clients)]
        self.selected_clients: List[str] = []
        self.client_participating_state = np.ones(conf.n_clients, dtype=bool)
        self.current_round: int = 0
        self.clients_acc_avg: float = 0.0
        self.clients_loss_avg: float = 0.0
        self.clients_acc = np.zeros(conf.n_clients)
        self.clients_loss = np.zeros(conf.n_clients)
        self.data_to_log = {}

        # CID Mapping: Bidirectional mapping between Flower UUIDs and numeric CIDs
        # uuid_to_cid: Maps ClientProxy.cid (UUID string) -> numeric CID (string)
        # cid_to_uuid: Maps numeric CID (string) -> ClientProxy.cid (UUID string)
        self.uuid_to_cid: Dict[str, str] = {}
        self.cid_to_uuid: Dict[str, str] = {}

        # Load data and model
        if conf.server.aggregation_method.name.lower() == "maxfl":
            logger.info("Loading server-side data and model for MaxFL aggregation")
            self.model: keras.Model
            self.x_train, self.y_train = None, None
            self.x_validation, self.y_validation = None, None
            self.x_test, self.y_test = None, None
            self._load_data()
            self._load_model()

        logger.info("FL Server initialization completed")

    def _load_model(self) -> None:
        """
        Load the model.
        """
        logger.debug(f"Loading server model (type: {self.conf.model.type})")
        mm = ModelManager(conf=self.conf, input_shape=self.x_train.shape)
        self.model = mm.get_model()
        logger.info("Server model loaded successfully")

    def _load_data(self) -> None:
        """
        Load the data.
        """
        logger.debug("Loading server dataset partition (partition 0)")
        dm = DSManager(n_clients=self.conf.n_clients, conf=self.conf.dataset)

        train, validation, test = dm.load_locally(partition_id=0)
        keys = list(test.features.keys())

        # Get label names
        self.labels = test.features["label"].names

        self.x_train, self.y_train = train[keys[0]], train[keys[1]]
        self.x_validation, self.y_validation = validation[keys[0]], validation[keys[1]]
        self.x_test, self.y_test = test[keys[0]], test[keys[1]]

        logger.info(
            "Server dataset loaded - "
            + f"Train: {len(self.x_train)}, Val: {len(self.x_validation)}, Test: {len(self.x_test)}"
        )

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

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """
        Initialize global model parameters.

        This method is called once at the beginning of the federated learning process.
        It returns the initial parameters for the global model.

        Args:
            client_manager: ClientManager to sample clients from.

        Returns:
            Initial parameters for the global model, or None to use random initialization.
        """
        # Return None to let Flower use the default initialization
        # Override this method if you need custom initialization
        return None

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluate the global model on the server side.

        This method is called after each round of training to evaluate the global model.
        It can be used to evaluate the model on a centralized test set.

        Args:
            server_round: Current server round.
            parameters: Model parameters to evaluate.

        Returns:
            A tuple of (loss, metrics) where loss is the evaluation loss and
            metrics is a dictionary of evaluation metrics. Returns None to skip
            centralized evaluation.
        """
        # Return None to skip centralized evaluation
        # Override this method if you want to evaluate on server-side data
        return None

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
        logger.info("=" * 60)
        logger.info(f"Round {server_round}/{self.conf.rounds} - Configuring training")
        logger.info("=" * 60)

        self.current_round = server_round

        # Step 1: Get ALL available clients from ClientManager
        # Use num_available() to get actual number of connected clients
        num_available = client_manager.num_available()
        logger.debug(f"Number of available clients: {num_available}")

        all_available_clients = client_manager.sample(
            num_clients=num_available,  # Sample all available clients
            min_num_clients=1,  # Accept at least 1 client (flexible)
        )
        logger.debug(f"Sampled clients: {len(all_available_clients)}")

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
            logger.debug("First round: Using all client IDs (UUID mapping not yet established)")

        # Step 3: Use client selection strategy to choose from AVAILABLE numeric CIDs
        selected_numeric_cids = self.client_selection.select(
            server=self,
            server_round=server_round,
            list_of_clients=available_numeric_cids,
        )
        self.selected_clients = selected_numeric_cids

        logger.info(
            f"Client selection ({self.conf.server.selection_method.name}): "
            + f"{len(selected_numeric_cids)}/{len(available_numeric_cids)} clients selected"
        )
        logger.info(f"Selected clients: {sorted([int(c) for c in selected_numeric_cids])}")

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
        logger.info(f"Round {server_round} - Aggregating training results")
        logger.info(f"Successful: {len(results)}, Failures: {len(failures)}")

        if failures:
            logger.warning(f"Training failures detected: {len(failures)}")
            for i, failure in enumerate(failures[:3]):  # Log first 3 failures
                logger.warning(f"Failure {i + 1}: {failure}")

        if not results:
            logger.error("No successful training results to aggregate!")
            return None, {}

        parameters, config = self.aggregate_method.agg_fit(
            server=self, server_round=server_round, results=results, failures=failures
        )

        logger.info(f"Aggregation ({self.conf.server.aggregation_method.name}) completed")
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
        logger.info(f"Round {server_round} - Configuring evaluation")
        config = {
            "rounds": server_round,
            "selected_by_server": ",".join(self.selected_clients),
        }
        logger.debug(f"Evaluation config: {config}")

        evaluate_ins = EvaluateIns(parameters=parameters, config=config)

        # Get all available clients
        num_available = client_manager.num_available()
        logger.debug(f"Number of available clients for evaluation: {num_available}")

        all_available_clients = client_manager.sample(
            num_clients=num_available,  # Sample all available clients
            min_num_clients=1,  # Accept at least 1 client
        )
        logger.debug(f"Available clients for evaluation: {len(all_available_clients)}")

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
        logger.info(f"Round {server_round} - Aggregating evaluation results")
        logger.info(f"Successful: {len(results)}, Failures: {len(failures)}")

        if failures:
            logger.warning(f"Evaluation failures detected: {len(failures)}")

        if not results:
            logger.error("No successful evaluation results to aggregate!")
            return None, {}

        loss, config = self.aggregate_method.agg_eval(self, server_round, results, failures)
        self._collect_clients_data(results)

        # Log round summary
        logger.info(f"Round {server_round} completed:")
        logger.info(f"  Average Accuracy: {self.clients_acc_avg:.4f}")
        logger.info(f"  Average Loss: {self.clients_loss_avg:.4f}")
        logger.info(
            f"  Participating clients: {np.count_nonzero(self.client_participating_state)}/{len(self.client_participating_state)}"  # noqa: E501
        )  # noqa: E501

        logger.log_metrics(
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
            "model_type": self.conf.model.type.lower(),
            "n_selected": len(self.selected_clients),
            "selection": f"[{';'.join(self.selected_clients)}]",
            "dataset": self.conf.dataset.dataset.lower(),
            "init_clients": self.conf.init_clients,
            "participating_state": f"[{';'.join([str(state) for state in self.client_participating_state])}]",  # noqa: E501
            "number_of_participating": np.count_nonzero(self.client_participating_state),
            "number_of_non_participating": np.count_nonzero(not self.client_participating_state),
            "training_method": self.conf.client.training_strategy.name.lower(),
            "aggregation_method": f"{self.conf.server.aggregation_method.name.lower()}",
            "selection_method": self.conf.server.selection_method.name.lower(),
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
        self.clients_loss_avg: float = np.mean(self.clients_loss)
