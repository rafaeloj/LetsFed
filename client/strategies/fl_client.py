import flwr as fl
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)
from keras import Model

from ...conf.structs import Environment
from ...dataset_manager.dataset_manager import DSManager
from ...metrics import MetricsManager
from ...model.model_manager import ModelManager
from ...utils.logger import Logger
from .parameters_strategy.base import ParametersStrategy
from .training.base import TrainingStrategy

logger = Logger(__name__)


class FLClient(fl.client.NumPyClient):
    """
    Federated Client for Flower.

    This class implements the Flower NumPyClient interface and
    encapsulates the federated learning client logic, including
    data loading, model management, state management, and training strategy.
    """

    def __init__(
        self,
        cid: str,
        config: Environment,
        training_strategy: TrainingStrategy,
        parameters_strategy: ParametersStrategy,
        metrics_manager: MetricsManager,
    ) -> None:
        # Initialize the federated client.
        logger.info(f"Initializing FLClient {cid}")
        self.cid: str = cid
        self.conf: Environment = config
        self.training_strategy: TrainingStrategy = training_strategy
        self.parameters_strategy: ParametersStrategy = parameters_strategy
        self.metrics_manager: MetricsManager = metrics_manager

        # Load data and model
        self.model: Model
        self.x_train, self.y_train = None, None
        self.x_validation, self.y_validation = None, None
        self.x_test, self.y_test = None, None
        self._load_data()
        self._load_model()

        # Initialize client parameters
        self.participating_state: bool = True
        self.selected: bool = False
        self.model_size: int = 0
        self.model_total_params: int = 0
        self.model_mean_params: float = 0.0

        # Initialize metrics dictionaries dynamically based on configured metrics
        # Always include 'loss' as it comes from model.fit/evaluate
        metric_names = ["loss"] + self.metrics_manager.get_metrics_names()

        self.fit_train_metrics: dict[str, float] = dict.fromkeys(metric_names, 0.0)
        self.fit_val_metrics: dict[str, float] = dict.fromkeys(metric_names, 0.0)
        self.eval_test_metrics: dict[str, float] = dict.fromkeys(metric_names, 0.0)

        # extra metrics generated during training by training strategies
        self.data_to_log: dict = {}

        logger.info(
            f"Client {cid} initialized successfully with "
            + f"{len(self.x_train)} training samples, "
            + f"{len(self.x_validation)} validation samples, "
            + f"{len(self.x_test)} test samples"
        )

    def _load_model(self) -> None:
        """
        Load the model.
        """
        logger.debug(f"Client {self.cid}: Loading model")
        mm = ModelManager(conf=self.conf, input_shape=self.x_train.shape)
        self.model = mm.get_model()
        logger.info(f"Client {self.cid}: Model loaded successfully")

    def _load_data(self) -> None:
        """
        Load the data.
        """
        logger.info(f"Client {self.cid}: Loading dataset partition")
        dm = DSManager(n_clients=self.conf.n_clients, conf=self.conf.dataset, seed=self.conf.seed)

        train, validation, test = dm.load_locally(partition_id=int(self.cid))
        keys = list(test.features.keys())

        # Get label names
        self.labels = test.features["label"].names

        self.x_train, self.y_train = train[keys[0]], train[keys[1]]
        self.x_validation, self.y_validation = validation[keys[0]], validation[keys[1]]
        self.x_test, self.y_test = test[keys[0]], test[keys[1]]

        # Normalize image data to [-1, 1] range for better convergence
        # This centers the data around 0, which works better with gradient descent
        # and modern weight initialization methods (Xavier/He)
        self.x_train = (self.x_train.astype("float32") - 127.5) / 127.5
        self.x_validation = (self.x_validation.astype("float32") - 127.5) / 127.5
        self.x_test = (self.x_test.astype("float32") - 127.5) / 127.5

        # Add channel dimension for CNN models (grayscale images need shape: height x width x 1)
        # This is required for Conv2D layers which expect 4D input: (batch, height, width, channels)
        if self.conf.model.type == "cnn" and len(self.x_train.shape) == 3:
            import numpy as np

            self.x_train = np.expand_dims(self.x_train, axis=-1)
            self.x_validation = np.expand_dims(self.x_validation, axis=-1)
            self.x_test = np.expand_dims(self.x_test, axis=-1)
            logger.debug(
                f"Client {self.cid}: Reshaped data for CNN - " + f"New shape: {self.x_train.shape}"
            )

        logger.info(
            f"Client {self.cid}: Data loaded and normalized to [-1, 1] - "
            + f"Train: {len(self.x_train)}, Val: {len(self.x_validation)}, "
            + f"Test: {len(self.x_test)} samples"
        )

    def get_participating_state(self) -> bool:
        """
        Get the participating state of the client.
        """
        return self.participating_state

    def set_participating_state(self, state: bool) -> None:
        """
        Set the participating state of the client.

        Args:
            state: New participating state
        """
        if state != self.participating_state:
            logger.debug(f"Client {self.cid}: Participation state changed to {state}")
        self.participating_state = state

    def get_parameters(self, config: Config) -> NDArrays:
        """
        Get the model parameters to send to server.

        Uses parameters strategy to determine which parameters to send.

        Args:
            config: Configuration dictionary

        Returns:
            Model parameters as NDArrays (may be partial based on strategy)
        """

        return self.parameters_strategy.get_parameters(self)

    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Set the model parameters received from server.

        Uses parameters strategy to merge received parameters with local parameters.

        Args:
            parameters: Model parameters as NDArrays (may be partial based on strategy)
        """
        self.parameters_strategy.set_parameters(self, parameters)

    def fit(self, parameters: NDArrays, config: Config) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Fit the model on the provided parameters.

        Args:
            parameters: Model parameters
            config: Configuration dictionary

        Returns:
            Updated model parameters, number of examples used for training, and additional metrics
        """
        logger.info(f"Client {self.cid}: Starting fit for round {config.get('rounds', 'unknown')}")
        result = self.training_strategy.fit(self, parameters, config)
        logger.debug(f"Client {self.cid}: Fit completed")
        return result

    def evaluate(
        self, parameters: NDArrays, config: Config
    ) -> tuple[float, int, dict[str, Scalar]]:
        """
        Evaluate the model on the provided parameters.

        Args:
            parameters: Model parameters
            config: Configuration dictionary

        Returns:
            Loss, number of examples used for evaluation, and additional metrics.
        """
        logger.info(
            f"Client {self.cid}: Starting evaluation for round {config.get('rounds', 'unknown')}"
        )
        loss, size, conf = self.training_strategy.evaluate(self, parameters, config)
        logger.log_metrics(f"/c-data-{self.cid}.csv", data=self.get_log_data(config))
        logger.debug(f"Client {self.cid}: Evaluation completed")
        return loss, size, conf

    def get_log_data(self, config: Config) -> dict[str, Scalar]:
        """
        Get the log data for the client.

        Args:
            config: Configuration dictionary

        Returns:
            Dictionary containing log data
        """
        # Build log data dynamically based on configured metrics
        log_data = {
            "rounds": config["rounds"],
            "participating_state": self.participating_state,
            "selected": self.selected,
            "cid": self.cid,
            # Model parameters
            "model_size": self.model_size,
            "model_total_params": self.model_total_params,
            "model_mean_params": self.model_mean_params,
        }

        # Add fit train metrics dynamically with prefix
        for metric_name, value in self.fit_train_metrics.items():
            log_data[f"fit_train_{metric_name}"] = value

        # Add fit validation metrics dynamically with prefix
        for metric_name, value in self.fit_val_metrics.items():
            log_data[f"fit_val_{metric_name}"] = value

        # Add evaluation test metrics dynamically with prefix
        for metric_name, value in self.eval_test_metrics.items():
            log_data[f"eval_test_{metric_name}"] = value

        # Add configuration and extra data
        log_data.update(
            {
                "training_method": self.conf.client.training_strategy,
                "aggregation_method": f"{self.conf.server.aggregation_method}",
                "selection_method": self.conf.server.selection_method,
                **self.data_to_log,
            }
        )

        return log_data
