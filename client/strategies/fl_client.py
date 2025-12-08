import flwr as fl
import tensorflow as tf
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)
from keras import Model

from ...conf.structs import Environment
from ...dataset_manager.dataloader import create_federated_datasets
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

        # TensorFlow datasets for training/validation/test
        self.train_dataset: tf.data.Dataset
        self.val_dataset: tf.data.Dataset
        self.test_dataset: tf.data.Dataset

        # Dataset metadata
        self.labels: list[str]
        self.num_classes: int
        self.input_shape: tuple
        self.train_size: int
        self.val_size: int
        self.test_size: int

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
            + "train/val/test datasets using TensorFlow"
        )

    def _load_model(self) -> None:
        """
        Load the model.
        """
        logger.debug(f"Client {self.cid}: Loading model")
        mm = ModelManager(conf=self.conf, input_shape=self.input_shape)
        self.model = mm.get_model()
        logger.info(f"Client {self.cid}: Model loaded successfully")

    def _load_data(self) -> None:
        """
        Load the data using TensorFlow datasets.

        This method creates tf.data.Dataset objects for train/validation/test
        using the create_federated_datasets factory function.

        The TensorFlow datasets are used for training (model.fit) and
        evaluation (model.evaluate, model.predict) as they provide:
        - Native Keras compatibility
        - Optimized performance with batching, shuffling, prefetching
        - Memory efficiency for large datasets
        """
        logger.info(f"Client {self.cid}: Loading dataset partition {self.cid}")

        # Create TensorFlow datasets using factory function
        self.train_dataset, self.val_dataset, self.test_dataset = create_federated_datasets(
            partition_id=int(self.cid), conf=self.conf
        )

        # Extract metadata from the first batch for model initialization
        # Get a single batch to determine input shape and number of classes
        for x_batch, y_batch in self.train_dataset.take(1):
            # Store input shape (excluding batch dimension)
            self.input_shape = x_batch.shape[1:]
            # Determine number of classes from labels
            self.num_classes = len(tf.unique(y_batch)[0])

        # Calculate dataset sizes by counting samples in each dataset
        # Note: This unbatches and counts all samples, done once during initialization
        self.train_size = sum(1 for _ in self.train_dataset.unbatch())
        self.val_size = sum(1 for _ in self.val_dataset.unbatch())
        self.test_size = sum(1 for _ in self.test_dataset.unbatch())

        # Set labels - will be populated from config or dataset metadata
        # For now, create generic labels based on num_classes
        self.labels = [f"class_{i}" for i in range(self.num_classes)]

        logger.info(
            f"Client {self.cid}: TensorFlow datasets created - "
            + f"Input shape: {self.input_shape}, "
            + f"Classes: {self.num_classes}, "
            + f"Sizes (train/val/test): {self.train_size}/{self.val_size}/{self.test_size}"
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
