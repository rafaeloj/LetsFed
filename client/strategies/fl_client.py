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
        metrics_manager: MetricsManager,
    ) -> None:
        # Initialize the federated client.
        logger.info(f"Initializing FLClient {cid}")
        self.cid: str = cid
        self.conf: Environment = config
        self.training_strategy: TrainingStrategy = training_strategy
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

        # Metrics dictionaries for each phase
        self.fit_train_metrics: dict[str, float] = {
            "acc": 0.0,
            "loss": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "auc": 0.0,
        }
        self.fit_val_metrics: dict[str, float] = {
            "acc": 0.0,
            "loss": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "auc": 0.0,
        }
        self.eval_test_metrics: dict[str, float] = {
            "acc": 0.0,
            "loss": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "auc": 0.0,
        }

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
        dm = DSManager(n_clients=self.conf.n_clients, conf=self.conf.dataset)

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
        Get the model parameters.

        Args:
            config: Configuration dictionary

        Returns:
            Model parameters as NDArrays
        """
        return self.model.get_weights()

    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Set the model parameters.

        Args:
            parameters: Model parameters as NDArrays
        """
        self.model.set_weights(parameters)

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
        logger.debug(
            f"Client {self.cid}: Evaluation completed - "
            + f"Loss: {loss:.4f}, Accuracy: {conf.get('acc', 0):.4f}"
        )
        return loss, size, conf

    def get_log_data(self, config: Config) -> dict[str, Scalar]:
        """
        Get the log data for the client.

        Args:
            config: Configuration dictionary

        Returns:
            Dictionary containing log data
        """
        return {
            "rounds": config["rounds"],
            "participating_state": self.participating_state,
            "selected": self.selected,
            "cid": self.cid,
            # Model parameters
            "model_size": self.model_size,
            "model_total_params": self.model_total_params,
            "model_mean_params": self.model_mean_params,
            # Fit metrics - Training
            "fit_train_acc": self.fit_train_metrics["acc"],
            "fit_train_loss": self.fit_train_metrics["loss"],
            "fit_train_precision": self.fit_train_metrics["precision"],
            "fit_train_recall": self.fit_train_metrics["recall"],
            "fit_train_f1_score": self.fit_train_metrics["f1_score"],
            "fit_train_auc": self.fit_train_metrics["auc"],
            # Fit metrics - Validation
            "fit_val_acc": self.fit_val_metrics["acc"],
            "fit_val_loss": self.fit_val_metrics["loss"],
            "fit_val_precision": self.fit_val_metrics["precision"],
            "fit_val_recall": self.fit_val_metrics["recall"],
            "fit_val_f1_score": self.fit_val_metrics["f1_score"],
            "fit_val_auc": self.fit_val_metrics["auc"],
            # Evaluation metrics
            "eval_test_acc": self.eval_test_metrics["acc"],
            "eval_test_loss": self.eval_test_metrics["loss"],
            "eval_test_precision": self.eval_test_metrics["precision"],
            "eval_test_recall": self.eval_test_metrics["recall"],
            "eval_test_f1_score": self.eval_test_metrics["f1_score"],
            "eval_test_auc": self.eval_test_metrics["auc"],
            # Configuration
            "training_method": self.conf.client.training_strategy,
            "aggregation_method": f"{self.conf.server.aggregation_method}",
            "selection_method": self.conf.server.selection_method,
            **self.data_to_log,
        }
