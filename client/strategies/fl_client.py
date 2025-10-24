import flwr as fl
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)
from keras import Model

from ...conf import Environment
from ...dataset_manager.dataset_manager import DSManager
from ...model.model_manager import ModelManager
from ...utils.logger import Logger
from .drivers.context import DriverContext
from .drivers.driver import Driver
from .training.base import TrainingStrategy

logger = Logger(__name__)


class FLClient(fl.client.NumPyClient):
    """
    Federated Client for Flower.

    This class implements the Flower NumPyClient interface and
    encapsulates the federated learning client logic, including
    data loading, model management, state management, and training strategy.
    """

    def __init__(self, cid: str, config: Environment, training_strategy: TrainingStrategy) -> None:
        # Initialize the federated client.
        self.cid: str = cid
        self.conf: Environment = config
        self.training_strategy: TrainingStrategy = training_strategy

        # Load data and model
        self.model: Model
        self.x_train, self.y_train = None, None
        self.x_validation, self.y_validation = None, None
        self.x_test, self.y_test = None, None
        self.load_data()
        self.load_model()

        # Initialize client parameters
        self.drivers: list[Driver] = []
        self.participating_state: bool = True
        self.desired_state: bool = True
        self.selected: bool = False
        self.model_size: int = 0
        self.g_eval_acc: float = 0
        self.g_fit_acc: float = 0
        self.g_eval_loss: float = 0
        self.g_fit_loss: float = 0
        self.data_to_log: dict = {}

        # Initialize training strategy
        self.training_strategy.init(self)

    def load_model(self) -> None:
        """
        Load the model.
        """
        mm = ModelManager(conf=self.conf, input_shape=self.x_train.shape)
        self.model = mm.get_model()

    def load_data(self) -> None:
        """
        Load the data.
        """
        dm = DSManager(n_clients=self.conf.n_clients, conf=self.conf.db)

        train, validation, test = dm.load_locally(partition_id=int(self.cid))
        keys = list(test.features.keys())

        # Get label names
        self.labels = test.features["label"].names

        self.x_train, self.y_train = train[keys[0]], train[keys[1]]
        self.x_validation, self.y_validation = validation[keys[0]], validation[keys[1]]
        self.x_test, self.y_test = test[keys[0]], test[keys[1]]

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
        self.participating_state = state

    def get_parameters(self) -> NDArrays:
        """
        Get the model parameters.

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

    def add_drivers(self, drivers: list[Driver]) -> None:
        """
        Add a list of drivers to the client.
        Implements the Plugin Architecture pattern.

        Args:
            drivers: List of driver instances to add
        """
        self.drivers.extend(drivers)

    def apply_drivers(self, parameters: NDArrays, config: Config) -> dict[str, float | int | bool]:
        """
        Apply all registered drivers in sequence using DriverContext.
        Implements the Chain of Responsibility pattern with explicit side effects.

        Args:
            parameters: Model parameters
            config: Configuration dictionary

        Returns:
            Dictionary of all modifications made by drivers
        """
        context = DriverContext()

        # Run all drivers, collecting their results in the context
        for driver in self.drivers:
            driver.run(self, parameters, config, context)

        # Apply all modifications from context to client attributes
        modifications = context.get_all()
        for key, value in modifications.items():
            setattr(self, key, value)

        return modifications

    def get_drivers(self) -> list[Driver]:
        """
        Get the list of registered drivers.

        Returns:
            List of driver instances
        """
        return self.drivers

    def fit(self, parameters: NDArrays, config: Config) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Fit the model on the provided parameters.

        Args:
            parameters: Model parameters
            config: Configuration dictionary

        Returns:
            Updated model parameters, number of examples used for training, and additional metrics
        """
        return self.training_strategy.fit(self, parameters, config)

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
        loss, size, conf = self.training_strategy.evaluate(self, parameters, config)
        logger.log(f"/c-data-{self.cid}.csv", data=self.get_log_data(config))
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
            "desired_state": self.desired_state,
            "selected": self.selected,
            "cid": self.cid,
            "model_size": self.model_size,
            "g_fit_acc": self.g_fit_acc,
            "g_fit_loss": self.g_fit_loss,
            "g_eval_acc": self.g_eval_acc,
            "g_eval_loss": self.g_eval_loss,
            "training_method": self.conf.client.training_strategy,
            "aggregation_method": f"{self.conf.server.aggregation.method}",
            "selection_method": self.conf.server.selection.method,
            **self.data_to_log,
        }
