from typing import TYPE_CHECKING, Any

import numpy as np
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)

from .....utils.logger import Logger
from .....utils.utils import Utils
from ...drivers.accuracy import AccuracyDriver
from ...drivers.driver import Driver
from ..base import TrainingStrategy
from ..structs import LetsFedTrainingStrategyConfig

if TYPE_CHECKING:
    from ...fl_client import FLClient

logger = Logger(__name__)


class LetsFedClient(TrainingStrategy):
    """
    Federated Learning strategy for the LetsFed framework.
    """

    def __init__(self, config: LetsFedTrainingStrategyConfig) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            client: The federated learning client instance.
        """
        super().__init__(config)
        self.add_drivers(self._get_drivers())
        logger.info("LetsFedClient training strategy initialized with drivers")

    @staticmethod
    def params_from_json(params: dict[str, Any]) -> LetsFedTrainingStrategyConfig:
        """
        Parse JSON parameters into LetsFedTrainingStrategyConfig.

        Args:
            params: Dictionary of parameters from YAML configuration.

        Returns:
            LetsFedTrainingStrategyConfig instance.
        """
        return LetsFedTrainingStrategyConfig(
            threshold_accuracy=params.get("threshold_accuracy", 1.0),
        )

    def _get_drivers(self) -> list[Driver]:
        """
        Get the list of drivers for the LetsFed client.

        Returns:
            list[Driver]: List of driver instances.
        """
        drivers = [
            AccuracyDriver(),
        ]
        logger.debug(f"LetsFedClient initialized with {len(drivers)} drivers")
        return drivers

    def fit(
        self, client: "FLClient", parameters: NDArrays, config: Config
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Train the model on the client data.

        Args:
            client (FLClient): The federated learning client.
            parameters (NDArrays): The model parameters.
            config (Config): Configuration dictionary.

        Returns:
            tuple[NDArrays, int, dict[str, Scalar]]: Updated model parameters,
            number of examples used for training, and additional metrics.
        """
        logger.debug(
            f"Client {client.cid}: Starting LetsFed fit - "
            + f"Participating: {client.get_participating_state()}"
        )

        # Initialize fit response
        fit_response = {
            "cid": client.cid,
            "participating_state": client.get_participating_state(),
        }

        # Calculate and store model size and parameter stats
        model_size = sum([layer.nbytes for layer in parameters])
        model_total_params = sum(w.size for w in parameters)
        model_mean_params = sum(w.mean() for w in parameters) / len(parameters)
        client.model_size = model_size
        client.model_total_params = model_total_params
        client.model_mean_params = model_mean_params
        logger.debug(
            f"Client {client.cid}: Model size = {model_size / (1024**2):.2f} MB, "
            + f"Received {len(parameters)} layers, "
            + f"{model_total_params} params, mean={model_mean_params:.6f}"
        )

        # Analyze if the client is selected
        client.selected = Utils.is_select_by_server(
            client.cid, config["selected_by_server"].split(",")
        )
        logger.info(
            f"Client {client.cid}: Selected = {client.selected}, "
            + f"Participating = {client.get_participating_state()}"
        )

        # Setting global parameters for selected clients if they want to participate
        if client.selected:
            if client.get_participating_state():
                logger.info(f"Client {client.cid}: Accepting global model and starting training")
                client.set_parameters(parameters)

            logger.info(
                f"Client {client.cid}: Starting training for {client.conf.client.epochs} epochs"
            )

            # Fitting model using TensorFlow datasets
            # The datasets are already batched, shuffled, and prefetched
            history = client.model.fit(
                client.train_dataset,
                epochs=client.conf.client.epochs,
                validation_data=client.val_dataset,
                verbose=0,
            )

            # Calculate training metrics
            y_train_pred_proba = client.model.predict(client.train_dataset, verbose=0)
            # Get true labels from train dataset
            y_train_true = np.concatenate([y for _, y in client.train_dataset], axis=0)
            client.fit_train_metrics = client.metrics_manager.calculate_metrics(
                y_train_true, y_train_pred_proba
            )
            client.fit_train_metrics["loss"] = np.mean(history.history["loss"])

            # Calculate validation metrics
            y_val_pred_proba = client.model.predict(client.val_dataset, verbose=0)
            # Get true labels from validation dataset
            y_val_true = np.concatenate([y for _, y in client.val_dataset], axis=0)
            client.fit_val_metrics = client.metrics_manager.calculate_metrics(
                y_val_true, y_val_pred_proba
            )
            client.fit_val_metrics["loss"] = np.mean(history.history.get("val_loss", [0]))

            # Log training metrics dynamically
            train_metrics_str = ", ".join(
                [f"{k.capitalize()}: {v:.4f}" for k, v in client.fit_train_metrics.items()]
            )
            val_metrics_str = ", ".join(
                [f"{k.capitalize()}: {v:.4f}" for k, v in client.fit_val_metrics.items()]
            )
            logger.info(
                f"Client {client.cid}: Training completed - "
                + f"Train [{train_metrics_str}], Val [{val_metrics_str}]"
            )
        else:
            logger.debug(f"Client {client.cid}: Skipping training (not selected)")

        # Return training parameters with dataset size from client metadata
        return client.get_parameters(config), client.train_size, fit_response

    def _manager_client_state(self, client: "FLClient") -> None:
        """
        Manage the client's participation state.

        Args:
            client (FLClient): The federated learning client.
        """
        old_state = client.get_participating_state()
        if not self.willing:
            client.set_participating_state(False)
            logger.info(f"Client {client.cid}: Changed participation state to False (not willing)")

        if self.willing:
            client.set_participating_state(True)
            if not old_state:
                logger.info(f"Client {client.cid}: Changed participation state to True (willing)")

    def evaluate(
        self, client: "FLClient", parameters: NDArrays, config: Config
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Evaluate the model on the client test data.

        Args:
            client (FLClient): The federated learning client.
            parameters (NDArrays): The model parameters.
            config (Config): Configuration dictionary.

        Returns:
            tuple[NDArrays, int, dict[str, Scalar]]: Loss, number of examples used for evaluation,
            and additional metrics.
        """
        logger.debug(f"Client {client.cid}: Starting LetsFed evaluation")

        # Analyze if the client is selected
        client.selected = Utils.is_select_by_server(
            str(client.cid), config["selected_by_server"].split(",")
        )

        # Analyzing if the client is selected by the server
        if client.selected:
            logger.debug(f"Client {client.cid}: Applying drivers for evaluation")
            # Set global model weights and apply drivers
            self.apply_drivers(client, parameters, config)
            self._manager_client_state(client)

            if client.get_participating_state():
                logger.debug(f"Client {client.cid}: Setting global parameters for evaluation")
                client.set_parameters(parameters)

        else:
            logger.debug(f"Client {client.cid}: Using existing weights (not selected)")

        # Evaluate the model
        eval_pred_probs = client.model.predict(client.test_dataset, verbose=0)
        eval_loss = client.model.evaluate(client.test_dataset, verbose=0)[0]
        # Get true labels from test dataset
        y_test_true = np.concatenate([y for _, y in client.test_dataset], axis=0)
        client.eval_test_metrics = client.metrics_manager.calculate_metrics(
            y_test_true, eval_pred_probs
        )  # noqa: E501
        client.eval_test_metrics["loss"] = float(eval_loss)

        # Check for invalid values
        import math

        if math.isnan(client.eval_test_metrics.get("loss", 0)) or math.isinf(
            client.eval_test_metrics.get("loss", 0)
        ):
            logger.error(
                f"Client {client.cid}: INVALID LOSS VALUE: {client.eval_test_metrics['loss']}"
            )
        if math.isnan(client.eval_test_metrics.get("accuracy", 0)) or math.isinf(
            client.eval_test_metrics.get("accuracy", 0)
        ):
            logger.error(
                f"Client {client.cid}: INVALID ACCURACY VALUE: "
                + f"{client.eval_test_metrics.get('accuracy', 0)}"
            )

        # Build eval_resp dynamically from eval_test_metrics
        eval_resp = {
            "cid": client.cid,
            "participating_state": client.get_participating_state(),
            **client.eval_test_metrics,  # Include all metrics dynamically
        }

        # Log evaluation metrics dynamically
        eval_metrics_str = ", ".join(
            [f"{k.capitalize()}: {v:.4f}" for k, v in client.eval_test_metrics.items()]
        )
        logger.info(
            f"Client {client.cid}: Evaluation completed - "
            + f"[{eval_metrics_str}], Participating: {client.get_participating_state()}"
        )

        # Return evaluation metrics with dataset size from client metadata
        return client.eval_test_metrics["loss"], client.test_size, eval_resp
