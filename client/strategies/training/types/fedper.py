from typing import TYPE_CHECKING, Any

import numpy as np
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)

from .....utils.logger import Logger
from .....utils.utils import Utils
from ..base import TrainingStrategy
from ..structs import FedPerTrainingStrategyConfig

if TYPE_CHECKING:
    from ...client_builder import FLClient

logger = Logger(__name__)


class FedPerClient(TrainingStrategy):
    """
    FedPer training strategy for federated learning clients.
    """

    def __init__(self, config: FedPerTrainingStrategyConfig) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            config: The training strategy configuration.
        """
        super().__init__(config)
        logger.info("FedPerClient training strategy initialized")

    @staticmethod
    def params_from_json(params: dict[str, Any]) -> FedPerTrainingStrategyConfig:
        """
        Parse JSON parameters into FedPerTrainingStrategyConfig.

        Args:
            params: Dictionary of parameters from YAML configuration.

        Returns:
            FedPerTrainingStrategyConfig instance.
        """
        return FedPerTrainingStrategyConfig()

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
        logger.debug(f"Client {client.cid}: Starting FedPer fit operation")
        # Initialize fit response
        fit_response = {"cid": client.cid, "participating_state": client.get_participating_state()}

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

        # Determine if client is selected by server
        client.selected = Utils.is_select_by_server(
            client.cid, config["selected_by_server"].split(",")
        )
        logger.info(f"Client {client.cid}: Selected = {client.selected}")

        if client.selected:
            # Store the last layer
            last_layer = client.model.layers[-1]

            # Set weights and replace the last layer with the stored one
            client.set_parameters(parameters)
            client.model.layers[-1] = last_layer

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

        # Return training parameters with dataset size from client metadata
        return client.get_parameters(config), client.train_size, fit_response

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
        logger.debug(f"Client {client.cid}: Starting FedPer evaluation")
        # Determine if client is selected by server
        client.selected = Utils.is_select_by_server(
            str(client.cid), config["selected_by_server"].split(",")
        )
        logger.info(f"Client {client.cid}: Selected = {client.selected}")

        # Set weights and replace the last layer with the stored one
        if client.selected:
            last_layer = client.model.layers[-1]
            client.set_parameters(parameters)
            client.model.layers[-1] = last_layer

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

        # Log evaluation metrics dynamically
        eval_metrics_str = ", ".join(
            [f"{k.capitalize()}: {v:.4f}" for k, v in client.eval_test_metrics.items()]
        )
        logger.info(f"Client {client.cid}: Evaluation completed - [{eval_metrics_str}]")

        # Build evaluation_response dynamically from eval_test_metrics
        evaluation_response = {
            "cid": client.cid,
            "participating_state": client.get_participating_state(),
            **client.eval_test_metrics,  # Include all metrics dynamically
        }

        # Return evaluation metrics with dataset size from client metadata
        return client.eval_test_metrics["loss"], client.test_size, evaluation_response
