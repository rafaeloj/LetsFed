import copy
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
from ..structs import QFFLTrainingStrategyConfig

if TYPE_CHECKING:
    from ...fl_client import FLClient

logger = Logger(__name__)


class QFFLClient(TrainingStrategy):
    """
    Quantized Federated Learning (QFFL) client strategy.
    """

    def __init__(self, config: QFFLTrainingStrategyConfig) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            client: The training strategy configuration.
        """
        super().__init__(config)
        logger.info("QFFLClient training strategy initialized")

    @staticmethod
    def params_from_json(params: dict[str, Any]) -> QFFLTrainingStrategyConfig:
        """
        Parse JSON parameters into QFFLTrainingStrategyConfig.

        Args:
            params: Dictionary of parameters from YAML configuration.

        Returns:
            QFFLTrainingStrategyConfig instance.
        """
        return QFFLTrainingStrategyConfig(eta=params.get("eta", 1.0))

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
        logger.debug(f"Client {client.cid}: Starting QFFL fit operation")
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
        logger.info(f"Client {client.cid}: Selected = {client.selected}")

        # Fitting model
        if client.selected:
            # Setting the parameters
            client.set_parameters(parameters)
            prev_model_parameters = copy.deepcopy(client.get_parameters(config))
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

            # Calculate delta parameters and scale by eta
            # delta = (current - previous) / eta
            # eta controls the step size: larger eta = smaller updates
            eta = self.config.eta
            delta_parameters = [
                (curr - prev) / eta
                for curr, prev in zip(
                    client.get_parameters(config), prev_model_parameters, strict=True
                )
            ]

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

            # Add training loss to fit_response for QFFL aggregation
            fit_response["loss"] = client.fit_train_metrics["loss"]

            # Return delta parameters with dataset size from client metadata
            return delta_parameters, client.train_size, fit_response

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
        logger.debug(f"Client {client.cid}: Starting QFFL evaluation")
        # Analyze if the client is selected
        client.selected = Utils.is_select_by_server(
            client.cid, config["selected_by_server"].split(",")
        )
        logger.info(f"Client {client.cid}: Selected = {client.selected}")

        # Set weights if selected
        if client.selected:
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

        # Log evaluation metrics dynamically
        eval_metrics_str = ", ".join(
            [f"{k.capitalize()}: {v:.4f}" for k, v in client.eval_test_metrics.items()]
        )
        logger.info(f"Client {client.cid}: Evaluation completed - [{eval_metrics_str}]")

        # Build eval_resp dynamically from eval_test_metrics
        eval_resp = {
            "cid": client.cid,
            "participating_state": client.get_participating_state(),
            **client.eval_test_metrics,  # Include all metrics dynamically
        }

        # Return evaluation metrics with dataset size from client metadata
        return client.eval_test_metrics["loss"], client.test_size, eval_resp
