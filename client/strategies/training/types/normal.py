from typing import TYPE_CHECKING, Any

import numpy as np
from flwr.common import Config, NDArrays, Scalar

from .....utils.logger import Logger
from .....utils.utils import Utils
from ..base import TrainingStrategy
from ..structs import NormalTrainingStrategyConfig

if TYPE_CHECKING:
    from ..base import FLClient

logger = Logger(__name__)


class NormalClient(TrainingStrategy):
    """
    Normal training strategy for federated learning clients.
    """

    def __init__(self, config: NormalTrainingStrategyConfig) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            config: The training strategy configuration.
        """
        super().__init__(config)
        logger.info("NormalClient training strategy initialized")

    @staticmethod
    def params_from_json(params: dict[str, Any]) -> NormalTrainingStrategyConfig:
        """
        Parse JSON parameters into NormalTrainingStrategyConfig.

        Args:
            params: Dictionary of parameters from YAML configuration.

        Returns:
            NormalTrainingStrategyConfig instance.
        """
        return NormalTrainingStrategyConfig()

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
        logger.debug(f"Client {client.cid}: Starting fit operation")

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

        # Anayze if the client is selected
        client.selected = Utils.is_select_by_server(
            client.cid, config["selected_by_server"].split(",")
        )
        logger.info(f"Client {client.cid}: Selected = {client.selected}")

        # Fitting model
        if client.selected:
            logger.info(
                (
                    f"Client {client.cid}: Starting training for "
                    + f"{client.conf.client.epochs} epochs"
                )
            )
            # Setting the parameters
            client.set_parameters(parameters)

            # Fitting model using TensorFlow datasets
            # The datasets are already batched, shuffled, and prefetched
            history = client.model.fit(
                client.train_dataset,
                epochs=client.conf.client.epochs,
                validation_data=client.val_dataset,
                verbose=0,
            )

            # Calculate training metrics
            # Important: Extract X and y from train_dataset to avoid shuffle issues
            # The train_dataset has reshuffle_each_iteration=True, which would cause
            # misalignment between predictions and true labels if we iterate multiple times
            x_train_batches = []
            y_train_batches = []
            for x_batch, y_batch in client.train_dataset:
                x_train_batches.append(x_batch)
                y_train_batches.append(y_batch)

            x_train = np.concatenate(x_train_batches, axis=0)
            y_train_true = np.concatenate(y_train_batches, axis=0)

            y_train_pred_proba = client.model.predict(x_train, verbose=0)
            client.fit_train_metrics = client.metrics_manager.calculate_metrics(
                y_train_true, y_train_pred_proba
            )
            client.fit_train_metrics["loss"] = np.mean(history.history["loss"])

            # Calculate validation metrics (validation dataset doesn't have shuffle, so it's safe)
            y_val_pred_proba = client.model.predict(client.val_dataset, verbose=0)
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
        logger.debug(f"Client {client.cid}: Starting evaluation")

        # Analyze if the client is selected
        client.selected = Utils.is_select_by_server(
            client.cid, config["selected_by_server"].split(",")
        )

        logger.debug(f"Client {client.cid}: Selected for evaluation = {client.selected}")

        # Set weights if selected
        if client.selected:
            logger.debug(f"Client {client.cid}: Setting parameters for evaluation")
            client.set_parameters(parameters)

        else:
            logger.debug(f"Client {client.cid}: Using existing weights (not selected)")

        # Evaluate the model (test dataset doesn't have shuffle, so it's safe)
        eval_pred_probs = client.model.predict(client.test_dataset, verbose=0)
        eval_loss = client.model.evaluate(client.test_dataset, verbose=0)[0]
        y_test_true = np.concatenate([y for _, y in client.test_dataset], axis=0)
        client.eval_test_metrics = client.metrics_manager.calculate_metrics(
            y_test_true, eval_pred_probs
        )
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
