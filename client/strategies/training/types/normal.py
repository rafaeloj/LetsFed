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
        return NormalTrainingStrategyConfig(learning_rate=params.get("learning_rate", 0.01))

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

            # Fitting model with batch size from config
            batch_size = client.conf.dataset.batch_size
            history = client.model.fit(
                client.x_train,
                client.y_train,
                epochs=client.conf.client.epochs,
                batch_size=batch_size,
                validation_data=(client.x_validation, client.y_validation),
                verbose=0,
            )

            # Calculate training metrics
            y_train_pred_proba = client.model.predict(client.x_train, verbose=0)
            client.fit_train_metrics = client.metrics_manager.calculate_metrics(
                client.y_train, y_train_pred_proba
            )
            client.fit_train_metrics["loss"] = np.mean(history.history["loss"])

            # Calculate validation metrics
            y_val_pred_proba = client.model.predict(client.x_validation, verbose=0)
            client.fit_val_metrics = client.metrics_manager.calculate_metrics(
                client.y_validation, y_val_pred_proba
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

        return client.get_parameters(config), client.x_train.shape[0], fit_response

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

        # Evaluate the model
        eval_pred_probs = client.model.predict(client.x_test, verbose=0)
        eval_loss = client.model.evaluate(client.x_test, client.y_test, verbose=0)[0]
        client.eval_test_metrics = client.metrics_manager.calculate_metrics(
            client.y_test, eval_pred_probs
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

        return client.eval_test_metrics["loss"], client.x_test.shape[0], eval_resp
