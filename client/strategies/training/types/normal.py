from typing import TYPE_CHECKING

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
            y_train_pred = np.argmax(y_train_pred_proba, axis=1)
            train_metrics = client.metrics_manager.calculate_all(
                client.y_train, y_train_pred, y_train_pred_proba
            )

            # Update fit_train_metrics
            client.fit_train_metrics["acc"] = train_metrics["accuracy"]
            client.fit_train_metrics["loss"] = np.mean(history.history["loss"])
            client.fit_train_metrics["precision"] = train_metrics["precision"]
            client.fit_train_metrics["recall"] = train_metrics["recall"]
            client.fit_train_metrics["f1_score"] = train_metrics["f1_score"]
            client.fit_train_metrics["auc"] = train_metrics["auc"]

            # Calculate validation metrics
            y_val_pred_proba = client.model.predict(client.x_validation, verbose=0)
            y_val_pred = np.argmax(y_val_pred_proba, axis=1)
            val_metrics = client.metrics_manager.calculate_all(
                client.y_validation, y_val_pred, y_val_pred_proba
            )

            # Update fit_val_metrics
            client.fit_val_metrics["acc"] = val_metrics["accuracy"]
            client.fit_val_metrics["loss"] = np.mean(history.history.get("val_loss", [0]))
            client.fit_val_metrics["precision"] = val_metrics["precision"]
            client.fit_val_metrics["recall"] = val_metrics["recall"]
            client.fit_val_metrics["f1_score"] = val_metrics["f1_score"]
            client.fit_val_metrics["auc"] = val_metrics["auc"]

            logger.info(
                f"Client {client.cid}: Training completed - "
                + f"Train Acc: {client.fit_train_metrics['acc']:.4f}, "
                + f"Train Loss: {client.fit_train_metrics['loss']:.4f}, "
                + f"Train P/R/AUC: {client.fit_train_metrics['precision']:.4f}/"
                + f"{client.fit_train_metrics['recall']:.4f}/"
                + f"{client.fit_train_metrics['auc']:.4f}, "
                + f"Val Acc: {client.fit_val_metrics['acc']:.4f}, "
                + f"Val Loss: {client.fit_val_metrics['loss']:.4f}, "
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

        # Evaluate the model using MetricsManager

        eval_pred_probs = client.model.predict(client.x_test, verbose=0)
        eval_metrics = client.metrics_manager.calculate_metrics(client.y_test, eval_pred_probs)
        eval_loss = client.model.evaluate(client.x_test, client.y_test, verbose=0)[0]

        client.eval_test_metrics["loss"] = float(eval_loss)
        client.eval_test_metrics["acc"] = eval_metrics["accuracy"]
        client.eval_test_metrics["precision"] = eval_metrics["precision"]
        client.eval_test_metrics["recall"] = eval_metrics["recall"]
        client.eval_test_metrics["f1_score"] = eval_metrics["f1_score"]
        client.eval_test_metrics["auc"] = eval_metrics["auc"]

        # Check for invalid values
        import math

        if math.isnan(client.eval_test_metrics["loss"]) or math.isinf(
            client.eval_test_metrics["loss"]
        ):
            logger.error(
                f"Client {client.cid}: INVALID LOSS VALUE: {client.eval_test_metrics['loss']}"
            )
        if math.isnan(client.eval_test_metrics["acc"]) or math.isinf(
            client.eval_test_metrics["acc"]
        ):
            logger.error(
                f"Client {client.cid}: INVALID ACCURACY VALUE: {client.eval_test_metrics['acc']}"
            )

        logger.info(
            f"Client {client.cid}: Evaluation completed - "
            + f"Acc: {client.eval_test_metrics['acc']:.4f}, "
            + f"Loss: {client.eval_test_metrics['loss']:.4f}, "
            + f"Precision: {client.eval_test_metrics['precision']:.4f}, "
            + f"Recall: {client.eval_test_metrics['recall']:.4f}, "
            + f"F1: {client.eval_test_metrics['f1_score']:.4f}, "
            + f"AUC: {client.eval_test_metrics['auc']:.4f}"
        )

        eval_resp = {
            "cid": client.cid,
            "acc": client.eval_test_metrics["acc"],
            "loss": client.eval_test_metrics["loss"],
            "precision": client.eval_test_metrics["precision"],
            "recall": client.eval_test_metrics["recall"],
            "f1_score": client.eval_test_metrics["f1_score"],
            "auc": client.eval_test_metrics["auc"],
            "participating_state": client.get_participating_state(),
        }

        return client.eval_test_metrics["loss"], client.x_test.shape[0], eval_resp
