from typing import TYPE_CHECKING

import numpy as np
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)

from .....utils.logger import Logger
from .....utils.utils import Utils
from ...drivers.driver import Driver
from ...drivers.maxfl_qk import MaxFLQkDriver
from ..base import TrainingStrategy
from ..structs import MaxFLTrainingStrategyConfig

if TYPE_CHECKING:
    from ...fl_client import FLClient

logger = Logger(__name__)


class MaxFLClient(TrainingStrategy):
    """
    MaxFL training strategy for federated learning clients.
    """

    def __init__(self, config: MaxFLTrainingStrategyConfig) -> None:
        """
        Method to initialize parameters of specific solution

        Args:
            config: The training strategy configuration.
        """
        super().__init__(config)
        self.qk = 1
        self.add_drivers(self._get_drivers())
        logger.info(f"MaxFLClient initialized with qk_threshold={config.maxfl_qk_threshold}")

    def _get_drivers(self) -> list[Driver]:
        """
        Get the list of drivers for the MaxFL client.

        Returns:
            list[Driver]: List of driver instances.
        """
        drivers = [MaxFLQkDriver()]
        logger.debug(f"MaxFLClient initialized with {len(drivers)} drivers")
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
            f"Client {client.cid}: Starting MaxFL fit - "
            + f"qk={self.qk:.4f}, Participating: {client.get_participating_state()}"
        )

        # Initialize fit response
        fit_response = {
            "cid": client.cid,
            "participating_state": client.get_participating_state(),
            "qk": self.qk,
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
            + f"Participating = {client.get_participating_state()}, qk = {self.qk:.4f}"
        )

        # Fitting model
        if client.selected:
            # Setting the parameters if client wants to participate
            if client.get_participating_state():
                logger.info(
                    f"Client {client.cid}: Accepting global model and training "
                    + f"(qk={self.qk:.4f})"
                )
                client.set_parameters(parameters)

                # Fitting model with batch size from config and validation data
                batch_size = client.conf.dataset.batch_size
                history = client.model.fit(
                    client.x_train,
                    client.y_train,
                    epochs=client.conf.client.epochs,
                    batch_size=batch_size,
                    validation_data=(client.x_validation, client.y_validation),
                    verbose=0,
                )
                client.fit_train_acc = np.mean(history.history["accuracy"])
                client.fit_train_loss = np.mean(history.history["loss"])
                client.fit_train_precision = np.mean(history.history.get("precision", [0]))
                client.fit_train_recall = np.mean(history.history.get("recall", [0]))
                client.fit_train_auc = np.mean(history.history.get("auc", [0]))

                client.fit_val_acc = np.mean(history.history.get("val_accuracy", [0]))
                client.fit_val_loss = np.mean(history.history.get("val_loss", [0]))
                client.fit_val_precision = np.mean(history.history.get("val_precision", [0]))
                client.fit_val_recall = np.mean(history.history.get("val_recall", [0]))
                client.fit_val_auc = np.mean(history.history.get("val_auc", [0]))

                # Evaluate on test set immediately after training
                test_results = client.model.evaluate(client.x_test, client.y_test, verbose=0)
                # test_results = [loss, accuracy, precision, recall, auc]
                client.fit_test_loss = float(test_results[0])
                client.fit_test_acc = float(test_results[1])
                client.fit_test_precision = float(test_results[2])
                client.fit_test_recall = float(test_results[3])
                client.fit_test_auc = float(test_results[4])

                logger.info(
                    f"Client {client.cid}: Training completed - "
                    + f"Train Acc: {client.fit_train_acc:.4f}, "
                    + f"Train Loss: {client.fit_train_loss:.4f}, "
                    + f"Train P/R/AUC: {client.fit_train_precision:.4f}/"
                    + f"{client.fit_train_recall:.4f}/{client.fit_train_auc:.4f}, "
                    + f"Val Acc: {client.fit_val_acc:.4f}, "
                    + f"Val Loss: {client.fit_val_loss:.4f}, "
                    + f"Test Acc: {client.fit_test_acc:.4f}, "
                    + f"Test Loss: {client.fit_test_loss:.4f}"
                )
            else:
                logger.info(f"Client {client.cid}: Declined participation (qk={self.qk:.4f})")
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
        logger.debug(f"Client {client.cid}: Starting MaxFL evaluation")

        # Analyze if the client is selected
        client.selected = Utils.is_select_by_server(
            client.cid, config["selected_by_server"].split(",")
        )

        # Apply drivers and determine participation
        if client.selected:
            logger.debug(f"Client {client.cid}: Applying MaxFL QK driver")
            # Apply drivers to compute qk and get modifications
            modifications = self.apply_drivers(client, parameters=parameters, config=config)

            # Log qk if it was computed
            if "qk" in modifications:
                client.data_to_log["qk"] = modifications["qk"]
                self.qk = modifications["qk"]
                logger.info(f"Client {client.cid}: Computed qk = {self.qk:.4f}")

            # Check if client should participate based on qk threshold
            qk_threshold = self.config.maxfl_qk_threshold
            if hasattr(self, "qk") and self.qk < qk_threshold:
                logger.info(
                    f"Client {client.cid}: Will participate "
                    + f"(qk={self.qk:.4f} < threshold={qk_threshold})"
                )
                client.set_participating_state(True)
                client.set_parameters(parameters)
            else:
                logger.info(
                    f"Client {client.cid}: Will NOT participate "
                    + f"(qk={self.qk:.4f} >= threshold={qk_threshold})"
                )
                client.set_participating_state(False)
        else:
            logger.debug(f"Client {client.cid}: Using existing weights (not selected)")

        # Evaluate the model
        eval_results = client.model.evaluate(client.x_test, client.y_test, verbose=0)
        # eval_results = [loss, accuracy, precision, recall, auc]
        client.eval_loss = float(eval_results[0])
        client.eval_acc = float(eval_results[1])
        client.eval_precision = float(eval_results[2])
        client.eval_recall = float(eval_results[3])
        client.eval_auc = float(eval_results[4])

        # Check for invalid values
        import math

        if math.isnan(client.eval_loss) or math.isinf(client.eval_loss):
            logger.error(f"Client {client.cid}: INVALID LOSS VALUE: {client.eval_loss}")
        if math.isnan(client.eval_acc) or math.isinf(client.eval_acc):
            logger.error(f"Client {client.cid}: INVALID ACCURACY VALUE: {client.eval_acc}")

        eval_resp = {
            "cid": client.cid,
            "acc": client.eval_acc,
            "loss": client.eval_loss,
            "precision": client.eval_precision,
            "recall": client.eval_recall,
            "auc": client.eval_auc,
            "participating_state": client.get_participating_state(),
        }
        logger.info(
            f"Client {client.cid}: Evaluation completed - "
            + f"Acc: {client.eval_acc:.4f}, Loss: {client.eval_loss:.4f}, "
            + f"Precision: {client.eval_precision:.4f}, Recall: {client.eval_recall:.4f}, "
            + f"AUC: {client.eval_auc:.4f}, "
            + f"Participating: {client.get_participating_state()}"
        )

        return client.eval_loss, client.x_test.shape[0], eval_resp
