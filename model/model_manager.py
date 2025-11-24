import os
from typing import List

import keras
import tensorflow as tf

from ..conf.structs import Environment
from ..utils.logger import Logger

logger = Logger(__name__)


class ModelManager:
    def __init__(self, conf: Environment, input_shape: List[int], path: str = "") -> None:
        self.conf = conf
        self.input_shape = input_shape
        self.model_path = f"/{conf.model.path}/{conf.model.type}.weights.h5"
        if path:
            self.model_path = f"{path}/{conf.model.path}/{conf.model.type}.weights.h5"

        logger.info(
            f"ModelManager initialized for model type '{conf.model.type}' "
            + f"with input shape {input_shape}"
        )

        # Initialize and load model
        self.model = self.get_model()
        self.load_model()

    def get_model(self) -> keras.Model:
        """Return model based on configuration."""
        logger.debug(f"Building model of type '{self.conf.model.type}'")
        if self.conf.model.type == "dnn":
            return self.dnn()
        if self.conf.model.type == "cnn":
            return self.cnn()

    def load_model(self) -> None:
        """Load model weights."""
        if os.path.exists(self.model_path):
            logger.info(f"Loading model weights from {self.model_path}")
            self.model.load_weights(self.model_path)
        else:
            logger.warning(
                f"No existing weights found at {self.model_path}, using initialized weights"
            )

    def save_model(self) -> None:
        """Save model weights."""
        logger.info(f"Saving model weights to {self.model_path}")
        self.model.save_weights(self.model_path)

    def dnn(self) -> keras.Model:
        """Define a simple DNN model with proper regularization."""
        logger.debug("Building DNN model architecture")

        # Use L2 regularization to prevent overfitting
        l2_reg = tf.keras.regularizers.l2(0.001)

        model: keras.Model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=self.input_shape[1:]),
                tf.keras.layers.Flatten(),
                # First hidden layer with batch normalization
                tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=l2_reg),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                # Second hidden layer
                tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=l2_reg),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                # Output layer
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.conf.client.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info(f"DNN model built with {model.count_params()} parameters")
        return model

    def cnn(self) -> keras.Model:
        """Define a simple CNN model."""
        logger.debug("Building CNN model architecture")
        model: keras.Model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32, (3, 3), activation="relu", input_shape=self.input_shape[1:]
                ),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.conf.client.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info(f"CNN model built with {model.count_params()} parameters")
        return model
