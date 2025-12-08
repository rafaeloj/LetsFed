"""
Centralized data loading for federated learning.

This module provides a unified interface for loading and preprocessing
federated learning datasets, eliminating code duplication between
client and server implementations.

Uses TensorFlow's tf.data.Dataset for native Keras compatibility.
"""

from typing import TYPE_CHECKING, Callable, Optional, Tuple

import numpy as np
import tensorflow as tf

from ..utils.logger import Logger
from .dataset_manager import DSManager
from .transforms import TransformPipeline

if TYPE_CHECKING:
    from ..conf.structs import Environment

logger = Logger(__name__)


class FederatedDataLoader:
    """
    Data Loader for Federated Learning using TensorFlow's tf.data.Dataset.

    This class encapsulates all data loading and preprocessing logic, providing
    a single source of truth for data preparation in both clients and server.
    Uses tf.data.Dataset for native Keras compatibility and optimal performance.

    Responsibilities:
    - Load data partitions via DSManager
    - Apply transformations (normalization, reshaping)
    - Create tf.data.Dataset pipelines with batching, shuffling, prefetching
    - Provide direct numpy array access for backward compatibility
    - Maintain metadata (labels, shapes)

    Examples:
        >>> # Create dataloader for a client partition
        >>> dataloader = FederatedDataLoader(partition_id=0, conf=config)
        >>>
        >>> # Get tf.data.Dataset for training (for use with model.fit)
        >>> train_ds = dataloader.get_train_dataset(batch_size=32, shuffle=True)
        >>> model.fit(train_ds, epochs=10)
        >>>
        >>> # Get validation dataset (for use with model.evaluate)
        >>> val_ds = dataloader.get_validation_dataset(batch_size=32)
        >>> metrics = model.evaluate(val_ds)
        >>>
        >>> # Get test dataset
        >>> test_ds = dataloader.get_test_dataset(batch_size=32)
        >>> predictions = model.predict(test_ds)
        >>>
        >>> # Backward compatibility: get raw numpy arrays
        >>> x_train, y_train = dataloader.get_train_data()
        >>> x_val, y_val = dataloader.get_validation_data()
        >>> x_test, y_test = dataloader.get_test_data()
    """

    def __init__(
        self,
        partition_id: int,
        conf: "Environment",
    ) -> None:
        """
        Initialize the federated dataloader.

        Args:
            partition_id: ID of the data partition to load (0 for server, 0..N-1 for clients)
            conf: Environment configuration containing dataset and model settings
        """
        logger.info(f"Initializing FederatedDataLoader for partition {partition_id}")

        self.partition_id = partition_id
        self.conf = conf

        # Data arrays (populated by _load_and_preprocess)
        self.x_train: np.ndarray
        self.y_train: np.ndarray
        self.x_validation: np.ndarray
        self.y_validation: np.ndarray
        self.x_test: np.ndarray
        self.y_test: np.ndarray

        # Metadata
        self.label_names: list[str]

        # Load and preprocess data
        self._load_and_preprocess()

        logger.info(
            "FederatedDataLoader initialized - "
            + f"Train: {len(self.x_train)} samples, "
            + f"Val: {len(self.x_validation)} samples, "
            + f"Test: {len(self.x_test)} samples"
        )

    def _load_and_preprocess(self) -> None:
        """
        Load data from disk and apply preprocessing transformations.

        This method:
        1. Uses DSManager to load the partition
        2. Extracts features and labels
        3. Applies normalization (via TransformPipeline)
        4. Reshapes for CNN if needed (via TransformPipeline)
        5. Stores metadata
        """
        logger.debug(f"Loading partition {self.partition_id} from disk")

        # Step 1: Load data via DSManager
        dm = DSManager(n_clients=self.conf.n_clients, conf=self.conf.dataset, seed=self.conf.seed)
        train, validation, test = dm.load_locally(partition_id=self.partition_id)

        # Step 2: Extract features and labels
        keys = list(test.features.keys())
        x_train_raw, y_train_raw = train[keys[0]], train[keys[1]]
        x_val_raw, y_val_raw = validation[keys[0]], validation[keys[1]]
        x_test_raw, y_test_raw = test[keys[0]], test[keys[1]]

        logger.debug(
            "Raw data loaded - "
            + f"Train: {x_train_raw.shape}, "
            + f"Val: {x_val_raw.shape}, "
            + f"Test: {x_test_raw.shape}"
        )

        # Step 3: Apply normalization
        # Use tanh normalization ([-1, 1]) as it works better with gradient descent
        # and modern weight initialization methods (Xavier/He)
        normalization_method = getattr(self.conf.dataset, "normalization_method", "tanh")

        self.x_train = TransformPipeline.normalize_images(x_train_raw, method=normalization_method)
        self.x_validation = TransformPipeline.normalize_images(
            x_val_raw, method=normalization_method
        )
        self.x_test = TransformPipeline.normalize_images(x_test_raw, method=normalization_method)

        # Step 4: Add channel dimension for CNN models
        # Conv2D layers expect 4D input: (batch, height, width, channels)
        if self.conf.model.type == "cnn":
            logger.debug("Model is CNN - adding channel dimension if needed")
            self.x_train = TransformPipeline.add_channel_dimension(self.x_train)
            self.x_validation = TransformPipeline.add_channel_dimension(self.x_validation)
            self.x_test = TransformPipeline.add_channel_dimension(self.x_test)

        # Step 5: Store labels (no transformation needed)
        self.y_train = y_train_raw
        self.y_validation = y_val_raw
        self.y_test = y_test_raw

        # Step 6: Store metadata
        self.label_names = test.features["label"].names

        logger.info(
            "Data preprocessing completed - "
            + f"Train shape: {self.x_train.shape}, "
            + f"Val shape: {self.x_validation.shape}, "
            + f"Test shape: {self.x_test.shape}, "
            + f"Normalization: {normalization_method}, "
            + f"Labels: {len(self.label_names)} classes"
        )

    def get_train_dataset(
        self,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        prefetch: bool = True,
        transform: Optional[Callable] = None,
    ) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset for training data.

        Args:
            batch_size: Batch size (defaults to conf.dataset.batch_size)
            shuffle: Whether to shuffle data (defaults to conf.dataset.shuffle_train)
            prefetch: Whether to enable prefetching for performance (default: True)
            transform: Optional transformation function to apply to batches

        Returns:
            tf.data.Dataset ready for use with model.fit()

        Example:
            >>> train_ds = dataloader.get_train_dataset(batch_size=32, shuffle=True)
            >>> model.fit(train_ds, epochs=10, validation_data=val_ds)
        """
        if batch_size is None:
            batch_size = self.conf.dataset.batch_size
        if shuffle is None:
            shuffle = self.conf.dataset.shuffle_train

        # Create dataset from numpy arrays
        dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))

        # Shuffle if requested
        if shuffle:
            # Use buffer size = dataset size for full shuffle
            # Set seed for reproducibility
            dataset = dataset.shuffle(
                buffer_size=len(self.x_train), seed=self.conf.seed, reshuffle_each_iteration=True
            )

        # Batch the data
        dataset = dataset.batch(batch_size, drop_remainder=False)

        # Apply custom transform if provided
        if transform is not None:
            dataset = dataset.map(transform, num_parallel_calls=tf.data.AUTOTUNE)

        # Prefetch for performance
        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        logger.debug(
            f"Created train dataset - Samples: {len(self.x_train)}, "
            + f"Batch size: {batch_size}, Shuffle: {shuffle}"
        )

        return dataset

    def get_validation_dataset(
        self,
        batch_size: Optional[int] = None,
        prefetch: bool = True,
        transform: Optional[Callable] = None,
    ) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset for validation data.

        Args:
            batch_size: Batch size (defaults to conf.dataset.batch_size)
            prefetch: Whether to enable prefetching for performance (default: True)
            transform: Optional transformation function to apply to batches

        Returns:
            tf.data.Dataset ready for use with model.evaluate()

        Example:
            >>> val_ds = dataloader.get_validation_dataset(batch_size=32)
            >>> metrics = model.evaluate(val_ds)
        """
        if batch_size is None:
            batch_size = self.conf.dataset.batch_size

        # Create dataset from numpy arrays
        dataset = tf.data.Dataset.from_tensor_slices((self.x_validation, self.y_validation))

        # Batch the data (never shuffle validation)
        dataset = dataset.batch(batch_size, drop_remainder=False)

        # Apply custom transform if provided
        if transform is not None:
            dataset = dataset.map(transform, num_parallel_calls=tf.data.AUTOTUNE)

        # Prefetch for performance
        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        logger.debug(
            f"Created validation dataset - Samples: {len(self.x_validation)}, "
            + f"Batch size: {batch_size}"
        )

        return dataset

    def get_test_dataset(
        self,
        batch_size: Optional[int] = None,
        prefetch: bool = True,
        transform: Optional[Callable] = None,
    ) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset for test data.

        Args:
            batch_size: Batch size (defaults to conf.dataset.batch_size)
            prefetch: Whether to enable prefetching for performance (default: True)
            transform: Optional transformation function to apply to batches

        Returns:
            tf.data.Dataset ready for use with model.predict() or model.evaluate()

        Example:
            >>> test_ds = dataloader.get_test_dataset(batch_size=32)
            >>> predictions = model.predict(test_ds)
        """
        if batch_size is None:
            batch_size = self.conf.dataset.batch_size

        # Create dataset from numpy arrays
        dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))

        # Batch the data (never shuffle test)
        dataset = dataset.batch(batch_size, drop_remainder=False)

        # Apply custom transform if provided
        if transform is not None:
            dataset = dataset.map(transform, num_parallel_calls=tf.data.AUTOTUNE)

        # Prefetch for performance
        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        logger.debug(
            f"Created test dataset - Samples: {len(self.x_test)}, " + f"Batch size: {batch_size}"
        )

        return dataset


# ============================================================================
# Factory Functions for Creating Datasets
# ============================================================================


def create_federated_datasets(
    partition_id: int,
    conf: "Environment",
    batch_size: Optional[int] = None,
    shuffle_train: Optional[bool] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Create train, validation, and test tf.data.Dataset objects for a partition.

    This is a convenience factory function that creates all three datasets
    with appropriate settings for federated learning.

    Args:
        partition_id: ID of the data partition to load
        conf: Environment configuration
        batch_size: Batch size for datasets (defaults to conf.dataset.batch_size)
        shuffle_train: Whether to shuffle training data (defaults to conf.dataset.shuffle_train)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)

    Example:
        >>> train_ds, val_ds, test_ds = create_federated_datasets(
        >>>     partition_id=0,
        >>>     conf=config,
        >>>     batch_size=32
        >>> )
        >>> model.fit(train_ds, epochs=10, validation_data=val_ds)
        >>> model.evaluate(test_ds)
    """
    # Create dataloader
    dataloader = FederatedDataLoader(partition_id=partition_id, conf=conf)

    # Create datasets
    train_dataset = dataloader.get_train_dataset(
        batch_size=batch_size,
        shuffle=shuffle_train,
    )

    val_dataset = dataloader.get_validation_dataset(
        batch_size=batch_size,
    )

    test_dataset = dataloader.get_test_dataset(
        batch_size=batch_size,
    )

    logger.info(
        f"Created federated datasets for partition {partition_id} - "
        + f"Batch size: {batch_size or conf.dataset.batch_size}"
    )

    return train_dataset, val_dataset, test_dataset
