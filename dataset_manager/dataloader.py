"""
DataLoader module for efficient data loading and preprocessing.

This module provides DataLoader classes for training and evaluation
with support for batching, shuffling, and preprocessing.
"""

from typing import Callable, Iterator, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class DataLoader:
    """
    DataLoader for batch processing of datasets.

    Supports batching, shuffling, and custom preprocessing functions.
    """

    def __init__(
        self,
        x: NDArray,
        y: NDArray,
        batch_size: int = 32,
        shuffle: bool = False,
        preprocess_fn: Optional[Callable] = None,
    ) -> None:
        """
        Initialize DataLoader.

        Args:
            x: Input features array
            y: Target labels array
            batch_size: Size of each batch
            shuffle: Whether to shuffle data each epoch
            preprocess_fn: Optional preprocessing function to apply to each batch
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.preprocess_fn = preprocess_fn

        self.n_samples = len(x)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
        self.indices = np.arange(self.n_samples)

    def __len__(self) -> int:
        """Return number of batches."""
        return self.n_batches

    def __iter__(self) -> Iterator[Tuple[NDArray, NDArray]]:
        """
        Iterate over batches.

        Yields:
            Tuple of (batch_x, batch_y)
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = self.indices[i : i + self.batch_size]
            batch_x = self.x[batch_indices]
            batch_y = self.y[batch_indices]

            if self.preprocess_fn is not None:
                batch_x, batch_y = self.preprocess_fn(batch_x, batch_y)

            yield batch_x, batch_y

    def get_full_data(self) -> Tuple[NDArray, NDArray]:
        """
        Get full dataset without batching.

        Returns:
            Tuple of (x, y)
        """
        if self.preprocess_fn is not None:
            return self.preprocess_fn(self.x, self.y)
        return self.x, self.y


def normalize_images(x: NDArray, y: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Normalize image data to [0, 1] range.

    Args:
        x: Image array
        y: Labels array

    Returns:
        Normalized (x, y) tuple
    """
    x_normalized = x.astype("float32") / 255.0
    return x_normalized, y


def augment_images(x: NDArray, y: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Apply data augmentation to images.

    Simple augmentation including random flips.

    Args:
        x: Image array
        y: Labels array

    Returns:
        Augmented (x, y) tuple
    """
    # Random horizontal flip
    flip_mask = np.random.random(len(x)) > 0.5
    x[flip_mask] = np.flip(x[flip_mask], axis=2)

    return x, y


def create_preprocessing_fn(
    normalize: bool = True,
    augment: bool = False,
) -> Optional[Callable]:
    """
    Create a preprocessing function with specified operations.

    Args:
        normalize: Whether to normalize images
        augment: Whether to apply data augmentation

    Returns:
        Preprocessing function or None
    """

    def preprocess(x: NDArray, y: NDArray) -> Tuple[NDArray, NDArray]:
        if normalize:
            x, y = normalize_images(x, y)
        if augment:
            x, y = augment_images(x, y)
        return x, y

    if normalize or augment:
        return preprocess
    return None
