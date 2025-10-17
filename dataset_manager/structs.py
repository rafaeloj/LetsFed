"""
Structs for dataset management.
"""

from dataclasses import dataclass
from typing import Tuple

from numpy.typing import NDArray


@dataclass
class DataPartition:
    """
    Represents a data partition for a federated learning client.

    Contains training, validation, and test splits.
    """

    train_x: NDArray
    train_y: NDArray
    val_x: NDArray
    val_y: NDArray
    test_x: NDArray
    test_y: NDArray

    @property
    def train_size(self) -> int:
        """Number of training samples."""
        return len(self.train_x)

    @property
    def val_size(self) -> int:
        """Number of validation samples."""
        return len(self.val_x)

    @property
    def test_size(self) -> int:
        """Number of test samples."""
        return len(self.test_x)

    def get_train_data(self) -> Tuple[NDArray, NDArray]:
        """Get training data."""
        return self.train_x, self.train_y

    def get_val_data(self) -> Tuple[NDArray, NDArray]:
        """Get validation data."""
        return self.val_x, self.val_y

    def get_test_data(self) -> Tuple[NDArray, NDArray]:
        """Get test data."""
        return self.test_x, self.test_y
