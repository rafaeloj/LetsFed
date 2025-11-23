"""
Base class for all metrics in the federated learning system.
"""

from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    """
    Abstract base class for all metrics.

    All metric implementations must inherit from this class and implement
    the calculate() method.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the metric.

        Args:
            name: The name of the metric (e.g., "accuracy", "precision")
        """
        self.name = name

    @abstractmethod
    def calculate(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray | None = None
    ) -> float:
        """
        Calculate the metric value.

        Args:
            y_true: True labels (ground truth)
            y_pred: Predicted labels (class predictions)
            y_pred_proba: Predicted probabilities (optional, needed for some metrics like AUC)

        Returns:
            The calculated metric value as a float
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __str__(self) -> str:
        return self.name
