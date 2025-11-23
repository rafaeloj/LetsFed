"""
Accuracy metric implementation.
"""

import numpy as np
from sklearn.metrics import accuracy_score

from ..base import Metric
from .structs import AccuracyMetricConfig


class AccuracyMetric(Metric):
    """
    Accuracy metric for classification tasks.

    Accuracy = (Number of correct predictions) / (Total number of predictions)
    """

    def __init__(self, config: AccuracyMetricConfig) -> None:
        """Initialize the Accuracy metric."""
        super().__init__(name=config)

    def calculate(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray | None = None
    ) -> float:
        """
        Calculate accuracy score.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (not used for accuracy)

        Returns:
            Accuracy score as a float between 0 and 1
        """
        return float(accuracy_score(y_true, y_pred))
