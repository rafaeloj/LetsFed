"""
Recall metric implementation.
"""

import numpy as np
from sklearn.metrics import recall_score

from ..base import Metric
from .structs import RecallMetricConfig


class RecallMetric(Metric):
    """
    Recall metric for classification tasks.

    Recall = True Positives / (True Positives + False Negatives)

    For multi-class problems, uses macro averaging (treats all classes equally).
    """

    def __init__(self, config: RecallMetricConfig) -> None:
        """
        Initialize the Recall metric.

        Args:
            config: Configuration for the Recall metric
        """
        super().__init__(name=config.name)
        self.average = config.average
        self.zero_division = config.zero_division

    def calculate(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray | None = None
    ) -> float:
        """
        Calculate recall score.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (not used for recall)

        Returns:
            Recall score as a float between 0 and 1
        """
        return float(
            recall_score(y_true, y_pred, average=self.average, zero_division=self.zero_division)
        )
