"""
Precision metric implementation.
"""

import numpy as np
from sklearn.metrics import precision_score

from ..base import Metric
from .structs import PrecisionMetricConfig


class PrecisionMetric(Metric):
    """
    Precision metric for classification tasks.

    Precision = True Positives / (True Positives + False Positives)

    For multi-class problems, uses macro averaging (treats all classes equally).
    """

    def __init__(self, config: PrecisionMetricConfig) -> None:
        """
        Initialize the Precision metric.

        Args:
            config: Configuration for the Precision metric
        """
        super().__init__(name=config.name)
        self.average = config.average
        self.zero_division = config.zero_division

    def calculate(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray | None = None
    ) -> float:
        """
        Calculate precision score.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (not used for precision)

        Returns:
            Precision score as a float between 0 and 1
        """
        return float(
            precision_score(y_true, y_pred, average=self.average, zero_division=self.zero_division)
        )
