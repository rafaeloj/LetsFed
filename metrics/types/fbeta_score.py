"""
F-Beta Score metric implementation.
"""

import numpy as np
from sklearn.metrics import fbeta_score

from ..base import Metric
from .structs import FBetaScoreMetricConfig


class FBetaScoreMetric(Metric):
    """
    F-Beta Score metric for classification tasks.

    F-Beta = (1 + beta²) * (Precision * Recall) / ((beta² * Precision) + Recall)

    The F-beta score is the weighted harmonic mean of precision and recall,
    reaching its optimal value at 1 and its worst value at 0.

    The beta parameter determines the weight of recall in the combined score:
    - beta < 1: more weight to precision
    - beta = 1: F1 score (equal weight)
    - beta > 1: more weight to recall
    - beta = 2: F2 score (recall weighted twice as much as precision)
    """

    def __init__(self, config: FBetaScoreMetricConfig) -> None:
        """
        Initialize the F-Beta Score metric.

        Args:
            config: Configuration for the F-Beta Score metric
        """
        super().__init__(name=config)
        self.beta = config.beta
        self.average = config.average
        self.zero_division = config.zero_division

    def calculate(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray | None = None
    ) -> float:
        """
        Calculate F-Beta score.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (not used for F-Beta)

        Returns:
            F-Beta score as a float between 0 and 1
        """
        return float(
            fbeta_score(
                y_true,
                y_pred,
                beta=self.beta,
                average=self.average,
                zero_division=self.zero_division,
            )
        )
