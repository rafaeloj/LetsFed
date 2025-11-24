"""
F1 Score metric implementation.
"""

from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from ..base import Metric
from .structs import F1ScoreMetricConfig


class F1ScoreMetric(Metric):
    """
    F1 Score metric for classification tasks.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    F1 score is the harmonic mean of precision and recall.
    For multi-class problems, uses macro averaging (treats all classes equally).
    """

    def __init__(self, config: F1ScoreMetricConfig) -> None:
        """
        Initialize the F1 Score metric.

        Args:
            config: Configuration for the F1 Score metric
        """
        super().__init__(config)
        self.average = config.average
        self.zero_division = config.zero_division

    @staticmethod
    def params_from_json(params: dict[str, Any]) -> F1ScoreMetricConfig:
        """
        Parse JSON parameters into F1ScoreMetricConfig.

        Args:
            params: Dictionary of parameters from YAML configuration.

        Returns:
            F1ScoreMetricConfig instance.
        """
        return F1ScoreMetricConfig(
            average=params.get("average", "macro"), zero_division=params.get("zero_division", 0)
        )

    def calculate(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray | None = None
    ) -> float:
        """
        Calculate F1 score.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (not used for F1)

        Returns:
            F1 score as a float between 0 and 1
        """
        return float(
            f1_score(y_true, y_pred, average=self.average, zero_division=self.zero_division)
        )
