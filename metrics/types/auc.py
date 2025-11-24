"""
AUC (Area Under the ROC Curve) metric implementation.
"""

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from ..base import Metric
from .structs import AUCMetricConfig


class AUCMetric(Metric):
    """
    AUC (Area Under the ROC Curve) metric for classification tasks.

    AUC measures the area under the Receiver Operating Characteristic curve.
    For multi-class problems, uses One-vs-Rest (OvR) strategy with macro averaging.

    Requires predicted probabilities (y_pred_proba) to calculate.
    """

    def __init__(self, config: AUCMetricConfig) -> None:
        """
        Initialize the AUC metric.

        Args:
            config: Configuration for the AUC metric
        """
        super().__init__(config)
        self.multi_class = config.multi_class
        self.average = config.average

    @staticmethod
    def params_from_json(params: dict[str, Any]) -> AUCMetricConfig:
        """
        Parse JSON parameters into AUCMetricConfig.

        Args:
            params: Dictionary of parameters from YAML configuration.

        Returns:
            AUCMetricConfig instance.
        """
        return AUCMetricConfig(
            multi_class=params.get("multi_class", "ovr"),
            average=params.get("average", "macro"),
        )

    def calculate(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray | None = None
    ) -> float:
        """
        Calculate AUC score.

        Args:
            y_true: True labels
            y_pred: Predicted labels (not used for AUC, here for compatibility)
            y_pred_proba: Predicted probabilities (REQUIRED for AUC calculation)

        Returns:
            AUC score as a float between 0 and 1, or 0.0 if calculation fails
        """
        if y_pred_proba is None:
            # Predicted probabilities required for AUC, return 0.0
            raise ValueError("y_pred_proba is required to calculate AUC.")

        try:
            # For multi-class classification
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 2:
                auc_score = roc_auc_score(
                    y_true, y_pred_proba, multi_class=self.multi_class, average=self.average
                )
            # For binary classification
            else:
                # If probabilities are 2D (n_samples, 2), use only positive class
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                auc_score = roc_auc_score(y_true, y_pred_proba)

            return float(auc_score)

        except Exception as exc:
            # AUC calculation failed, re-raise with exception chaining
            raise ValueError("Failed to calculate AUC.") from exc
