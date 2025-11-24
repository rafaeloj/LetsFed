"""
Manager for calculating multiple metrics at once.
"""

import numpy as np

from .base import Metric
from .factory import MetricFactory
from .structs import MetricConfig


class MetricsManager:
    """
    Manager class for handling multiple metrics.

    This class provides a pipeline for calculating multiple metrics
    on prediction results in a single call.
    """

    def __init__(self, metrics_config: list[MetricConfig] | None = None) -> None:
        """
        Initialize the MetricsManager.

        Args:
            metrics_config: Configuration for metrics (list of MetricConfig).
                          If None or empty, uses default metrics.
        """
        if metrics_config is None or len(metrics_config) == 0:
            # Use default metrics configuration
            self.metrics = MetricFactory.create_default_metrics()
        else:
            # Create metric instances from configuration
            # metrics_config is a list[MetricConfig]
            self.metrics = [MetricFactory.create(metric_config) for metric_config in metrics_config]

    def add_metric(self, metric: Metric) -> None:
        """
        Add a metric to the manager.

        Args:
            metric: Metric instance to add
        """
        self.metrics.append(metric)

    def remove_metric(self, metric_name: str) -> None:
        """
        Remove a metric from the manager by name.

        Args:
            metric_name: Name of the metric to remove
        """
        self.metrics = [m for m in self.metrics if m.name != metric_name]

    def get_metrics_names(self) -> list[str]:
        """
        Get the names of all metrics in the manager.

        Returns:
            List of metric names
        """
        return [m.name for m in self.metrics]

    def calculate_all(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray | None = None
    ) -> dict[str, float]:
        """
        Calculate all metrics in the pipeline.

        Args:
            y_true: True labels (ground truth)
            y_pred: Predicted labels (class predictions)
            y_pred_proba: Predicted probabilities (optional, needed for AUC)

        Returns:
            Dictionary mapping metric names to their calculated values

        Examples:
            >>> manager = MetricsManager()
            >>> y_true = np.array([0, 1, 2, 0, 1, 2])
            >>> y_pred = np.array([0, 1, 2, 0, 2, 1])
            >>> y_pred_proba = np.array([[0.8, 0.1, 0.1], ...])
            >>> results = manager.calculate_all(y_true, y_pred, y_pred_proba)
            >>> print(results)
            {'accuracy': 0.666, 'precision': 0.666, 'recall': 0.666, ...}
        """
        results = {}
        for metric in self.metrics:
            try:
                value = metric.calculate(y_true, y_pred, y_pred_proba)
                results[metric.name] = value
            except Exception as e:
                # If a metric fails, log it and continue with others
                print(f"Warning: Failed to calculate {metric.name}: {e}")
                results[metric.name] = 0.0

        return results

    def calculate_metrics(self, y_true: np.ndarray, y_pred_probs: np.ndarray) -> dict[str, float]:
        """
        Calculate all metrics from predicted probabilities.

        This is a convenience method that converts probabilities to class predictions
        and then calls calculate_all().

        Args:
            y_true: True labels (ground truth)
            y_pred_probs: Predicted probabilities (shape: [n_samples, n_classes])

        Returns:
            Dictionary mapping metric names to their calculated values

        Examples:
            >>> manager = MetricsManager()
            >>> y_true = np.array([0, 1, 2, 0, 1, 2])
            >>> y_pred_probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], ...])
            >>> results = manager.calculate_metrics(y_true, y_pred_probs)
            >>> print(results)
            {'accuracy': 0.666, 'precision': 0.666, 'recall': 0.666, 'f1_score': 0.666, 'auc': 0.85}
        """
        # Convert probabilities to class predictions
        y_pred = np.argmax(y_pred_probs, axis=1)
        # Call calculate_all with both predictions and probabilities
        return self.calculate_all(y_true, y_pred, y_pred_probs)

    def __repr__(self) -> str:
        metric_names = [m.name for m in self.metrics]
        return f"MetricsManager(metrics={metric_names})"

    def __str__(self) -> str:
        return f"MetricsManager with {len(self.metrics)} metrics: {[m.name for m in self.metrics]}"
