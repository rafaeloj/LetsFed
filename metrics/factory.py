"""
Factory for creating metric instances.
"""

from typing import Type

from .base import Metric
from .structs import MetricConfig
from .types import (
    AccuracyMetric,
    AUCMetric,
    F1ScoreMetric,
    FBetaScoreMetric,
    PrecisionMetric,
    RecallMetric,
)
from .types.structs import (
    AccuracyMetricConfig,
    AUCMetricConfig,
    F1ScoreMetricConfig,
    PrecisionMetricConfig,
    RecallMetricConfig,
)


class MetricFactory:
    """
    Factory class for creating metric instances.

    Provides a centralized way to create metrics by name with default or custom configurations.
    """

    _registry: dict[str, Type[Metric]] = {
        "accuracy": AccuracyMetric,
        "precision": PrecisionMetric,
        "recall": RecallMetric,
        "f1_score": F1ScoreMetric,
        "fbeta_score": FBetaScoreMetric,
        "auc": AUCMetric,
    }

    @classmethod
    def create(cls, metric_config: MetricConfig) -> Metric:
        """
        Create a metric instance from a MetricConfig.

        Args:
            metric_config: Configuration for the metric

        Returns:
            An instance of the requested metric

        Raises:
            ValueError: If the metric name is not recognized

        Examples:
            >>> from metrics.structs import PrecisionMetricConfig
            >>> config = PrecisionMetricConfig(average='weighted')
            >>> precision = MetricFactory.create_from_config(config)
        """
        metric_class = cls._registry.get(metric_config.name.lower())
        if metric_class is None:
            raise ValueError(
                f"Unknown metric: {metric_config.name}. "
                + f"Available metrics: {', '.join(cls._registry.keys())}"
            )

        metric_class_config = metric_class.params_from_json(metric_config.params)

        return metric_class(metric_class_config)

    @staticmethod
    def create_default_metrics() -> list[Metric]:
        """
        Create a set of default metrics commonly used in classification tasks.

        Returns:
            List of metric instances: [accuracy, precision, recall, f1_score, auc]
        """
        return [
            AccuracyMetric(AccuracyMetricConfig()),
            PrecisionMetric(PrecisionMetricConfig(average="macro", zero_division=0)),
            RecallMetric(RecallMetricConfig(average="macro", zero_division=0)),
            F1ScoreMetric(F1ScoreMetricConfig(average="macro", zero_division=0)),
            AUCMetric(AUCMetricConfig(multi_class="ovr", average="macro")),
        ]
