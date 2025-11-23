"""
Configuration structures for metrics.
"""

from dataclasses import dataclass, field

from .types.structs import MetricConfig


@dataclass
class MetricsConfig:
    """
    Configuration for all metrics used in the federated learning client.

    This is a dictionary-like structure where each key is a metric name
    and each value is a MetricConfig instance.
    """

    metrics: dict[str, MetricConfig] = field(default_factory=dict)
    """Dictionary mapping metric names to their configurations"""

    def __post_init__(self) -> None:
        """Validate and convert metrics dictionary."""
        if self.metrics is None:
            self.metrics = {}

    def add_metric(self, metric_config: MetricConfig) -> None:
        """
        Add a metric configuration.

        Args:
            metric_config: Metric configuration to add
        """
        self.metrics[metric_config.name] = metric_config

    def get_metric(self, name: str) -> MetricConfig | None:
        """
        Get a metric configuration by name.

        Args:
            name: Name of the metric

        Returns:
            Metric configuration or None if not found
        """
        return self.metrics.get(name)

    def get_metric_names(self) -> list[str]:
        """
        Get list of all metric names.

        Returns:
            List of metric names
        """
        return list(self.metrics.keys())

    def is_empty(self) -> bool:
        """
        Check if no metrics are configured.

        Returns:
            True if no metrics are configured, False otherwise
        """
        return len(self.metrics) == 0
