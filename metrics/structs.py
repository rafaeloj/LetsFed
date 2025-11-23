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

    def is_empty(self) -> bool:
        """
        Check if no metrics are configured.

        Returns:
            True if no metrics are configured, False otherwise
        """
        return len(self.metrics) == 0
