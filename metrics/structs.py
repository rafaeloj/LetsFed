"""
Configuration structures for metrics.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricConfig:
    """
    Base configuration for a single metric.

    All metric configurations should use this class or extend it.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
