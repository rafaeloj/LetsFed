# ============================================================================
# Aggregation Method Dataclasses
# ============================================================================


from dataclasses import dataclass, field
from typing import Any


@dataclass
class AggregationMethodConfig:
    """
    Base configuration for aggregation methods.

    Uses a generic params dict to allow different parameters per strategy.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FedAvgAggregationMethodConfig:
    """FedAvg aggregation method configuration - no extra parameters."""

    pass


@dataclass
class MaxFLAggregationMethodConfig:
    """MaxFL aggregation method configuration."""

    epsilon: float = 10.0
    learning_rate: float = 0.01


@dataclass
class QFFLAggregationMethodConfig:
    """QFFL aggregation method configuration."""

    rho: float = 0.1
    learning_rate: float = 0.01
