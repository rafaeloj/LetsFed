# ============================================================================
# Aggregation Method Dataclasses
# ============================================================================


from dataclasses import MISSING, dataclass


@dataclass
class AggregationMethodConfig:
    """Base class for aggregation methods."""

    name: str = MISSING


@dataclass
class FedAvgAggregationMethodConfig(AggregationMethodConfig):
    """FedAvg aggregation method configuration."""


@dataclass
class MaxFLAggregationMethodConfig(AggregationMethodConfig):
    """MaxFL aggregation method configuration."""

    epsilon: float = 10.0
    learning_rate: float = 0.01


@dataclass
class QFFLAggregationMethodConfig(AggregationMethodConfig):
    """QFFL aggregation method configuration."""

    rho: float = 0.1
    learning_rate: float = 0.01
