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
class LetsFedAggregationMethodConfig:
    """LetsFed aggregation method configuration - no extra parameters."""

    pass


@dataclass
class MaxFLAggregationMethodConfig:
    """MaxFL aggregation method configuration."""

    epsilon: float = 0.1
    learning_rate: float = 1.0


@dataclass
class QFFLAggregationMethodConfig:
    """
    q-FFL aggregation method configuration.

    Args:
        q: Fairness parameter. Higher q gives more weight to clients with higher loss.
           q=0: uniform weighting (equivalent to FedAvg)
           q>0: fairness-aware weighting (clients with higher loss get more weight)
    """

    q: float = 0.5  # Fairness parameter (0 = FedAvg, >0 = fairness-aware)
