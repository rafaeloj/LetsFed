# ============================================================================
# Parameters Strategy Dataclasses (Server)
# ============================================================================

from dataclasses import MISSING, dataclass, field
from typing import Any


@dataclass
class ParametersStrategyConfig:
    """Base class for parameters strategies."""

    name: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalParametersStrategyConfig:
    """Normal parameters strategy configuration - shares all parameters."""

    pass  # No additional configuration needed


@dataclass
class LayerWiseParametersStrategyConfig:
    """Layer-wise parameters strategy configuration - shares only first K layers."""

    num_shared_layers: int = 5  # Number of layers to share with clients
