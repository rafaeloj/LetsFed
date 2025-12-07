# ============================================================================
# Selection Method Dataclasses
# ============================================================================


from dataclasses import MISSING, dataclass, field
from typing import Any


@dataclass
class SelectionMethodConfig:
    """
    Base configuration for selection methods.

    Uses a generic params dict to allow different parameters per strategy.
    """

    name: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RandomSelectionMethodConfig:
    """Random selection method configuration."""

    perc_of_clients: float = 0.5


@dataclass
class DeevSelectionMethodConfig:
    """DEEV selection method configuration."""

    decay: float = 0.05


@dataclass
class PoCSelectionMethodConfig:
    """PoC selection method configuration."""

    perc_of_clients: float = 0.5


@dataclass
class RoundRobinSelectionMethodConfig:
    """Round Robin selection method configuration."""

    perc_of_clients: float = 0.5
    n_clients: int = 5


@dataclass
class LetsFedSelectionMethodConfig:
    """LetsFed selection method configuration."""

    participating_method: SelectionMethodConfig
    non_participating_method: SelectionMethodConfig
