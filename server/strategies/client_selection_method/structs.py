# ============================================================================
# Selection Method Dataclasses
# ============================================================================


from dataclasses import MISSING, dataclass


@dataclass
class SelectionMethodConfig:
    """Base class for selection methods."""

    name: str = MISSING


@dataclass
class RandomSelectionMethodConfig(SelectionMethodConfig):
    """Random selection method configuration."""

    perc_of_clients: float = 0.3


@dataclass
class DeevSelectionMethodConfig(SelectionMethodConfig):
    """DEEV selection method configuration."""

    decay: float = 0.95


@dataclass
class PoCSelectionMethodConfig(SelectionMethodConfig):
    """PoC selection method configuration."""

    perc_of_clients: float = 0.3


@dataclass
class RoundRobinSelectionMethodConfig(SelectionMethodConfig):
    """Round Robin selection method configuration."""

    perc_of_clients: float = 0.3
    n_clients: int = 10


@dataclass
class LetsFedSelectionMethodConfig(SelectionMethodConfig):
    """LetsFed selection method configuration."""

    participating_selection_method: SelectionMethodConfig = MISSING
    non_participating_selection_method: SelectionMethodConfig = MISSING
