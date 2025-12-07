# ============================================================================
# Training Strategy Dataclasses
# ============================================================================


from dataclasses import MISSING, dataclass, field
from typing import Any


@dataclass
class TrainingStrategyConfig:
    """Base class for training strategies."""

    name: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalTrainingStrategyConfig:
    """Normal training strategy configuration."""

    pass


@dataclass
class LetsFedTrainingStrategyConfig:
    """LetsFed training strategy configuration."""

    threshold_accuracy: float = 1.0


@dataclass
class MaxFLTrainingStrategyConfig:
    """MaxFL training strategy configuration."""

    maxfl_qk_threshold: float = 0.5
    pre_training_epochs: int = 10


@dataclass
class FedPerTrainingStrategyConfig:
    """FedPer training strategy configuration."""

    pass


@dataclass
class QFFLTrainingStrategyConfig:
    """QFFL training strategy configuration."""

    eta: float = 1.0
