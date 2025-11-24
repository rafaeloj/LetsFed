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

    name: str = "normal"
    learning_rate: float = 0.01


@dataclass
class LetsFedTrainingStrategyConfig:
    """LetsFed training strategy configuration."""

    name: str = "lets_fed"
    learning_rate: float = 0.01
    threshold_accuracy: float = 1.0


@dataclass
class MaxFLTrainingStrategyConfig:
    """MaxFL training strategy configuration."""

    name: str = "maxfl"
    learning_rate: float = 0.01
    maxfl_qk_threshold: float = 0.5
    pre_training_epochs: int = 10


@dataclass
class FedPerTrainingStrategyConfig:
    """FedPer training strategy configuration."""

    name: str = "fed_per"
    learning_rate: float = 0.01


@dataclass
class QFFLTrainingStrategyConfig:
    """QFFL training strategy configuration."""

    name: str = "qffl"
    learning_rate: float = 0.01
    eta: float = 0.1
