# ============================================================================
# Training Strategy Dataclasses
# ============================================================================


from dataclasses import MISSING, dataclass


@dataclass
class TrainingStrategyConfig:
    """Base class for training strategies."""

    name: str = MISSING
    learning_rate: float = 0.01


@dataclass
class NormalTrainingStrategyConfig(TrainingStrategyConfig):
    """Normal training strategy configuration."""


@dataclass
class LetsFedTrainingStrategyConfig(TrainingStrategyConfig):
    """LetsFed training strategy configuration."""

    threshold_accuracy: float = 1.0


@dataclass
class MaxFLTrainingStrategyConfig(TrainingStrategyConfig):
    """MaxFL training strategy configuration."""

    maxfl_qk_threshold: float = 0.5
    pre_training_epochs: int = 10


@dataclass
class FedPerTrainingStrategyConfig(TrainingStrategyConfig):
    """FedPer training strategy configuration."""


@dataclass
class QFFLTrainingStrategyConfig(TrainingStrategyConfig):
    """QFFL training strategy configuration."""

    eta: float = 0.1
