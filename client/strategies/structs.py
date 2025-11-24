from dataclasses import MISSING, dataclass

from ...metrics.structs import MetricConfig
from .training.structs import TrainingStrategyConfig


@dataclass
class ClientConfig:
    """Client configuration for training parameters."""

    epochs: int = MISSING
    learning_rate: float = MISSING  # Global learning rate used by ModelManager
    training_strategy: TrainingStrategyConfig = MISSING
    metrics: list[MetricConfig] | None = None
