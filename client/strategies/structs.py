from dataclasses import MISSING, dataclass

from ...metrics.structs import MetricsConfig
from .training.structs import TrainingStrategyConfig


@dataclass
class ClientConfig:
    """Client configuration for training parameters."""

    epochs: int = MISSING
    training_strategy: TrainingStrategyConfig = MISSING
    metrics: MetricsConfig | None = None
