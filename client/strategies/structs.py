from dataclasses import MISSING, dataclass

from .training.structs import TrainingStrategyConfig


@dataclass
class ClientConfig:
    """Client configuration for training parameters."""

    epochs: int = MISSING
    training_strategy: TrainingStrategyConfig = MISSING
