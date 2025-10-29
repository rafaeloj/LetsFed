from dataclasses import MISSING, dataclass


@dataclass
class ModelConfig:
    """Model configuration."""

    type: str = MISSING  # 'dnn' or 'cnn'
    path: str = MISSING
