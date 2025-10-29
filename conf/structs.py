"""
Configuration dataclasses for the federated learning environment.

This module contains the main configuration structure that aggregates
all configuration aspects of the federated learning system.
"""

from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

# ============================================================================
# Aggregation Method Dataclasses
# ============================================================================


@dataclass
class AggregationMethodConfig:
    """Base class for aggregation methods."""

    name: str = MISSING


@dataclass
class FedAvgAggregationMethodConfig(AggregationMethodConfig):
    """FedAvg aggregation method configuration."""


@dataclass
class MaxFLAggregationMethodConfig(AggregationMethodConfig):
    """MaxFL aggregation method configuration."""

    epsilon: float = 10.0
    learning_rate: float = 0.01


@dataclass
class QFFLAggregationMethodConfig(AggregationMethodConfig):
    """QFFL aggregation method configuration."""

    rho: float = 0.1
    learning_rate: float = 0.01


# ============================================================================
# Selection Method Dataclasses
# ============================================================================


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


# ============================================================================
# Training Strategy Dataclasses
# ============================================================================


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


# ============================================================================
# Main Configuration Dataclasses
# ============================================================================


@dataclass
class ServerConfig:
    """Server configuration including network and FL parameters."""

    ip: str = MISSING
    port: int = MISSING

    aggregation_method: AggregationMethodConfig = MISSING
    selection_method: SelectionMethodConfig = MISSING


@dataclass
class ClientConfig:
    """Client configuration for training parameters."""

    epochs: int = MISSING
    learning_rate: float = MISSING
    training_strategy: TrainingStrategyConfig = MISSING

    # LetsFed specific
    participating: bool = True


@dataclass
class PartitionerConfig:
    """Dataset partitioner configuration."""

    method: str = MISSING  # 'dirichlet' or 'iid'

    # Dirichlet specific
    dirichlet_alpha: Optional[float] = None
    partition_by: str = "label"
    self_balancing: bool = False
    min_partition_size: int = 10
    shuffle: bool = True


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    dataset: str = MISSING
    path: str = MISSING
    train_partitioner: PartitionerConfig = MISSING
    test_partitioner: PartitionerConfig = MISSING

    # DataLoader parameters
    batch_size: int = 32
    shuffle_train: bool = True
    prefetch: bool = True


@dataclass
class ModelConfig:
    """Model configuration."""

    type: str = MISSING  # 'dnn' or 'cnn'
    path: str = MISSING


@dataclass
class Environment:
    """
    Main environment configuration that aggregates all aspects
    of the federated learning system.
    """

    # General parameters
    rounds: int = MISSING
    n_clients: int = MISSING
    init_clients: float = 1.0
    gpu: bool = False

    # Module configurations
    server: ServerConfig = MISSING
    client: ClientConfig = MISSING
    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING

    # Logging
    log_path: str = "logs"
