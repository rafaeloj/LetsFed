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
class AggregationMethod:
    """Base class for aggregation methods."""

    name: str = MISSING


@dataclass
class FedAvgAggregation(AggregationMethod):
    """FedAvg aggregation method configuration."""

    name: str = "fedavg"


@dataclass
class MaxFLAggregation(AggregationMethod):
    """MaxFL aggregation method configuration."""

    name: str = "maxfl"
    epsilon: float = 10.0
    learning_rate: float = 0.01
    pre_training_epochs: int = 5


@dataclass
class QFFLAggregation(AggregationMethod):
    """QFFL aggregation method configuration."""

    name: str = "qffl"
    rho: float = 0.1
    learning_rate: float = 0.01


# ============================================================================
# Selection Method Dataclasses
# ============================================================================


@dataclass
class SelectionMethod:
    """Base class for selection methods."""

    name: str = MISSING
    perc_of_clients: float = 0.3


@dataclass
class RandomSelection(SelectionMethod):
    """Random selection method configuration."""

    name: str = "random"


@dataclass
class DeevSelection(SelectionMethod):
    """DEEV selection method configuration."""

    name: str = "deev"
    decay: float = 0.95


@dataclass
class PoCSelection(SelectionMethod):
    """PoC selection method configuration."""

    name: str = "poc"


@dataclass
class RoundRobinSelection(SelectionMethod):
    """Round Robin selection method configuration."""

    name: str = "round_robin"


@dataclass
class LetsFedSelection(SelectionMethod):
    """LetsFed selection method configuration."""

    name: str = "letsfed"
    participating_method: str = "random"
    non_participating_method: str = "poc"


# ============================================================================
# Training Strategy Dataclasses
# ============================================================================


@dataclass
class TrainingStrategy:
    """Base class for training strategies."""

    name: str = MISSING


@dataclass
class NormalTraining(TrainingStrategy):
    """Normal training strategy configuration."""

    name: str = "normal"


@dataclass
class LetsFedTraining(TrainingStrategy):
    """LetsFed training strategy configuration."""

    name: str = "letsfed"
    threshold: float = 1.0


@dataclass
class MaxFLTraining(TrainingStrategy):
    """MaxFL training strategy configuration."""

    name: str = "maxfl"


@dataclass
class FedPerTraining(TrainingStrategy):
    """FedPer training strategy configuration."""

    name: str = "fedper"


@dataclass
class QFFLTraining(TrainingStrategy):
    """QFFL training strategy configuration."""

    name: str = "qffl"


# ============================================================================
# Main Configuration Dataclasses
# ============================================================================


@dataclass
class ServerConfig:
    """Server configuration including network and FL parameters."""

    ip: str = MISSING
    port: int = MISSING

    aggregation_method: AggregationMethod = MISSING
    selection_method: SelectionMethod = MISSING


@dataclass
class ClientConfig:
    """Client configuration for training parameters."""

    epochs: int = MISSING
    learning_rate: float = MISSING
    training_strategy: TrainingStrategy = MISSING

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
