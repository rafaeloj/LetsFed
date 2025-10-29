from dataclasses import MISSING, dataclass
from typing import Optional


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

    # Dataset parameters
    dataset: str = MISSING
    path: str = MISSING
    train_partitioner: PartitionerConfig = MISSING
    test_partitioner: PartitionerConfig = MISSING

    # DataLoader parameters
    batch_size: int = 32
    shuffle_train: bool = True
    prefetch: bool = True
