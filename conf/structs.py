"""
Configuration dataclasses for the federated learning environment.
"""

from dataclasses import dataclass

from omegaconf import MISSING

from ..client.strategies.structs import ClientConfig
from ..dataset_manager.structs import DatasetConfig
from ..model.structs import ModelConfig
from ..server.strategies.structs import ServerConfig


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
    seed: int = 42  # Global random seed for reproducibility

    # Module configurations
    server: ServerConfig = MISSING
    client: ClientConfig = MISSING
    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING

    # Logging
    log_path: str = "logs"
