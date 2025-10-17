"""
Configuration package for the Federated Learning framework.

This package provides structured configuration management using dataclasses.
"""

from .loader import load_config, save_config
from .structs import (
    ClientConfig,
    DatasetConfig,
    Environment,
    ModelConfig,
    PartitionerConfig,
    ServerConfig,
)

__all__ = [
    "Environment",
    "ServerConfig",
    "ClientConfig",
    "DatasetConfig",
    "ModelConfig",
    "PartitionerConfig",
    "load_config",
    "save_config",
]
