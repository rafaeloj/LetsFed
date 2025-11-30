"""
Parameters Strategy Module for Federated Learning Client.

This module provides different strategies for sharing model parameters
between server and client, allowing for various personalization and
communication efficiency approaches.
"""

from .base import ParametersStrategy
from .factory import ParametersStrategyFactory
from .structs import ParametersStrategyConfig

__all__ = ["ParametersStrategy", "ParametersStrategyFactory", "ParametersStrategyConfig"]
