"""
Base class for all metrics in the federated learning system.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .structs import MetricConfig


class Metric(ABC):
    """
    Abstract base class for all metrics.

    All metric implementations must inherit from this class and implement
    the calculate() method.
    """

    def __init__(self, config: MetricConfig) -> None:
        """
        Initialize the metric.

        Args:
            config: The configuration for the metric.
        """
        self.name = config.name

    @staticmethod
    @abstractmethod
    def params_from_json(params: dict[str, Any]) -> Any:  # noqa: ANN401
        """
        Parse JSON parameters into the strategy-specific config.

        Args:
            params: Dictionary of parameters from YAML configuration.

        Returns:
            Strategy-specific configuration dataclass instance.
        """
        ...

    @abstractmethod
    def calculate(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray | None = None
    ) -> float:
        """
        Calculate the metric value.

        Args:
            y_true: True labels (ground truth)
            y_pred: Predicted labels (class predictions)
            y_pred_proba: Predicted probabilities (optional, needed for some metrics like AUC)

        Returns:
            The calculated metric value as a float
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __str__(self) -> str:
        return self.name
