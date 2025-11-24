from dataclasses import dataclass

from ..structs import MetricConfig


@dataclass
class AccuracyMetricConfig(MetricConfig):
    """Configuration for Accuracy metric."""

    name: str = "accuracy"


@dataclass
class PrecisionMetricConfig(MetricConfig):
    """Configuration for Precision metric."""

    name: str = "precision"
    average: str = "macro"
    """Averaging strategy: 'macro', 'micro', 'weighted', etc."""
    zero_division: int = 0
    """Value to return when there is a zero division"""


@dataclass
class RecallMetricConfig(MetricConfig):
    """Configuration for Recall metric."""

    name: str = "recall"
    average: str = "macro"
    """Averaging strategy: 'macro', 'micro', 'weighted', etc."""
    zero_division: int = 0
    """Value to return when there is a zero division"""


@dataclass
class F1ScoreMetricConfig(MetricConfig):
    """Configuration for F1 Score metric."""

    name: str = "f1_score"
    average: str = "macro"
    """Averaging strategy: 'macro', 'micro', 'weighted', etc."""
    zero_division: int = 0
    """Value to return when there is a zero division"""


@dataclass
class FBetaScoreMetricConfig(MetricConfig):
    """Configuration for F-Beta Score metric."""

    name: str = "fbeta_score"
    beta: float = 1.0
    """Beta parameter (beta=1 gives F1, beta=2 gives F2, etc.)"""
    average: str = "macro"
    """Averaging strategy: 'macro', 'micro', 'weighted', etc."""
    zero_division: int = 0
    """Value to return when there is a zero division"""


@dataclass
class AUCMetricConfig(MetricConfig):
    """Configuration for AUC metric."""

    name: str = "auc"
    multi_class: str = "ovr"
    """Strategy for multi-class: 'ovr' (one-vs-rest) or 'ovo' (one-vs-one)"""
    average: str = "macro"
    """Averaging strategy: 'macro', 'weighted', etc."""
