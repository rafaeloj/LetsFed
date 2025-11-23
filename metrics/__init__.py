"""
Metrics module for federated learning evaluation.

This module provides a comprehensive set of metrics for evaluating
federated learning models, including accuracy, precision, recall, F1-score, and AUC.
"""

from .base import Metric
from .factory import MetricFactory
from .manager import MetricsManager

__all__ = ["Metric", "MetricFactory", "MetricsManager"]
