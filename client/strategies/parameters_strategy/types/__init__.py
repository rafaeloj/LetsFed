"""
Parameters Strategy Types Module.

This module contains different implementations of parameters sharing strategies.
"""

from .layerwise import LayerWiseParametersStrategy
from .normal import NormalParametersStrategy

__all__ = ["NormalParametersStrategy", "LayerWiseParametersStrategy"]
