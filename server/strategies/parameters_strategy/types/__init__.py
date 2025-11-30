"""
Parameters Strategy Types Module (Server).

This module contains different implementations of parameters sharing strategies
for the server side.
"""

from .layerwise import LayerWiseParametersStrategy
from .normal import NormalParametersStrategy

__all__ = ["NormalParametersStrategy", "LayerWiseParametersStrategy"]
