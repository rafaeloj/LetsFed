"""
Factory for client selection methods.

This module implements the Factory pattern for creating client selection
strategies.
"""

from typing import Type

from ....conf.structs import Environment
from .base import ClientSelectionMethod
from .types.deev import DEEV
from .types.letsfed import LetsFedSelection
from .types.poc import POC
from .types.random import RandomSelection
from .types.round_robin import RoundRobinSelection


class ClientSelectionFactory:
    """
    Factory for creating client selection strategies.

    Implements the Factory design pattern for instantiating different
    client selection methods based on configuration.
    """

    _registry: dict[str, Type[ClientSelectionMethod]] = {
        "random": RandomSelection,
        "deev": DEEV,
        "poc": POC,
        "round_robin": RoundRobinSelection,
        "letsfed": LetsFedSelection,
    }

    @classmethod
    def create(cls, config: Environment) -> ClientSelectionMethod:
        """
        Create a client selection method based on configuration.

        Args:
            config: Environment configuration

        Returns:
            Instance of appropriate ClientSelectionMethod subclass

        Raises:
            ValueError: If selection method is not recognized
        """
        method = config.server.selection_method.lower()

        if method not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown selection method: '{method}'. " + f"Available methods: {available}"
            )

        selection_class = cls._registry[method]
        return selection_class()

    @classmethod
    def register(cls, name: str, selection_class: Type[ClientSelectionMethod]) -> None:
        """
        Register a new client selection method.

        Args:
            name: Method name to register
            selection_class: Selection class to associate with the name

        Raises:
            ValueError: If name already registered or class is invalid
        """
        if name in cls._registry:
            raise ValueError(f"Selection method '{name}' is already registered")

        if not issubclass(selection_class, ClientSelectionMethod):
            raise ValueError(
                "Selection class must be a subclass of ClientSelectionMethod, "
                + f"got {selection_class}"
            )

        cls._registry[name] = selection_class

    @classmethod
    def get_available_methods(cls) -> list[str]:
        """Get list of available selection methods."""
        return list(cls._registry.keys())
