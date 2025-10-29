"""
Factory for client selection methods.

This module implements the Factory pattern for creating client selection
strategies.
"""

from typing import Type

from .base import ClientSelectionMethod
from .structs import SelectionMethodConfig
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
    def create(cls, config: SelectionMethodConfig) -> ClientSelectionMethod:
        """
        Create a client selection method based on configuration.

        Args:
            config: SelectionMethodConfig configuration

        Returns:
            Instance of appropriate ClientSelectionMethod subclass

        Raises:
            ValueError: If selection method is not recognized
        """
        method = config.name.lower()

        if method not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown selection method: '{method}'. " + f"Available methods: {available}"
            )

        selection_class = cls._registry[method]
        return selection_class(config)

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
