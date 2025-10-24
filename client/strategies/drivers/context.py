from dataclasses import dataclass, field
from typing import Union

# Type alias for values that can be stored in context
ContextValue = Union[int, float, str, bool, list, dict, None]


@dataclass
class DriverContext:
    """
    Context object that drivers can use to store their results.

    This provides a clean separation between driver logic and client state mutation,
    making drivers more testable and their side effects explicit.

    Example:
        context = DriverContext()
        driver.run(client, parameters, config, context)

        # Check what the driver modified
        if context.has('qk'):
            qk_value = context.get('qk')
    """

    modifications: dict[str, ContextValue] = field(default_factory=dict)

    def set(self, key: str, value: ContextValue) -> None:
        """
        Store a modification in the context.

        Args:
            key: The attribute name to modify
            value: The new value for the attribute
        """
        self.modifications[key] = value

    def get(self, key: str, default: ContextValue = None) -> ContextValue:
        """
        Retrieve a value from the context.

        Args:
            key: The attribute name to retrieve
            default: Default value if key doesn't exist

        Returns:
            The value associated with the key, or default if not found
        """
        return self.modifications.get(key, default)

    def has(self, key: str) -> bool:
        """
        Check if a key exists in the context.

        Args:
            key: The attribute name to check

        Returns:
            True if the key exists, False otherwise
        """
        return key in self.modifications

    def get_all(self) -> dict[str, ContextValue]:
        """
        Get all modifications as a dictionary.

        Returns:
            Dictionary of all modifications
        """
        return self.modifications.copy()

    def clear(self) -> None:
        """Clear all modifications from the context."""
        self.modifications.clear()

    def merge(self, other: "DriverContext") -> None:
        """
        Merge another context into this one.

        Args:
            other: Another DriverContext to merge
        """
        self.modifications.update(other.modifications)
