#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
"""Backend functions for tracking subclasses.

Types used in parameters need to be cast from strings to the appropriate
data structure for CrossSim to work. Because types can have arbitrary levels
of inheritance, or user-defined types may be used there is no way hard-code
the importing of the appropriate type. Class registration provides a central
point to track and load these types.
"""
from __future__ import annotations

import logging
from typing import TypeVar, ItemsView, KeysView, Any

log = logging.getLogger(__name__)

T = TypeVar("T")


def register_subclasses(parent_class):
    """Registers any subclasses of a parent class with CrossSim.

    Any classes not registered cannot be loaded into parameters from
    string or JSON formats. Adding this decorator to a base class allows
    subclasses to be tracked by the CrossSim type registry manager.
    """
    prior_init_subclass = parent_class.__init_subclass__
    registry_manager = RegistryManager()
    registry_manager.register(parent_class)
    subclass_registry = registry_manager._registry.get(parent_class)

    def __init_subclass__(cls, *args, **kwargs):
        # Add subclass to registry
        subclass_registry[cls.__name__] = cls
        # If subclass is registered, get those too
        subclass_registry.update(registry_manager.get(cls))
        prior_init_subclass(*args, **kwargs)

    parent_class.__init_subclass__ = classmethod(__init_subclass__)
    return parent_class


class RegistryManager:
    """Manages tracking subclasses of CrossSim base types."""

    _instance: RegistryManager
    _registry: dict[T, dict[str, T]]

    def __new__(cls, *args, **kwargs):
        """Creates or return a registry manager singleton.

        If a registry manager object already exists, that instance is returned.

        Args:
            *args: Positional arguments for the RegistryManager initialization
            **kwargs: Keyword arguments for the RegistryManager initialization

        Returns:
            RegistryManager: Backend singleton
        """
        if not hasattr(cls, "_instance"):
            # Initialize a new instance if one does not already exist
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initializes the registry manager."""
        if not hasattr(self, "_registry"):
            self._registry = {}

    def register(self, key: T):
        """Registers a base class to have subclasses tracked.

        Args:
            key (T): Base class to track.
        """
        self._registry.setdefault(key, {})

    def get(self, key: T | str) -> dict[str, T]:
        """Returns a copy of the subclass registry for the specified base type.

        If key is not present an empty dictionary is returned.

        Args:
            key (T): Base class to get subclass registry for

        Returns:
            dict[str, T]: Subclass registry dictionary in the form
                {subcls.__name__: subcls}
        """
        if isinstance(key, str):
            registered_classes = {k.__name__: k for k in self._registry.keys()}
            key = registered_classes.get(key, None)
        return self._registry.get(key, {}).copy()

    def get_from_root(self, key: T | str) -> T:
        """Returns the associated type, looking through the entire registry.

        Args:
            key: Type to look for, either as the type itself or its name
                as a string

        Raises:
            KeyError: Raised if key is not in present in the registry.

        Returns:
            type: The type requested
        """
        for parent, subclass_registry in self._registry.items():
            parent_name = str(parent.__name__)
            if key in [parent_name, parent]:
                return parent
            for subclass_name, subclass in subclass_registry.items():
                if key in [subclass_name, subclass]:
                    return subclass
        raise KeyError(f"Registry does not contain {key}")

    def get_from_key(self, parent: T, key_name: str, key_value: str) -> T | None:
        """Gets a child class matching a specified key.

        Args:
            parent: Parent class to get child class of.
            key_name: Name of the attribute that holds the key value.
            key_value: Value of th key to match

        Returns:
            T: Child class with matching key. Parent is returned if no subclass
                has a key that matches the value provided.
        """
        registry = self.get(parent)
        mapping = {getattr(subcls, key_name): subcls for subcls in registry.values()}
        class_type = None
        if key_value in mapping:
            class_type = mapping[key_value]
            log.debug(
                f"Selected subclass '{class_type}' of {parent} "
                f"(matches {key_name}={key_value})",
            )
        else:
            valid_keys = list(mapping.keys())
            class_type = parent
            log.warning(
                f"No {class_type.__name__} found with {key_name}={key_value}. "
                f"Was {key_value} and its parameter class imported? "
                f"Valid keys are: {valid_keys}"
                f"Falling back to {class_type.__name__}, initialization may fail.",
            )
        return class_type

    def items(self) -> ItemsView[T, dict[str, T]]:
        """Returns a dict_view with (key, value) pairs of the registry.

        In thef from (parent_class, subclass_registry).

        Returns:
            ItemsView[T, dict[str, T]]: dict_view with (key, value) pairs
                in the form (parent_class, subclass_registry).
        """
        return self._registry.items()

    def keys(self) -> KeysView[T]:
        """Returns a dict_view with keys of the parent classes.

        Returns:
            KeysView[T]]: dict_view with keys of the parent classes.
        """
        return self._registry.keys()

    def convert(self, type_: T, value: Any) -> T:
        """Converts a value to a registered CrossSim type.

        The converter chosen for the value is the first converter in the type's
        mro that match's the converter's type.

        Args:
            type_: Type of object to convert to.
            value: Value to convert

        Returns:
            T: Converted value of type specified.
        """
        # Delayed import to prevent circular dependencies
        # This will register all the built in converters
        from simulator.backend.converters.builtin import RegisteredConverter

        converters = self.get(RegisteredConverter)
        converter = converters.pop("DefaultConverter")
        candidates = {v.converter_type: v for v in converters.values()}
        for parent in type_.mro():
            if parent not in candidates:
                continue
            converter = candidates[parent]
            break
        converted_value = converter.create(value_type=type_, value=value)
        return converted_value

    def __contains__(self, key: Any) -> bool:
        """Returns true if provided key is in the registry.

        Args:
            key: Key to check in registry

        Returns:
            bool: True if key is in registry. False otherwise.
        """
        try:
            _ = self.get_from_root(key=key)
            return True
        except KeyError:
            return False
