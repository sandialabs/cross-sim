#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from typing import Any, get_type_hints, TypeVar

from simulator.backend.registry import RegistryManager
from .parameters import BaseParameters

T = TypeVar("T")


def resolve_registered_and_builtin_types(t: str | type) -> type:
    """Resolves the type of an input.

    Args:
        t: The type to resolve. If t is already a type, it is returned as is.
            If t is a string, an attempt will be made to fetch a type by the
            same name will be fetched from the RegistryManager, or by looking
            at built in types.

    Returns:
        type: The resolved type from the input.
    """
    if isinstance(t, type):
        return t
    if not isinstance(t, str):
        raise TypeError("resolve_type expects either a string or a type.")

    registry_manager = RegistryManager()
    dummy_type = type("_", (), {"__annotations__": {"type": t}})
    localns = {}
    for k, v in registry_manager.items():
        localns.update(**{**v, k.__name__: k})
    try:
        resolved_type = get_type_hints(dummy_type, localns=localns)["type"]
    except NameError as e:
        raise NameError(
            f"Could not resolve the type '{e.name}', it is not registered or a builtin",
        ) from e
    return resolved_type


def convert_type(
    value_type: T,
    value: Any,
    key_name: str,
    parent: BaseParameters,
) -> T:
    """Attempts to convert a value to a specified type.

    Args:
        value_type: Type to convert value to
        value: Value to convert
        key_name: Name of key that is being converted
        parent: Parent parameter where the value is being converted.

    Raises:
        ValueError: Raised if a value couldn't be converted.

    Returns:
        T: Returns an object of specified type.
    """
    registry_manager = RegistryManager()
    value_type = registry_manager.get_from_root(value_type)
    try:
        value = registry_manager.convert(value_type, value)
        if isinstance(value, BaseParameters):
            value._parent = parent
        return value
    except (ValueError, KeyError) as e:
        if not isinstance(parent, BaseParameters):
            key = key_name
        else:
            # Helps the user find where they incorrectly configured parameters
            if parent.root is parent:
                key = f"{key_name}"
            else:
                key = f"{parent.get_path_from_root()}.{key_name}"
        conversion_error_msg = "\n".join(e.args)
        error_msg = (
            f"Could not convert key '{key}' to type '{value_type.__name__}' "
            f"using '{key} = {value}'.\n"
            f"Reason:\n\n"
            f"{conversion_error_msg}"
        )
        raise ValueError(error_msg) from e
