#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from typing import TypeVar, Any
from dataclasses import is_dataclass

from simulator.parameters.base import BaseParameters
from simulator.backend.converters.registered_converter import RegisteredConverter

S = TypeVar("S")
T = TypeVar("T")


class ParametersConverter(RegisteredConverter):
    """Converter for CrossSim parameters."""

    converter_type: T = BaseParameters

    @classmethod
    def _create(
        cls: S,
        value_type: T,
        value: Any,
    ) -> T:
        """Create a new parameter of appropriate type given a value.

        Args:
            value_type: Type of object to create, should be a of type or subtype
                cls.converter_type
            value: Value to be given to value_type

        Returns:
            T: Object of type value_type created using associate with the given
                value.
        """
        # If value is already a param, exit early
        if isinstance(value, value_type):
            return value
        elif isinstance(value, str):
            return value_type.from_json(filepath=value)

        _implicitly_initialized = False
        if issubclass(value_type, value.__class__):
            # value_type is a child class of value's class
            # Try a best attempt conversion
            value = value.as_dict()
        elif value is None:
            value = {}
            _implicitly_initialized = True

        value = value_type(**value)
        value._implicitly_initialized = _implicitly_initialized
        return value

    @classmethod
    def _is_subclass_of_converter_type(cls: S, value_type: T):
        msg = (
            f"Cannot use {cls.__name__} to create {value_type.__name__}, "
            f"{value_type.__name__} is not a subclass of {cls.converter_type.__name__}"
        )
        is_dataclass_class = is_dataclass(value_type) and isinstance(value_type, type)
        is_subclass = issubclass(value_type, cls.converter_type)
        if not is_dataclass_class or not is_subclass:
            raise TypeError(msg)
