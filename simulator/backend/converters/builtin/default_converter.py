#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from typing import TypeVar, Any
from simulator.backend.converters.registered_converter import RegisteredConverter

S = TypeVar("S")
T = TypeVar("T")


class DefaultConverter(RegisteredConverter):
    """Default factory for classes registered by CrossSim."""

    factory_type: T = object

    @classmethod
    def _create(
        cls: S,
        value_type: T,
        value: Any,
    ) -> T:
        """Create a new object of appropriate type given args and kwargs.

        Args:
            value_type: Type of object to create, should be a of type or subtype
                cls.converter_type
            value: Value to be given to value_type

        Returns:
            T: Object of type value_type created using associate with the given
                value.
        """
        if isinstance(value, dict):
            return value_type(**value)
        else:
            return value_type(value)
