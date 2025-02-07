#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from typing import TypeVar, Any

from simulator.parameters.base import RegisteredEnum
from simulator.backend.converters.registered_converter import RegisteredConverter

S = TypeVar("S")
T = TypeVar("T")


class EnumConverter(RegisteredConverter):
    """Converter for registered CrossSim enums."""

    converter_type: T = RegisteredEnum

    @classmethod
    def _create(
        cls: S,
        value_type: T,
        value: Any,
    ) -> T:
        """Create a new regsitered CrossSim enum value.

        Args:
            value_type: Type of object to create, should be a of type or subtype
                cls.converter_type
            value: Value to be given to value_type
            **context: Additional context that may be used when creating object,
                or when creating error messages.

        Returns:
            T: Object of type value_type created using associate with the given
                value.
        """
        # Syntax for getting by str enum vs. value of enum is different
        if isinstance(value, str):
            converted_value = value_type[value]
        else:
            converted_value = value_type(value)
        return converted_value

    @classmethod
    def create_error_msg(
        cls: S,
        value: Any,
        value_type: T,
    ) -> str:
        """Creates an error message when failing to create the expected type
        from the provided value.

        Args:
            value: Value provided from the user.
            value_type: Type of the value that is to be made.

        Returns:
            str: An error message with more context for the user.
        """
        error_msg = super().create_error_msg(
            value=value,
            value_type=value_type,
        )
        extra_info = f"Valid values are: {', '.join(value_type._member_names_)}"
        return "\n\n".join([error_msg, extra_info])
