#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from typing import TypeVar, Any

from simulator.parameters.base.dict_parameters import BaseDictParameters
from .parameters_converter import ParametersConverter

S = TypeVar("S")
T = TypeVar("T")


class DictParametersConverter(ParametersConverter):
    """Converter for CrossSim parameters."""

    converter_type: T = BaseDictParameters

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
        if not isinstance(value, dict):
            pass
        elif len(value) == 1 and value_type.dict_field_name in value:
            pass
        else:
            field_value_type = value_type._field_value_type()
            # Convert each subkey
            value = {
                k: ParametersConverter._create(value_type=field_value_type, value=v)
                for k, v in value.items()
            }
            # Nest dict so that default constructer accepts the dict
            value = {
                value_type.dict_field_name: value,
            }
        return super()._create(value_type=value_type, value=value)
