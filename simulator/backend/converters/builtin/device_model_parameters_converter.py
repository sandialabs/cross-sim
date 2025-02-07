#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from typing import TypeVar, Any

from simulator.backend.registry import RegistryManager
from simulator.parameters.device import DeviceModelParameters
from .parameters_converter import ParametersConverter

S = TypeVar("S")
T = TypeVar("T")


class DeviceModelParametersConverter(ParametersConverter):
    """Converter for CrossSim parameters."""

    converter_type: T = DeviceModelParameters

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
        registry_manager = RegistryManager()
        key_name = "model"
        if value is None:
            key_value = "IdealDevice"
        else:
            key_value = value["model"]
        param_class = registry_manager.get_from_key(
            parent=value_type,
            key_name=key_name,
            key_value=key_value,
        )
        return super()._create(value_type=param_class, value=value)
