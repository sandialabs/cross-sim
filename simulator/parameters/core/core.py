#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from __future__ import annotations

import logging

from typing import Any
from dataclasses import dataclass, field

from simulator.backend.registry import register_subclasses, RegistryManager
from simulator.parameters.base import (
    BaseDictParameters,
    BaseParameters,
    RegisteredEnum,
)
from simulator.parameters.utils import interpret_key

log = logging.getLogger(__name__)


class ConvertingConfiguration(RegisteredEnum):
    """Describes the scheme for behavior of the ADC and DAC of a core.

    "SKIP_CONVERSION": ADC/DAC ownership is not defined at the current
        level of the core
    "SHARED_PER_CHILD": ADC/DAC is shared between all child cores.
    "UNIQUE_PER_CHILD": ADC/DAC is unique between all child cores.
    """

    SKIP_CONVERSION = 0
    SHARED_PER_CHILD = 1
    UNIQUE_PER_CHILD = 2


@dataclass(repr=False)
class SubcoreParameters(BaseDictParameters):
    """Class to hold multiple subcore parameters."""

    dict_field_name: str = field(default="subcores", repr=False, init=False)
    subcores: dict[str, CoreParameters] = field(
        default_factory=dict,
    )

    def _get(self, key: list[str]) -> tuple[Any, list[str]]:
        """Called internally by get().

        This function is responsible for taking a key and interpretting the key
        for the respective parameter then fetching the appropriate value at that
        key.

        This logic is split so that the error handling and recursive logic can
        be shared between params, while subclasses are free to define their own
        behavior on how to interpret a key, which is needed especially in dict
        params.

        Args:
            key: List of key parts to get fetch value with.

        Returns:
            Tuple[Any, list]: Value at the first matching result from the key,
                and a list of the key parts remaining
        """
        key0 = key[0]
        if key0 not in self._dict:
            key0 = interpret_key(key0)
        return self._dict[key0], key[1:]

    def pop_key(self, key: str) -> Any:
        """Pops and returns a value at a given key.

        Args:
            key: Key to pop value from.

        Returns:
            Any: Value at the specified key.
        """
        try:
            return self._dict.pop(key)
        except KeyError:
            k = interpret_key(key)
            return self._dict.pop(k)


@register_subclasses
@dataclass(repr=False)
class CoreParameters(BaseParameters):
    """Describes the configuration of a generic core.

    Args:
        core_type: Type of core the parameter is for.
        mapping: Mapping parameters for the core.
        subcores: Contains parameters for child cores.
        adc_scheme: Scheme used to describe ADC initialization.
        dac_scheme: Scheme used to describe DAC initialization.
    """

    core_type: str = "ICore"
    clipping: bool = True
    subcores: SubcoreParameters = None
    adc_scheme: ConvertingConfiguration = ConvertingConfiguration.SKIP_CONVERSION
    dac_scheme: ConvertingConfiguration = ConvertingConfiguration.SKIP_CONVERSION

    def __new__(cls, *args, **kwargs):
        """Returns an unintialized instance of the class."""
        registry_manager = RegistryManager()
        key_name = "core_type"
        key_value = kwargs.get("core_type", cls.core_type)
        param_class = registry_manager.get_from_key(
            parent=CoreParameters,
            key_name=key_name,
            key_value=key_value,
        )
        param = super().__new__(param_class)
        return param
