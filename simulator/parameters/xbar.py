#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from simulator.parameters.adc import PairedADCParameters
from simulator.parameters.dac.dac import PairedDACParameters
from simulator.parameters.base import BaseParameters
from simulator.parameters.base.dict_parameters import BaseDictParameters
from simulator.parameters.device import DeviceParameters


@dataclass(repr=False)
class GroupedXbarParameters(BaseDictParameters):
    """Class to hold multiple grouped xbar parameters.

    Attributes:
        parent (BaseParameters): Parent of this parameter in a nested parameter
            tree
        parents (list[BaseParameters]): List of all parameters directly above
            the current parameter. The first index of the list is the root
            parameter
        root (BaseParameters): Root of the parameter tree
    """

    dict_field_name: str = field(default="xbar", repr=False, init=False)
    xbar: dict[str, XbarParameters] = field(
        default_factory=lambda: {
            "default": XbarParameters(),
        },
    )

    def match(self, key: str) -> XbarParameters:
        """Returns the matching XbarParameters for a given key."""
        default = self.xbar.get("default", None)
        # Sort by most specific (amount of sections), descending
        xbar_keys = sorted(
            [k for k in self.xbar.keys() if k != "default"],
            key=lambda k: len(k.split(".")),
            reverse=True,
        )
        key_parts = key.split(".")
        matching_keys = []
        for xbar_key in xbar_keys:
            xkey_parts = xbar_key.split(".")
            if xkey_parts == key_parts[: len(xkey_parts)]:
                matching_keys.append(xbar_key)
        if len(matching_keys) == 1:
            return self._dict[matching_keys[0]]
        elif len(matching_keys) > 1:
            raise KeyError(
                f"Cannot get xbar parameters for '{key}' due to ambiguity. ",
                f"Xbar parameters exist for: {matching_keys}.",
            )
        if default is not None:
            return default
        raise KeyError(
            f"No key found matching {key} and no default provided. "
            f"Valid keys are {xbar_keys}",
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
        xbar_keys = sorted(
            self.xbar.keys(),
            key=lambda k: len(k.split(".")),
            reverse=True,
        )
        for xbar_key in xbar_keys:
            xkey_parts = xbar_key.split(".")
            if xkey_parts == key[: len(xkey_parts)]:
                return self._dict[xbar_key], key[len(xkey_parts) :]


@dataclass(repr=False)
class XbarParameters(BaseParameters):
    """Parameters that describe the behavior of the crossbar (xbar).

    Attributes:
        device: Parameters for the device used
        array:  Parameters for the array
        adc:  Parameters for the ADC
        dac:  Parameters for the DAC

    Raises:
        ValueError: Raised if per input bit slicing improperly configured
        ValueError: Raised if gate input mode used without input bit slicing
    """

    # Using field default factory to avoid isssues with declar
    device: DeviceParameters = None
    array: ArrayParameters = None
    adc: PairedADCParameters = None
    dac: PairedDACParameters = None

    def validate(self) -> None:
        """Checks the parameters for invalid settings."""
        super().validate()
        if (self.adc.mvm.adc_per_ibit and not self.dac.mvm.input_bitslicing) or (
            self.adc.vmm.adc_per_ibit and not self.dac.vmm.input_bitslicing
        ):
            raise ValueError(
                "ADC per input bit (adc_per_ibit) requires input bit slicing",
            )
        if self.array.parasitics.gate_input and (
            not self.dac.mvm.input_bitslicing or self.dac.mvm.slice_size > 1):
            raise ValueError("Gate input mode can only be used with input bit slicing")


@dataclass(repr=False)
class ArrayParameters(BaseParameters):
    """Parameters to desribe the behavior of the array.

    Attributes:
        Icol_max: Maximum current in a column, in units of the maximum current
            that can be drawn by a single device in the array. Any column
            current that exceeds (-Icol_max, +Icol_max) will be clipped to these
            bounds
        parasitics: Parameters for array parasitics
    """

    Icol_max: float = 0
    parasitics: ParasiticParameters = None


@dataclass(repr=False)
class ParasiticParameters(BaseParameters):
    """Parameters to describe behavior of parasitics.

    Attributes:
        enable: Whether to enable parasitic resistance model. For bit
            sliced, this indicates whether parasitics is enabled for ANY of the
            slices
        Rp_row: Parasitic resistance of the row metallic interconnects
        Rp_col: Parasitic resistince of the column metallic interconnects
        gate_input: If True, no parasitic voltage drops occur on the input side
            regardless of Rp value. This implements the configuration where the
            input row or column is connected to the gate of a transistor at
            every cell. Because the transistor behaves as a switch, the input
            signal must be binary. That means input bit slicing must be enabled
    """

    enable: bool = False
    Rp_row: float = 0
    Rp_col: float = 0
    gate_input: bool = False

    @property
    def Rp_row_norm(self) -> float:
        # Return the row parasitic resistance normalized by the minimum device resistance
        return self.Rp_row / self.parent.parent.device.Rmin

    @property
    def Rp_col_norm(self) -> float:
        # Return the column parasitic resistance normalized by the minimum device resistance
        return self.Rp_col / self.parent.parent.device.Rmin