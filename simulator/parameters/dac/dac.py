#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
"""Defines parameters related to generic DACs."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from simulator.parameters.base import (
    BaseParameters,
    BasePairedParameters,
)
from simulator.backend.registry import RegistryManager, register_subclasses
from simulator.parameters.mapping import MappingParameters

log = logging.getLogger(__name__)


@dataclass(repr=False)
class PairedDACParameters(BasePairedParameters):
    """Pairs DAC parameters for MVM and VMM operations.

    Attributes:
        _match: Whether to sync mvm and vmm parameters
        mvm: DAC parameters for mvm operations
        vmm: VMM parameters for vmm operations
    """

    _match: bool = True
    mvm: DACParameters = None
    vmm: DACParameters = None


@register_subclasses
@dataclass(repr=False)
class DACParameters(BaseParameters):
    """Parameters for the digital-to-analog converter.

    Used to quantize the input signals that are passed to the array.

    Attributes:
        model: name of the model used to specify quantization behavior. This
            must match the name of a child class of IDAC, other than "DAC"
        bits: bit resolution of the digital input
        input_bitslicing: whether to bit slice the digital inputs to the MVM/VMM
            and accumulate the results from the different input bit slices using
            shift-and-add operations.
        sign_bit: whether the digital input is encoded using sign-magnitude
            representation, with a range that is symmetric around zero
        slice_size: Default slice size for input bit slicing. Can be overridden
        from within the individual cores

    Raises:
        ValueError: Raised if input bitslicing is enabled with incompatible
            options
    """

    model: str = "IdealDAC"
    bits: int = 0
    input_bitslicing: bool = False
    signed: bool = True
    slice_size: int = 1
    mapping: MappingParameters = None

    def __new__(cls, *args, **kwargs):
        """Returns an unintialized instance of the class."""
        registry_manager = RegistryManager()
        key_name = "model"
        key_value = kwargs.get("model", cls.model)
        param_class = registry_manager.get_from_key(
            parent=DACParameters,
            key_name=key_name,
            key_value=key_value,
        )
        param = super().__new__(param_class)
        return param

    @property
    def sign_bit(self) -> bool:
        """Returns true if a sign bit is necessary."""
        return self.min < 0
