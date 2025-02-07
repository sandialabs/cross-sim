#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
"""Defines parameters related to generic ADCs."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from simulator.parameters.base import (
    BaseParameters,
    BasePairedParameters,
    RegisteredEnum,
)
from simulator.backend.registry import RegistryManager, register_subclasses

log = logging.getLogger(__name__)


class ADCRangeLimits(RegisteredEnum):
    """Defines how ADC range limits are used.

    "CALIBRATED" : ADC min and max are specified manually
    "MAX" : ADC limits are computed to cover the max possible range of ADC
        inputs, given the size of the array
    "GRANULAR" : ADC limits are computed so that the ADC level spacing is the
        minimum possible separation of two ADC inputs given the target
        resolution of weights. Assumes input bit slicing is used
        (with 1-bit DACs).
    """

    CALIBRATED = 1
    MAX = 2
    GRANULAR = 3


@register_subclasses
@dataclass(repr=False)
class ADCParameters(BaseParameters):
    """Parameters for the behavior of the analog-to-digital converter.

    Used to digitize the analog MVM/VMM outputs from the array.

    Attributes:
        model (str): name of the ADC model. This must match the name of
            a child class of IADC, other than "ADC"
        bits (int): bit resolution of the ADC digital output
        stochastic_rounding (bool): whether to probabilistically round
            an ADC input value to one of its two adjacent ADC levels,
            with a probability set by the distance to the level. If False,
            value is always rounded to the closer level.
        adc_per_ibit (bool): whether to digitize the MVM result of each
            input bit slice. This is only used if input_bitslicing = True
            in the associated DACParameters. If False, it is assumed by
            shift-and-add accumulation of input bits is done using analog
            peripheral circuits and only the final result is digitized.
        calibrated_range (list): the manually specified ADC min and max.
            This is only used if adc_range_option = ADCRangeLimits.CALIBRATED.
            If not using BITSLICED core, this must be a 1D array of length 2.
            If using BITSLICED core, this must be a 2D array with shape
            (num_slices, 2) that stores the ADC min/max for each bit
            slice of the core.
        adc_range_option (ADCRangeLimits): Which method is used to set
            ADC range limits

    Raises:
        ValueError: Raised if granular ADC is enabled with incompatible options
    """

    model: str = "IdealADC"
    bits: int = 0
    weight_bits: int = 0
    signed: bool = True
    stochastic_rounding: bool = False
    adc_per_ibit: bool = False
    calibrated_range: list = None
    adc_range_option: ADCRangeLimits = ADCRangeLimits.CALIBRATED

    def __new__(cls, *args, **kwargs):
        """Returns an unintialized instance of the class."""
        registry_manager = RegistryManager()
        key_name = "model"
        key_value = kwargs.get("model", cls.model)
        param_class = registry_manager.get_from_key(
            parent=ADCParameters,
            key_name=key_name,
            key_value=key_value,
        )
        param = super().__new__(param_class)
        return param

    def validate(self) -> None:
        """Validates the configuration of the ADC."""
        super().validate()
        if self.adc_range_option is ADCRangeLimits.GRANULAR and not self.adc_per_ibit:
            raise ValueError(
                "Granular ADC range is only supported for digital input "
                "shift and add (adc_per_ibit)",
            )


@dataclass(repr=False)
class PairedADCParameters(BasePairedParameters):
    """Pairs ADC parameters for MVM and VMM operations.

    Attributes:
        _match (bool): Whether to sync mvm and vmm parameters
        mvm (ADCParameters): ADC parameters for mvm operations
        vmm (ADCParameters): VMM parameters for vmm operations
    """

    _match: bool = True
    mvm: ADCParameters = None
    vmm: ADCParameters = None
