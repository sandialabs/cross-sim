#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
"""Defines parameters related to an ideal ADC."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from simulator.parameters.adc.adc import ADCParameters

log = logging.getLogger(__name__)


@dataclass(repr=False)
class IdealADCParameters(ADCParameters):
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
