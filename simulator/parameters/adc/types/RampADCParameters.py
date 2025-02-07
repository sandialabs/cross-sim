#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
"""Defines parameters related to the RampADC model."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from simulator.parameters.adc.adc import ADCParameters

log = logging.getLogger(__name__)


@dataclass(repr=False)
class RampADCParameters(ADCParameters):
    """Ramp ADC specific non-ideality parameters.

    Attributes:
        gain_db (float): Open-loop gain in decibels of the operational
            amplifier at the output of the capacitive DAC (CDAC) used to
            generate the voltage ramp
        sigma_capacitor (float): Standard deviation of the random variability
            in the minimum-sized capacitor in the CDAC, normalized by the
            minimum capacitance value.
        sigma_comparator (float): Standard deviation of the random variability
            in the input offset voltage of the comparator used for ramp
            comparison. There is a comparator associated with every array
            column (MVM) and/or row (VMM). The offset is normalized by
            the reference voltage used for the ramp.
        symmetric_cdac (bool): Whether to use the symmetric CDAC design
            that treats the ADC levels as two's complement signed integers.
            If False, uses an alternative CDAC design that treats the ADC
            levels as unsigned integers.
    """

    model: str = "RampADC"
    gain_db: float = 100
    sigma_capacitor: float = 0.0
    sigma_comparator: float = 0.0
    symmetric_cdac: bool = True
