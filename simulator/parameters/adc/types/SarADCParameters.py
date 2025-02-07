#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
"""Defines parameters related to the SarADC model."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from simulator.parameters.adc.adc import ADCParameters

log = logging.getLogger(__name__)


@dataclass(repr=False)
class SarADCParameters(ADCParameters):
    """SAR ADC specific non-ideality parameters.

    Attributes:
        gain_db (float): Open-loop gain in decibels of the operational
            amplifier at the output of the capacitive DAC (CDAC)
        sigma_capacitor (float): Standard deviation of the random variability
            in the minimum-sized capacitor in the CDAC, normalized by
            the minimum capacitance value.
        sigma_comparator (float): Standard deviation of the random variability
            in the input offset voltage of the comparator. The comparator
            compares the analog ADC input to the analog CDAC output during
            each SAR cycle. There is a comparator for every group of ADC
            inputs
        split_cdac (bool): Whether to use the split capacitor CDAC design
            to reduce the average size of the capacitor in the CDAC.
        group_size (int): Number of ADC inputs that share a SAR unit. Inputs
            within a group use the same CDAC and comparator. This corresponds
            to the number of grouped columns (MVM) or grouped rows (VMM)
            of the array.
    """

    model: str = "SarADC"
    gain_db: float = 100
    sigma_capacitor: float = 0.0
    sigma_comparator: float = 0.0
    split_cdac: bool = True
    group_size: int = 8
