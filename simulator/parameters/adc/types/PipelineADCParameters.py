#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
"""Defines parameters related to Pipeline and Cyclic ADC implementations."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from simulator.parameters.adc.adc import ADCParameters

log = logging.getLogger(__name__)


@dataclass(repr=False)
class PipelineADCParameters(ADCParameters):
    """Pipeline ADC specific non-ideality parameters.

    Attributes:
        gain_db (float): Open-loop gain in decibels of the operational
            amplifier used as the residue amplifier in a 1.5-bit stage
            of the pipeline ADC.
        sigma_C1 (float): Standard deviation of the random variability
            in capacitor C1 used to amplify the voltage by 2X in the
            1.5-bit switched-capacitor stage. The amplification factor
            is (1 + C1/C2), where C1 = C2 in the ideal case.
        sigma_C2 (float): Standard deviation of the random variability
            in capacitor C2 in the 1.5-bit ADC stage, normalized by the
            nominal value of C2.
        sigma_Cpar (float): Standard deviation of the random variability
            in the parasitic capacitance at the negative input of the
            operational amplifier in he 1.5-bit stage, normalized by the
            nominal value of C1.
        sigma_comparator (float): Standard deviation of the random
            variability in the input offsetvoltage of the comparators used
            in the 1.5-bit stages.
        group_size (int): Number of ADC inputs that share single pipeline
            ADC and its random capacitor mismatches and comparator offsets.
            This corresponds to the number of grouped columns (MVM) or
            grouped rows (VMM) of the array.
    """

    model: str = "PipelineADC"
    gain_db: float = 100
    sigma_C1: float = 0.0
    sigma_C2: float = 0.0
    sigma_Cpar: float = 0.0
    sigma_comparator: float = 0.0
    group_size: int = 8
