#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from __future__ import annotations

import logging
from dataclasses import dataclass

from simulator.parameters.dac.dac import DACParameters

log = logging.getLogger(__name__)


@dataclass(repr=False)
class QuantizerDACParameters(DACParameters):
    """Parameters for the quantized digital-to-analog converter.

    Used to quantize the input signals that are passed to the array.

    Attributes:
        model (DACModel): name of the model used to specify quantization
            behavior. This must match the name of a child class of IDAC,
            other than "DAC"
        bits (int): bit resolution of the digital input
        input_bitslicing (bool): whether to bit slice the digital inputs
            to the MVM/VMM and accumulate the results from the different
            input bit slices using shift-and-add operations.
        sign_bit (bool): whether the digital input is encoded using
            sign-magnitude representation, with a range that is symmetric
            around zero
        slice_size (int): Default slice size for input bit slicing. Can
            be overridden from within the individual cores

    Raises:
        ValueError: Raised if input bitslicing is enabled with incompatible
            options.
    """

    model: str = "QuantizerDAC"

    def validate(self) -> None:
        """Checks the DAC parameter for invalid configurations."""
        super().validate()
        if self.input_bitslicing and self.bits == 0:
            raise ValueError("Cannot use input bit slicing if inputs are not quantized")


@dataclass(repr=False)
class SignMagnitudeDACParameters(DACParameters):
    """Parameters for the quantized digital-to-analog converter.

    Used to quantize the input signals that are passed to the array.

    Attributes:
        model (DACModel): name of the model used to specify quantization
            behavior. This must match the name of a child class of IDAC,
            other than "DAC"
        bits (int): bit resolution of the digital input
        input_bitslicing (bool): whether to bit slice the digital inputs
            to the MVM/VMM and accumulate the results from the different
            input bit slices using shift-and-add operations.
        sign_bit (bool): whether the digital input is encoded using
            sign-magnitude representation, with a range that is symmetric
            around zero
        slice_size (int): Default slice size for input bit slicing. Can
            be overridden from within the individual cores

    Raises:
        ValueError: Raised if input bitslicing is enabled with incompatible
            options.
    """

    model: str = "SignMagnitudeDAC"
