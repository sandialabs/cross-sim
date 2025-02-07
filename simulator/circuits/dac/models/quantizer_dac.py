#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
import logging
from typing import Iterator

import numpy as np
import numpy.typing as npt

from simulator.circuits.dac.idac import IDAC, _InputBitslice
from simulator.backend.compute import ComputeBackend

from simulator.parameters.dac import DACParameters
from simulator.parameters.mapping import MappingParameters

log = logging.getLogger(__name__)
xp: np = ComputeBackend()  # Represents either cupy or numpy


class QuantizerDAC(IDAC):
    """Digital-to-Analog converter that quantizes unsigned values."""

    def __init__(
        self,
        dac_params: DACParameters,
    ):
        """Initializes a quantizer DAC.

        Args:
            dac_params: Parameters for the DAC

        Raises:
            ValueError: Raised on invalid configurations.
        """
        super().__init__(dac_params)
        self.levels = 2**self.bits
        self.min = self.input_mapping_params.min
        self.max = self.input_mapping_params.max

        # Calculate levels
        if not self.signed and (self.min < 0 or self.max < 0) and self.bits > 0:
            raise ValueError("Quantizer DAC: sign bit disabled but limits are negative")

    def set_limits(self, matrix: npt.ArrayLike):
        """Sets the limits of the DAC.

        May depend on the input matrix but not required.
        """
        pass

    def convert(self, vector: npt.ArrayLike) -> npt.NDArray:
        """Converts a vector from digital value to analog values.

        Returns a vector converted from one containing digital values
        to one containing analog values.
        Converts with respect to the limits set for the dac.
        """
        # If bits is None, just return the input
        if not self.bits or self.bits == 0:
            return vector

        # Clip vector
        input_ = vector.clip(self.min, self.max)

        # Quantize
        # set qmult (quantization multiplier):  multiply by this factor to
        # convert every level to an absolute range of 1
        # The -1 is because the first level is 0, i.e. 2 bits define 3 segments
        # between 0 and 3
        qmult = (self.levels - 1) / (self.max - self.min)

        # do quantization using rounding
        input_ -= self.min
        input_ *= qmult
        input_ = xp.rint(input_, out=input_)
        input_ /= qmult
        input_ += self.min

        return input_

    def convert_sliced(
        self,
        vector: npt.ArrayLike,
        slice_size=None,
    ) -> Iterator[_InputBitslice]:
        """Returns an iterator that converts slices of digital values to analog.

        Converts with respect to the limits set for the dac.
        """
        if slice_size:
            self.slice_size = slice_size

        if self.signed:
            raise ValueError(
                "QuantizerDAC does not support input bit slicing with signed inputs: "
                "use SignMagnitudeDAC",
            )

        # First, convert the inputs to integers from 0 to 2^n-1
        magbits = self.bits
        x_int = xp.rint(vector * (pow(2, magbits) - 1))

        total_sliced = 0
        global_correction = pow(2, magbits) / (pow(2, magbits) - 1)
        while total_sliced < magbits:
            slice_multiplier = 2**self.slice_size
            slice_divisor = slice_multiplier - 1
            x_int, x_mvm = np.divmod(x_int, float(slice_multiplier))
            islice = x_mvm / slice_divisor
            total_sliced += self.slice_size
            idx = total_sliced - magbits
            local_correction = slice_divisor / slice_multiplier
            output = _InputBitslice(
                islice=islice,
                idx=idx,
                correction_factor=global_correction * local_correction,
            )
            yield output


class SignMagnitudeDAC(QuantizerDAC):
    """Digital-to-Analog converter that quantizes signed values."""

    def __init__(
        self,
        dac_params: DACParameters,
    ):
        """Initializes a sign magnitude DAC.

        Args:
            dac_params: Parameters for the DAC

        Raises:
            ValueError: Raised on invalid configurations.
        """
        super().__init__(dac_params)

        # Remove one level to account for signed zero
        self.levels = self.levels - 1

        self.min = -1
        self.max = 1

        if not self.signed:
            raise ValueError(
                "SignMagnitudeDAC must be signed, "
                "use QuantizerDAC for unsigned conversions",
            )

        # TODO: Do we need percentile for DAC inputs?
        #       (2025-01-03)
        # if not self.input_mapping_params.percentile and (
        #     self.input_mapping_params.max != -self.input_mapping_params.min
        # ):
        if self.input_mapping_params.max != -self.input_mapping_params.min:
            raise ValueError("SignMagnitudeDAC requires a symmetric range about 0")

    def set_limits(self, matrix: npt.ArrayLike):
        """Sets the limits of the DAC.

        May depend on the input matrix but not required.
        """
        pass

    def convert_sliced(
        self,
        vector: npt.ArrayLike,
        slice_size=None,
    ) -> Iterator[_InputBitslice]:
        """Returns an iterator that converts slices of digital values to analog.

        Converts with respect to the limits set for the dac.
        """
        if slice_size:
            self.slice_size = slice_size

        # First, convert the inputs to integers from 0 to 2^n-1
        x_mag = xp.abs(vector)
        x_sign = xp.sign(vector)
        magbits = self.bits - 1
        x_int = xp.rint(x_mag * (pow(2, magbits) - 1))

        total_sliced = 0
        global_correction = pow(2, magbits) / (pow(2, magbits) - 1)
        while total_sliced < magbits:
            slice_multiplier = 2**self.slice_size
            slice_divisor = slice_multiplier - 1
            x_int, x_mvm = np.divmod(x_int, float(slice_multiplier))
            islice = x_sign * x_mvm / slice_divisor
            total_sliced += self.slice_size
            idx = total_sliced - magbits
            local_correction = slice_divisor / slice_multiplier
            output = _InputBitslice(
                islice=islice,
                idx=idx,
                correction_factor=global_correction * local_correction,
            )
            yield output
