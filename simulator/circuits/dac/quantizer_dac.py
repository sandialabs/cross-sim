#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from .idac import IDAC
from simulator.backend import ComputeBackend

xp = ComputeBackend()  # Represents either cupy or numpy


class QuantizerDAC(IDAC):
    def __init__(self, dac_params, core_params):
        super().__init__(dac_params, core_params)
        self.levels = 2**self.bits

        if self.core_params.percentile is not None:
            if self.signed:
                self.min = -1
                self.max = 1
            else:
                self.min = 0
                self.max = 1
        else:
            if self.signed:
                self.min = 2 * self.core_params.min / self.core_params.range
                self.max = 2 * self.core_params.max / self.core_params.range
            else:
                self.min = self.core_params.min / self.core_params.range
                self.max = self.core_params.max / self.core_params.range

        # Calculate levels
        if not self.signed and (self.min < 0 or self.max < 0) and self.bits > 0:
            raise ValueError("Quantizer DAC: sign bit disabled but limits are negative")

    def set_limits(self, matrix):
        pass

    def convert(self, vector):
        # If bits is None, just return the input
        if not self.bits or self.bits == 0:
            return vector

        # Clip vector
        input_ = vector.clip(self.min, self.max)

        # Quantize
        # set qmult (quantization multiplier):  multiply by this factor to convert every level to an absolute range of 1
        # The -1 is because the first level is 0, i.e. 2 bits define 3 segments between 0 and 3
        qmult = (self.levels - 1) / (self.max - self.min)

        # do quantization using rounding
        input_ -= self.min
        input_ *= qmult
        input_ = xp.rint(input_, out=input_)
        input_ /= qmult
        input_ += self.min

        return input_

    def convert_sliced(self, vector, slice_size=None):
        if slice_size is None:
            slice_size = self.slice_size

        slice_multiplier = 2**slice_size
        slice_divisor = slice_multiplier - 1

        if self.signed:
            raise ValueError(
                "QuantizerDAC does not support input bit slicing with signed inputs: use SignMagnitudeDAC",
            )

        # First, convert the inputs to integers from 0 to 2^n-1
        magbits = self.bits
        x_int = xp.rint(vector * (pow(2, magbits) - 1))

        sliced_vect = []
        for k in range(int(xp.ceil(magbits / slice_size))):
            x_mvm = x_int % slice_multiplier
            sliced_vect.append(x_mvm / slice_divisor)
            x_int = x_int // slice_multiplier

        return sliced_vect


class SignMagnitudeDAC(QuantizerDAC):
    def __init__(self, dac_params, core_params):
        super().__init__(dac_params, core_params)

        # Remove one level to account for signed zero
        self.levels = self.levels - 1

        self.min = -1
        self.max = 1

        if not self.signed:
            raise ValueError(
                "SignMagnitudeDAC must be signed, use QuantizerDAC for unsigned conversions",
            )

        if not self.core_params.percentile and (
            self.core_params.max != -self.core_params.min
        ):
            raise ValueError("SignMagnitudeDAC requires a symmetric range about 0")

    def set_limits(self, matrix):
        pass

    def convert_sliced(self, vector, slice_size=None):
        if slice_size is None:
            slice_size = self.slice_size

        slice_multiplier = 2**slice_size
        slice_divisor = slice_multiplier - 1

        # First, convert the inputs to integers from 0 to 2^n-1
        x_mag = xp.abs(vector)
        x_sign = xp.sign(vector)
        magbits = self.bits - 1
        x_int = xp.rint(x_mag * (pow(2, magbits) - 1))

        sliced_vect = []
        for k in range(int(xp.ceil(magbits / slice_size))):
            x_mvm = x_int % slice_multiplier
            if self.signed:
                x_mvm *= x_sign
            sliced_vect.append(x_mvm / slice_divisor)
            x_int = x_int // slice_multiplier

        return sliced_vect
