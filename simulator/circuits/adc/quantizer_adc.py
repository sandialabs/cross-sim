#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from .iadc import IADC
from simulator.backend import ComputeBackend

xp = ComputeBackend()  # Represents either cupy or numpy


class QuantizerADC(IADC):
    def __init__(
        self,
        adc_params,
        dac_params,
        core_params,
        simulation_params,
        bitslice,
    ):
        super().__init__(
            adc_params,
            dac_params,
            core_params,
            simulation_params,
            bitslice,
        )
        self.levels = 2**self.bits

    def convert(self, vector):
        if self.bits is None or self.bits == 0:
            return vector

        # Clip vector
        input_ = vector.clip(self.min, self.max)

        # Quantize
        # Set qmult (quantization multiplier): converts every level to an absolute range of 1
        # The -1 is because the first level is 0, i.e. 2 bits define 3 segments between 0 and 3
        qmult = (self.levels - 1) / self.range

        # do quantization using rounding
        input_ -= self.min  # shift min to zero
        input_ *= qmult  # multiply by a quantization factor to allow for rounding -> sigma becomes defined relative to 1 level

        if self.adc_params.stochastic_rounding:
            input_floor = xp.floor(input_)
            input_ = input_floor + (
                xp.random.random_sample(xp.shape(input_)) < (input_ - input_floor)
            )
        else:
            input_ = xp.rint(input_, out=input_)

        input_ /= qmult
        input_ += self.min  # shift zero back

        return input_

    def set_limits(self, matrix):
        super().set_limits(matrix)


class SignMagnitudeADC(QuantizerADC):
    def __init__(
        self,
        adc_params,
        dac_params,
        core_params,
        simulation_params,
        bitslice,
    ):
        super().__init__(
            adc_params,
            dac_params,
            core_params,
            simulation_params,
            bitslice,
        )
        if not self.adc_params.signed:
            raise ValueError(
                "SignMagnitudeADC must be signed, use QuantizerADC for unsigned conversions",
            )

        # Remove one level to account for signed zero
        self.levels = self.levels - 1

    def set_limits(self, matrix):
        super().set_limits(matrix)
        if self.max != -self.min:
            raise ValueError("SignMagnitudeADC requires a symmetric range about 0")
