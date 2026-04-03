#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from .iadc import IADC
from simulator.backend import ComputeBackend

xp = ComputeBackend()  # Represents either cupy or numpy


class QuantizerADC(IADC):
    """Analog-to-Digital converter that simulates an ideal quantizer.

    This quantizer a range that is non-symmetric about zero. Typically this
    means [0,1] or [0,N] but other ranges are possible but may be difficult to
    implement in practice.
    """

    def __init__(
        self,
        adc_params,
        dac_params,
        core_params,
        simulation_params,
        bitslice,
    ):
        """Initializes the ideal non-symmetric quantizer.

        Args:
            adc_params: Parameters to describe ADC model
            dac_params:
                Parameters of the DAC driving the array. Unused by this model.
                Used by IADC __init__ to model the "full precision guarantee"
            core_params:
                Parameters of the core the ADC is connected to. Unused by this
                model. Used by IADC __init__ and set_limts to model the
                "full precision guarantee"
            simulation_params:
                Parameters of the simulation. Used for configuring ADC input
                profiling.
            bitslice:
                Which bitslice this ADC is connected to. Ignored if the ADC is
                connected to a non-bitsliced core.
        """
        super().__init__(
            adc_params,
            dac_params,
            core_params,
            simulation_params,
            bitslice,
        )
        self.levels = 2**self.bits

    def convert(self, vector):
        """Perform an ADC conversion with an ideal quantizer."""
        if self.bits is None or self.bits == 0:
            return vector

        # Clip vector
        input_ = vector.clip(self.min, self.max)

        # Quantize
        # Set qmult (quantization multiplier): converts every level to an
        # absolute range of 1 The -1 is because the first level is 0, i.e.
        # 2 bits define 3 segments between 0 and 3
        qmult = (self.levels - 1) / self.range

        # do quantization using rounding
        # shift min to zero
        input_ -= self.min
        # multiply by a quantization factor to allow for rounding
        # -> sigma becomes defined relative to 1 level
        input_ *= qmult

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
        """Initializes the ADC limits for the ADC model."""
        super().set_limits(matrix)


class SignMagnitudeADC(QuantizerADC):
    """Analog-to-Digital converter that simulates an ideal signed quantizer.

    This quantizer a range that is symmetric about zero. Importantly this
    model assumes that 0 is a discrete code such that the system has 1 fewer
    levels than would be implied by the number of bits.
    """

    def __init__(
        self,
        adc_params,
        dac_params,
        core_params,
        simulation_params,
        bitslice,
    ):
        """Initializes the ideal signed quantizer.

        Args:
            adc_params: Parameters to describe ADC model
            dac_params:
                Parameters of the DAC driving the array. Unused by this model.
                Used by IADC __init__ to model the "full precision guarantee"
            core_params:
                Parameters of the core the ADC is connected to. Unused by this
                model. Used by IADC __init__ and set_limts to model the
                "full precision guarantee"
            simulation_params:
                Parameters of the simulation. Used for configuring ADC input
                profiling.
            bitslice:
                Which bitslice this ADC is connected to. Ignored if the ADC is
                connected to a non-bitsliced core.
        """
        super().__init__(
            adc_params,
            dac_params,
            core_params,
            simulation_params,
            bitslice,
        )
        if not self.adc_params.signed:
            raise ValueError(
                "SignMagnitudeADC must be signed, use QuantizerADC for "
                "unsigned conversions",
            )

        # Remove one level to account for signed zero
        self.levels = self.levels - 1

    def set_limits(self, matrix):
        """Initializes the ADC limits for the ADC model."""
        super().set_limits(matrix)
        if self.max != -self.min:
            raise ValueError("SignMagnitudeADC requires a symmetric range about 0")
