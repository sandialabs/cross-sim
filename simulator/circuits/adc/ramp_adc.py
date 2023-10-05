#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from .iadc import IADC
from .iadc import get_digit
from simulator.backend import ComputeBackend

xp = ComputeBackend()


class RampADC(IADC):
    """This class implements the ramp ADC model described in:
    M. Spear et al, "The Impact of Analog-to-Digital Converter Architecture and Variability on Analog Neural Network Accuracy"
    IEEE Journal on Exploratory Solid-State Computational Devices and Circuits (accepted), 2023.

    This ramp ADC assumes that a single capacitive DAC is shared by all outputs of the array
    The capacitive DAC generates a ramp signal, which is compared to the analog output of each column using
    a per-column comparator.
    This model simulates random mismatches in the capacitance inside the DACs and random offsets in the column comparators,
    as well as finite gain in the amplifier output of the DAC.
    """

    # Initialize Ramp ADC
    def set_limits(self, matrix):
        super().set_limits(matrix)

        # Ramp ADC parameters
        self.ramp_params = self.adc_params

        # Set the reference voltage (normalized)
        self.Vref = xp.maximum(xp.abs(self.min), xp.abs(self.max))

        gain_db = self.ramp_params.gain_db
        sigma_capacitor = self.ramp_params.sigma_capacitor
        sigma_comparator = self.ramp_params.sigma_comparator * self.Vref
        Ncopy = (
            self.simulation_params.convolution.x_par
            * self.simulation_params.convolution.y_par
        )

        # Generate random comparator offsets
        self.comparator_offsets = xp.random.normal(
            loc=0.0,
            scale=sigma_comparator,
            size=matrix.shape[0],
        )
        if Ncopy > 1:
            self.comparator_offsets = xp.tile(self.comparator_offsets, Ncopy)

        # Draw random capacitor mismatches
        cap_mismatches = xp.random.normal(
            loc=0.0,
            scale=sigma_capacitor,
            size=self.bits + 1,
        )

        # Create the list of DAC levels
        self.dac_levels = xp.zeros(2**self.bits)
        beta = self.dac_levels.copy()
        bits_vec = xp.arange(self.bits)  # up to N-1
        bits_vec_ext = xp.arange(self.bits + 1)  # up to N

        if self.ramp_params.symmetric_cdac:
            # Account for the fact that capacitance mismatch scales with the square root of capacitor size
            # in a typical foundry process
            cap_errors = xp.zeros(self.bits + 1)
            cap_errors[: self.bits] = 2**bits_vec + cap_mismatches[: self.bits] * (
                2 ** (bits_vec / 2)
            )
            cap_errors[self.bits] = 2 ** (self.bits - 1) + cap_mismatches[self.bits] * (
                2 ** ((self.bits - 1) / 2)
            )

            for count in range(2**self.bits):
                # Check MSB (sign bit)
                if get_digit(count - 2 ** (self.bits - 1), self.bits) == 1:
                    temp = xp.sum(
                        get_digit(2 ** (self.bits - 1) - count, bits_vec)
                        * cap_errors[: self.bits],
                    )
                else:
                    temp = xp.sum(
                        get_digit(count - 2 ** (self.bits - 1), bits_vec)
                        * cap_errors[: self.bits],
                    )
                beta[count] = cap_errors[self.bits] / (cap_errors[self.bits] + temp)
                D = get_digit(count - 2 ** (self.bits - 1), self.bits - 1)
                self.dac_levels[count] = temp / cap_errors[self.bits - 1] * (
                    self.Vref - 2 * D * self.Vref
                ) + self.Vref / (2**self.bits)

        else:
            cap_errors = 2**bits_vec_ext + cap_mismatches * (2 ** (bits_vec_ext / 2))
            for count in range(2**self.bits):
                temp = xp.sum(get_digit(count, bits_vec) * cap_errors[: self.bits])
                beta[count] = cap_errors[self.bits] / (cap_errors[self.bits] + temp)
                self.dac_levels[count] = temp / cap_errors[
                    self.bits
                ] * 2 * self.Vref - self.Vref * (2**self.bits - 1) / (2**self.bits)

        # Adjust by the amplifier gain
        gain = 10 ** (gain_db / 20)
        self.dac_levels = self.dac_levels / (1 + 1 / (gain * beta))

    # Run-time method to simulate Ramp ADC
    def convert(self, vector):
        if self.bits is None or self.bits == 0:
            return vector

        # Clip vector
        input_ = vector.clip(self.min, self.max)

        # Matmul
        if len(input_.shape) == 2:
            Vdiff = xp.abs(
                self.dac_levels[:, None, None]
                - input_[None, :, :]
                - self.comparator_offsets[None, :, None],
            )
        # MVM
        else:
            Vdiff = xp.abs(self.dac_levels[:, None] - input_ - self.comparator_offsets)

        Vi = xp.argmin(Vdiff, axis=0)
        output = 2 * self.Vref * (Vi / 2**self.bits) - self.Vref * (
            2**self.bits - 1
        ) / (2**self.bits)

        return output
