#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from .iadc import IADC
from simulator.backend import ComputeBackend

xp = ComputeBackend()


class CyclicADC(IADC):
    """This class implements the Cyclic ADC model described in:
    M. Spear et al, "The Impact of Analog-to-Digital Converter Architecture and Variability on Analog Neural Network Accuracy"
    IEEE Journal on Exploratory Solid-State Computational Devices and Circuits (accepted), 2023.

    The cyclic ADC operates in the same way as the pipeline ADC, except that instead of having (N-1) 1.5-bit
    stages, a single 1.5-bit stage is re-used cyclically across all the bits. This means the ADC cannot be
    pipelined.

    This ADC uses the same set of parameters as the pipeline ADC

    """

    # Initialize Cyclic ADC
    def set_limits(self, matrix):
        super().set_limits(matrix)

        # Cyclic ADC parameters
        self.cyclic_params = self.adc_params

        # SW packing
        Ncopy = (
            self.simulation_params.convolution.x_par
            * self.simulation_params.convolution.y_par
        )

        # Set the reference voltage (normalized)
        self.Vref = xp.maximum(xp.abs(self.min), xp.abs(self.max))

        sigma_C1 = self.cyclic_params.sigma_C1
        sigma_C2 = self.cyclic_params.sigma_C2
        sigma_Cpar = self.cyclic_params.sigma_Cpar
        sigma_comparator = self.cyclic_params.sigma_comparator * self.Vref
        gain_db = self.cyclic_params.gain_db
        group_size = self.cyclic_params.group_size

        # Number of independent cylic ADCs used by the array
        num_adcs = int(xp.ceil(matrix.shape[0] / group_size))
        self.gain = 10 ** (gain_db / 20)
        self.Vh = self.Vref / 4
        self.Vl = -self.Vref / 4

        # Sample comparator offsets
        comparator_offsets = xp.random.normal(
            loc=0.0,
            scale=sigma_comparator,
            size=(num_adcs, 3),
        )
        comparator_offsets = xp.repeat(comparator_offsets, group_size, axis=0)
        self.comparator_offsets = comparator_offsets[: matrix.shape[0], :]

        # Sample capacitance errors
        C1 = xp.random.normal(loc=1.0, scale=sigma_C1, size=(num_adcs,))
        C2 = xp.random.normal(loc=1.0, scale=sigma_C2, size=(num_adcs,))
        Cpar = xp.random.normal(loc=0.0, scale=sigma_Cpar, size=(num_adcs,))
        C1 = xp.repeat(C1, group_size)
        C2 = xp.repeat(C2, group_size)
        Cpar = xp.repeat(Cpar, group_size)
        C1 = C1[: matrix.shape[0]]
        C2 = C2[: matrix.shape[0]]
        Cpar = Cpar[: matrix.shape[0]]

        if Ncopy > 1:
            x_par = self.simulation_params.convolution.x_par
            y_par = self.simulation_params.convolution.y_par
            self.comparator_offsets = xp.tile(
                self.comparator_offsets,
                (x_par * y_par, 1),
            )
            C1 = xp.tile(C1, x_par * y_par)
            C2 = xp.tile(C2, x_par * y_par)
            Cpar = xp.tile(Cpar, x_par * y_par)

        # Compute the beta and gamma factors
        self.beta = C1 / (C1 + C2 + Cpar)
        self.gamma = C2 / C1

    # Run-time method to simulate cyclic ADC
    def convert(self, vector):
        if self.bits is None or self.bits == 0:
            return vector

        # Clip vector
        input_ = vector.clip(self.min, self.max)

        # The loop below implements the cyclic ADC bit-by-bit sequential conversion
        # Iterate through the bits, but operate on all ADC inputs in parallel
        Vx = input_.copy()

        ### MVM
        if len(input_.shape) == 1:
            b = xp.zeros((self.bits, self.bits, len(input_)))
            for n in range(self.bits):
                if n == self.bits - 1:
                    b[n, n, :] = Vx > self.comparator_offsets[:, 2]

                else:
                    b[n, n, :] = Vx > (self.Vh + self.comparator_offsets[:, 0])
                    b[n, n + 1, :] = (
                        Vx < (self.Vh + self.comparator_offsets[:, 0])
                    ) * (Vx > (self.Vl + self.comparator_offsets[:, 1]))
                    # The 1.0 is used to convert to float
                    C = 1.0 * (Vx < (self.Vl + self.comparator_offsets[:, 1])) - 1.0 * (
                        Vx > (self.Vh + self.comparator_offsets[:, 0])
                    )
                    Vx = ((1 + self.gamma) * Vx + C * self.Vref * self.gamma) / (
                        1 + 1 / (self.gain * self.beta)
                    )

            code = xp.sum(b, axis=0)
            codeint = xp.zeros(len(input_))
            for n in range(self.bits):
                codeint += (2**n) * code[self.bits - n - 1, :]

        ### Matmul
        else:
            b = xp.zeros((self.bits, self.bits, input_.shape[0], input_.shape[1]))

            for n in range(self.bits):
                if n == self.bits - 1:
                    b[n, n, :, :] = Vx > self.comparator_offsets[:, 2, None]

                else:
                    b[n, n, :, :] = Vx > (self.Vh + self.comparator_offsets[:, 0, None])
                    b[n, n + 1, :, :] = (
                        Vx < (self.Vh + self.comparator_offsets[:, 0, None])
                    ) * (Vx > (self.Vl + self.comparator_offsets[:, 1, None]))
                    C = 1.0 * (
                        Vx < (self.Vl + self.comparator_offsets[:, 1, None])
                    ) - 1.0 * (Vx > (self.Vh + self.comparator_offsets[:, 0, None]))
                    Vx = (
                        (1 + self.gamma[:, None]) * Vx
                        + C * self.Vref * self.gamma[:, None]
                    ) / (1 + 1 / (self.gain * self.beta[:, None]))

            code = xp.sum(b, axis=0)
            codeint = xp.zeros(input_.shape)
            for n in range(self.bits):
                codeint += (2**n) * code[self.bits - n - 1, :, :]

        output = (
            2 * self.Vref * (codeint / 2**self.bits)
            - self.Vref * (2**self.bits - 1) / 2**self.bits
        )

        return output
