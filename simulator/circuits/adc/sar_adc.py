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


class SarADC(IADC):
    """This class implements the Successive Approximation Register (SAR) ADC model described in:
    M. Spear et al, "The Impact of Analog-to-Digital Converter Architecture and Variability on Analog Neural Network Accuracy"
    IEEE Journal on Exploratory Solid-State Computational Devices and Circuits (accepted), 2023.

    The SAR ADC compares analog outputs against a capacitive DAC (CDAC) using a comparator. The number of CDAC queries is
    the # ADC bits, by using a binary search algorithm.
    The capacitance CDAC outputs can have error due to random capacitance mismatch. The comparator can have random offset.
    There is also error introduced by finite gain in the op amp at the output of the CDAC.

    The CDAC can have a split or non-split implementation which have different error sensitivities. See Section II of the paper
    for details and a discussion of design trade-offs.

    We assume that a single CDAC is shared by multiple columns, defined by the parameter group_size
    The columns of an array are then partitioned into groups using group_size. Different groups use different CDACs with
    independent random errors. Columns within the same group are affected by the same random errors in the CDAC and comparator.
    """

    # Initialize SAR ADC
    def set_limits(self, matrix):
        super().set_limits(matrix)

        # SAR ADC parameters
        self.sar_params = self.adc_params

        # SW packing
        Ncopy = (
            self.simulation_params.convolution.x_par
            * self.simulation_params.convolution.y_par
        )

        # Set the reference voltage (normalized)
        self.Vref = xp.maximum(xp.abs(self.min), xp.abs(self.max))

        split_cdac = self.sar_params.split_cdac
        gain_db = self.sar_params.gain_db
        sigma_capacitor = self.sar_params.sigma_capacitor
        sigma_comparator = self.sar_params.sigma_comparator * self.Vref
        group_size = self.sar_params.group_size

        num_adcs = int(xp.ceil(matrix.shape[0] / group_size))
        gain = 10 ** (gain_db / 20)

        # DAC levels (output)
        dac_levels_all = xp.zeros((num_adcs, 2**self.bits))

        # Utility constants
        bits_vec = xp.arange(self.bits)
        counts = xp.arange(2**self.bits)

        # Draw random capacitor mismatches
        cap_mismatches = xp.random.normal(
            loc=0.0,
            scale=sigma_capacitor,
            size=self.bits + 1,
        )

        if not split_cdac:
            #### Standard non-split DAC

            # Utility constants and conditions based on DAC level
            X1 = 2 ** (self.bits - 1) - counts
            cond1 = get_digit(counts - 2 ** (self.bits - 1), self.bits - 1) == 1
            cond2 = xp.logical_not(cond1)

            for rd in range(num_adcs):
                # Account for the fact that capacitance mismatch scales with the square root of capacitor size
                # in a typical foundry process
                cap_errors = xp.zeros(self.bits + 1)
                cap_errors[: self.bits] = 2**bits_vec + cap_mismatches[
                    : self.bits
                ] * (2 ** (bits_vec / 2))
                cap_errors[self.bits] = 2 ** (self.bits - 1) + cap_mismatches[
                    self.bits
                ] * (2 ** ((self.bits - 1) / 2))

                temp1 = xp.sum(
                    get_digit(X1[:, None], bits_vec[None, :]) * cap_errors[None, :-1],
                    axis=1,
                )
                temp2 = xp.sum(
                    get_digit(-X1[:, None], bits_vec[None, :]) * cap_errors[None, :-1],
                    axis=1,
                )
                temp = temp1 * cond1 + temp2 * cond2

                beta = cap_errors[self.bits] / (cap_errors[self.bits] + temp)
                D = get_digit(counts - 2 ** (self.bits - 1), self.bits - 1)
                dac_levels = temp / cap_errors[self.bits - 1] * (
                    self.Vref - 2 * D * self.Vref
                ) + self.Vref / (2**self.bits)
                dac_levels_all[rd, :] = dac_levels / (1 + 1 / (gain * beta))

        else:
            #### Split DAC

            # Utility constants
            NH = int(xp.floor(self.bits / 2))
            X1 = 2 ** (self.bits - 1) - counts
            X2 = xp.arange(NH)
            X3 = counts - 2 ** (self.bits - 1)
            X4 = xp.arange(NH, self.bits)

            # Conditions based on DAC level
            cond0 = counts == 0
            cond1 = get_digit(counts - 2 ** (self.bits - 1), self.bits - 1) == 1
            cond2 = xp.logical_not(cond1)
            cond1 = xp.logical_and(cond1, xp.logical_not(cond0))
            cond2 = xp.logical_and(cond2, xp.logical_not(cond0))

            for rd in range(num_adcs):
                # Account for typical foundry variation and the split DAC attenuation factor
                cap_errors = xp.zeros(self.bits + 1)
                for n in range(self.bits - 1):
                    if n < xp.floor(self.bits / 2):
                        cap_errors[n] = 2**n + cap_mismatches[n] * 2 ** (n / 2)
                    else:
                        cap_errors[n] = 2 ** (
                            n - xp.floor(self.bits / 2)
                        ) + cap_mismatches[n] * 2 ** ((n - xp.floor(self.bits / 2)) / 2)
                cap_errors[self.bits - 1] = 1 + cap_mismatches[self.bits - 1]
                cb_coef = (2 ** xp.floor(self.bits / 2)) / (
                    (2 ** (xp.floor(self.bits / 2))) - 1
                )
                cap_errors[self.bits] = cb_coef + cap_mismatches[
                    self.bits
                ] * cb_coef ** (1 / 2)

                cb = cap_errors[self.bits]
                cLSB = cap_errors[self.bits - 1] + xp.sum(cap_errors[0:NH])
                cMSB = xp.sum(cap_errors[NH : self.bits - 1])

                clsbs1 = xp.sum(
                    get_digit(X1[:, None], X2[None, :]) * cap_errors[None, :NH],
                    axis=1,
                )
                cmsbs1 = xp.sum(
                    get_digit(X1[:, None], X4[None, :])
                    * cap_errors[None, NH : self.bits],
                    axis=1,
                )
                temp1 = (clsbs1 * cb + cmsbs1 * (cLSB + cb)) / (
                    (cLSB + cb) * (cMSB + cb) - cb**2
                )

                clsbs2 = xp.sum(
                    get_digit(X3[:, None], X2[None, :]) * cap_errors[None, :NH],
                    axis=1,
                )
                cmsbs2 = xp.sum(
                    get_digit(X3[:, None], X4[None, :])
                    * cap_errors[None, NH : self.bits],
                    axis=1,
                )
                temp2 = (clsbs2 * cb + cmsbs2 * (cLSB + cb)) / (
                    (cLSB + cb) * (cMSB + cb) - cb**2
                )

                temp = temp1 * cond1 + temp2 * cond2 + cond0

                beta = 1
                D = get_digit(counts - 2 ** (self.bits - 1), self.bits - 1)
                dac_levels = temp * (self.Vref - 2 * D * self.Vref) + self.Vref / (
                    2**self.bits
                )
                dac_levels_all[rd, :] = dac_levels / (1 + 1 / (gain * beta))

        # Duplicate DAC for the columns within a group that share it
        dac_levels_all = xp.repeat(dac_levels_all, group_size, axis=0)
        self.dac_levels_all = dac_levels_all[: matrix.shape[0], :] - self.Vref / 2 ** (
            self.bits
        )

        # Generate comparator offsets
        comparator_offsets = xp.random.normal(
            loc=0,
            scale=sigma_comparator,
            size=num_adcs,
        )
        comparator_offsets = xp.repeat(comparator_offsets, group_size, axis=0)
        self.comparator_offsets = comparator_offsets[: matrix.shape[0]]

        if Ncopy > 1:
            x_par = self.simulation_params.convolution.x_par
            y_par = self.simulation_params.convolution.y_par
            self.dac_levels_all = xp.tile(self.dac_levels_all, (x_par * y_par, 1))
            self.comparator_offsets = xp.tile(self.comparator_offsets, x_par * y_par)

        # Pre-computation of indexing matrix
        if self.simulation_params.convolution.conv_matmul:
            range_vec = xp.arange(matrix.shape[0])
            self.range_mat = xp.repeat(
                range_vec[:, None],
                self.simulation_params.convolution.Nwindows,
                axis=1,
            )

    # Run-time method to simulate SAR ADC
    def convert(self, vector):
        if self.bits is None or self.bits == 0:
            return vector

        # Clip vector
        input_ = vector.clip(self.min, self.max)

        # The loop below implements the SAR binary search.
        # Iterate through the bits, but operate on all ADC inputs in parallel

        # Container for SAR register values
        dig_reg = xp.zeros(input_.shape, dtype=int)

        ### MVM
        if len(input_.shape) == 1:
            for i in range(self.bits):
                dig_comp = 2 ** (self.bits - i - 1)
                Vdac = self.dac_levels_all[xp.arange(len(input_)), dig_comp + dig_reg]
                bi = input_ > (Vdac - self.comparator_offsets)
                dig_reg += 2 ** (self.bits - i - 1) * bi

        ### Matmul
        else:
            for i in range(self.bits):
                dig_comp = 2 ** (self.bits - i - 1)
                Vdac = self.dac_levels_all[self.range_mat, dig_comp + dig_reg]
                bi = input_ > (Vdac - self.comparator_offsets[:, None])
                dig_reg += 2 ** (self.bits - i - 1) * bi

        output = (
            2 * self.Vref * (dig_reg / 2**self.bits)
            - self.Vref * (2**self.bits - 1) / 2**self.bits
        )

        return output
