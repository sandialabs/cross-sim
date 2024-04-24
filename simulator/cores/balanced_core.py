#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np

from .wrapper_core import WrapperCore
from simulator.parameters.core_parameters import BalancedCoreStyle
from simulator.circuits.adc.adc import ADC
from simulator.circuits.dac.dac import DAC
from simulator.backend import ComputeBackend

xp = ComputeBackend()


class BalancedCore(WrapperCore):
    """A balanced core consisting of two inner cores.

    One core is designated as "positive"; the other one as "negative". The actual value is the sum of the values of the inner cores.

    Both cores are started at the middle of their dynamic range. Requested updates are divided by two, and then performed on both cores,
        but the update performed on the negative core first has its sign flipped.
    """

    def __init__(self, clipper_core_factory, params):
        """:param clipper_core_factory:
        :param params: all parameters
        :type params: Parameters
        :return:
        """
        WrapperCore.__init__(self, clipper_core_factory, params)

        self.core_pos = clipper_core_factory()
        self.core_neg = clipper_core_factory()

        # Need to include the VMM case here also at some point, but right now it's only
        # used in the MVM code
        self.adc_range_option = self.params.xbar.adc.mvm.adc_range_option
        self.adc_per_ibit = self.params.xbar.adc.mvm.adc_per_ibit

        # subtract_current_in_xbar requires fast_balanced = False and interleaved_posneg = False
        self.dac_params = self.params.xbar.dac
        self.adc_params = self.params.xbar.adc
        self.input_params = self.params.core.mapping.inputs
        self.interleaved_posneg = self.params.core.balanced.interleaved_posneg
        self.subtract_current_in_xbar = (
            self.params.core.balanced.subtract_current_in_xbar
        )
        self.fast_balanced = self.params.simulation.fast_balanced
        self.Icol_max = self.params.xbar.array.Icol_max
        self.clip_Icol = self.Icol_max > 0

        # Create ADC and DAC
        self.adc = ADC(
            self.adc_params,
            self.dac_params,
            self.params.core,
            self.params.simulation,
        )
        self.dac = DAC(self.dac_params, self.params.core)

        # Counter for MVMs/VMMs run
        self.i_op = 0

    def _wrapper_set_matrix(self, matrix, weight_limits=None, error_mask=None):
        # Store the matrix shape
        self.W_shape = matrix.shape
        matrix_norm = matrix / self.max
        Wrange_xbar = self.params.xbar.device.Grange_norm

        if self.core_pos.params.core.balanced.style is BalancedCoreStyle.ONE_SIDED:
            if self.params.xbar.device.cell_bits > 0:
                Wmin_res = 2 ** (-(self.params.xbar.device.cell_bits + 1))
            else:
                Wmin_res = 0

            mat_pos = self.core_pos.params.xbar.device.Gmin_norm * (
                matrix_norm < Wmin_res
            ) + (
                self.core_pos.params.xbar.device.Gmin_norm + Wrange_xbar * matrix_norm
            ) * (
                matrix_norm >= Wmin_res
            )
            mat_neg = self.core_pos.params.xbar.device.Gmin_norm * (
                matrix_norm >= -Wmin_res
            ) + (
                self.core_pos.params.xbar.device.Gmin_norm - Wrange_xbar * matrix_norm
            ) * (
                matrix_norm < -Wmin_res
            )
            mat_pos = mat_pos.astype(xp.float32)
            mat_neg = mat_neg.astype(xp.float32)

        else:
            mat_pos = (
                self.core_pos.params.xbar.device.Gmin_norm
                + Wrange_xbar * (1 + matrix_norm) / 2
            )
            mat_neg = (
                self.core_pos.params.xbar.device.Gmin_norm
                + Wrange_xbar * (1 - matrix_norm) / 2
            )
            mat_pos = mat_pos.astype(xp.float32)
            mat_neg = mat_neg.astype(xp.float32)

        self.core_pos.set_matrix(mat_pos, error_mask=error_mask)
        self.core_neg.set_matrix(mat_neg, error_mask=error_mask)

        if self.fast_balanced:
            # If fast balanced is on, core_pos and core_neg are not used for MVM
            # The DAC in core_pos is still used
            # core_neg is kept in memory to avoid errors when calling set_matrix() again
            self.W_balanced = (
                self.core_pos._read_matrix() - self.core_neg._read_matrix()
            )

        # ADC range options
        if self.adc_params.mvm.bits > 0 or self.adc_params.vmm.bits > 0:
            self.adc.set_limits(matrix)

        self.dac.set_limits(matrix)

        # If profiling ADC inputs, initialize data structure here now that matrix dimensions are known
        # Currently assuming profiling is only done for MVMs
        if self.params.simulation.analytics.profile_adc_inputs:
            # This is to ensure accurate binning of column currents to specific MVMs
            if (
                self.params.simulation.convolution.x_par > 1
                or self.params.simulation.convolution.y_par > 1
            ):
                raise ValueError(
                    "If profiling bit slicing currents, must use x_par, y_par = 1",
                )
            if self.dac_params.mvm.input_bitslicing:
                magbits = self.params.xbar.dac.mvm.bits
                if self.params.xbar.dac.mvm.signed:
                    magbits -= 1
            else:
                magbits = 1
            Nout_mvm = matrix.shape[0]
            if not self.subtract_current_in_xbar:
                Nout_mvm *= 2
            Nmvms = self.params.simulation.analytics.ntest
            if self.params.simulation.convolution.is_conv_core:
                Nmvms *= self.params.simulation.convolution.Nwindows
            if (
                self.params.simulation.convolution.conv_matmul
                and self.params.simulation.convolution.is_conv_core
            ):
                self.outputs_per_op = (
                    self.params.simulation.convolution.Nwindows * Nout_mvm
                )
            else:
                self.outputs_per_op = Nout_mvm
            self.adc_inputs = xp.zeros((magbits, Nmvms * Nout_mvm), dtype=xp.float32)

    def _wrapper_set_vmm_inputs(self, vector):
        vec_dac = self.dac.vmm.convert(vector)
        self.core_pos.set_vmm_inputs(vec_dac)
        if not self.params.simulation.fast_balanced:
            self.core_neg.set_vmm_inputs(vec_dac)

    def _wrapper_set_mvm_inputs(self, vector):
        vec_dac = self.dac.mvm.convert(vector)
        self.core_pos.set_mvm_inputs(vec_dac)
        if not self.params.simulation.fast_balanced:
            self.core_neg.set_mvm_inputs(vec_dac)

    def _wrapper_run_xbar_mvm(self):
        vector = self.core_pos.vector_mvm
        op = "mvm"
        return self.run_xbar_operation(op, vector)

    def _wrapper_run_xbar_vmm(self):
        vector = self.core_pos.vector_vmm
        op = "vmm"
        return self.run_xbar_operation(op, vector)

    def run_xbar_operation(self, op, vector):
        function = "run_xbar_" + op
        core_pos_operation = getattr(self.core_pos, function)
        core_neg_operation = getattr(self.core_neg, function)
        adc_params = getattr(self.adc_params, op)
        dac_params = getattr(self.dac_params, op)
        adc = getattr(self.adc, op)
        dac = getattr(self.dac, op)

        ################################
        ##  ANALOG INPUT ENCODING
        ################################
        if not dac_params.input_bitslicing:
            if not self.interleaved_posneg:
                if self.fast_balanced:
                    if op == "mvm":
                        output = xp.matmul(self.W_balanced, vector)
                    else:
                        output = xp.matmul(vector, self.W_balanced)
                else:
                    output_pos = core_pos_operation()
                    output_neg = core_neg_operation()

                    if self.clip_Icol:
                        output_pos = output_pos.clip(-self.Icol_max, self.Icol_max)
                        output_neg = output_neg.clip(-self.Icol_max, self.Icol_max)

                    if self.subtract_current_in_xbar:
                        output = output_pos - output_neg
            else:
                output = core_pos_operation(core_neg=self.core_neg)
                if self.clip_Icol:
                    output = output.clip(-self.Icol_max, self.Icol_max)

            # ADC input profiling
            if self.params.simulation.analytics.profile_adc_inputs:
                i1 = self.i_op * self.outputs_per_op
                i2 = i1 + self.outputs_per_op
                if self.subtract_current_in_xbar or self.interleaved_posneg:
                    self.adc_inputs[0, i1:i2] = xp.array(
                        output.flatten(),
                        dtype=xp.float32,
                    )
                else:
                    self.adc_inputs[0, i1:i2] = xp.array(
                        xp.concatenate((output_pos.flatten(), output_neg.flatten())),
                        dtype=xp.float32,
                    )

        ################################
        ##  INPUT BIT SLICING
        ################################
        else:
            vector_slices = dac.convert_sliced(vector)
            num_input_slices = len(vector_slices)
            slice_size = dac_params.slice_size
            magbits = dac_params.bits
            if dac_params.signed:
                magbits -= 1

            for k in range(num_input_slices):
                if not self.interleaved_posneg:
                    if self.fast_balanced:
                        if op == "mvm":
                            output_bal = xp.matmul(self.W_balanced, vector_slices[k])
                        else:
                            output_bal = xp.matmul(vector_slices[k], self.W_balanced)
                    else:
                        output_pos = core_pos_operation(vector=vector_slices[k])
                        output_neg = core_neg_operation(vector=vector_slices[k])

                        # Clip the accumulated current on each column
                        if self.clip_Icol:
                            output_pos = output_pos.clip(-self.Icol_max, self.Icol_max)
                            output_neg = output_neg.clip(-self.Icol_max, self.Icol_max)

                        if self.subtract_current_in_xbar:
                            output_bal = output_pos - output_neg
                else:
                    output_bal = core_pos_operation(
                        vector=vector_slices[k],
                        core_neg=self.core_neg,
                    )

                    if self.clip_Icol:
                        output_bal = output_bal.clip(-self.Icol_max, self.Icol_max)

                # Profiling of bit sliced array outputs
                if self.params.simulation.analytics.profile_adc_inputs:
                    i1 = self.i_op * self.outputs_per_op
                    i2 = i1 + self.outputs_per_op
                    if self.subtract_current_in_xbar or self.interleaved_posneg:
                        self.adc_inputs[k, i1:i2] = xp.array(
                            output_bal.flatten(),
                            dtype=xp.float32,
                        )
                    else:
                        self.adc_inputs[k, i1:i2] = xp.array(
                            xp.concatenate(
                                (output_pos.flatten(), output_neg.flatten()),
                            ),
                            dtype=xp.float32,
                        )

                # ADC
                if adc_params.adc_per_ibit:
                    if self.subtract_current_in_xbar or self.interleaved_posneg:
                        output_bal = adc.convert(output_bal)
                    else:
                        output_pos = adc.convert(output_pos)
                        output_neg = adc.convert(output_neg)

                if self.subtract_current_in_xbar or self.interleaved_posneg:
                    if k == 0:
                        output = output_bal.copy()
                    else:
                        output += output_bal
                else:
                    if k == 0:
                        output_pos_all = output_pos.copy()
                        output_neg_all = output_neg.copy()
                    else:
                        output_pos_all += output_pos
                        output_neg_all += output_neg

                # Charge division or shift right
                if self.subtract_current_in_xbar:
                    output /= 2**slice_size
                else:
                    output_pos_all /= 2.0**slice_size
                    output_neg_all /= 2.0**slice_size

            # Scaling correction
            if self.subtract_current_in_xbar:
                # Correct for the fact that for multi-bit slices the highest value is
                # represented as 1.0 rather than 2^bits-1
                output *= (2**slice_size) - 1
                # Correct for the fact that the underutilization of the most
                # significant slice is not an exact divisor
                output *= 2 ** (slice_size * num_input_slices - magbits)
                # Correct for the fact that sum (1/2^i) is not exactly 1
                output *= pow(2, magbits) / (pow(2, magbits) - 1)
            else:
                # same corrections as above
                output_pos = output_pos_all * pow(2, magbits) / (pow(2, magbits) - 1)
                output_pos *= ((2**slice_size) - 1) * (
                    2 ** (slice_size * num_input_slices - magbits)
                )
                output_neg = output_neg_all * pow(2, magbits) / (pow(2, magbits) - 1)
                output_neg *= ((2**slice_size) - 1) * (
                    2 ** (slice_size * num_input_slices - magbits)
                )

        self.i_op += 1

        # ADC conversion
        if self.subtract_current_in_xbar:
            if not adc_params.adc_per_ibit:
                output = adc.convert(output)
        else:
            if not adc_params.adc_per_ibit:
                output_pos = adc.convert(output_pos)
                output_neg = adc.convert(output_neg)
            output = output_pos - output_neg

        return output

    def _wrapper_read_matrix(self):
        if not self.params.simulation.fast_balanced:
            output = self.core_pos._read_matrix() - self.core_neg._read_matrix()
        else:
            output = self.W_balanced.copy()
        output /= self.params.xbar.device.Grange_norm
        output *= self.max
        return output

    def _wrapper_save_matrix(self):
        return np.concatenate(
            self.core_pos._save_matrix(),
            self.core_neg._save_matrix(),
        )

    def _wrapper_restore_matrix(self, matrix):
        matrix = np.split(matrix, 2)
        self.core_pos._restore_matrix(matrix[0])
        self.core_neg._restore_matrix(matrix[1])

    def expand_matrix(self, Ncopy):
        # Calls expand_matrix in the inner cores
        # Makes multiple copies of matrix to compute multiple MVMs in parallel

        if not self.params.simulation.fast_balanced:
            self.core_pos.expand_matrix(Ncopy)
            self.core_neg.expand_matrix(Ncopy)
        else:
            if not self.params.simulation.convolution.weight_reorder:
                Nx, Ny = self.W_balanced.shape
                W_temp = self.W_balanced.copy()
                self.W_shape = self.W_balanced.shape
                self.W_balanced = xp.zeros(
                    (Ncopy * Nx, Ncopy * Ny),
                    dtype=self.W_balanced.dtype,
                )
                for m in range(Ncopy):
                    x_start, x_end = m * Nx, (m + 1) * Nx
                    y_start, y_end = m * Ny, (m + 1) * Ny
                    self.W_balanced[x_start:x_end, y_start:y_end] = W_temp.copy()

            else:
                self.W_balanced = self.core_pos.weight_reorder(self.W_balanced.copy())

    def unexpand_matrix(self):
        if not self.params.simulation.fast_balanced:
            # Calls unexpand_matrix in the inner cores
            self.core_pos.unexpand_matrix()
            self.core_neg.unexpand_matrix()
        else:
            self.W_balanced = self.W_balanced[: self.W_shape[0], : self.W_shape[1]]
