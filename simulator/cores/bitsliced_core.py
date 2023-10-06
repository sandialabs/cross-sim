#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np
from .wrapper_core import WrapperCore
from simulator.parameters.core_parameters import BitSlicedCoreStyle, OffsetCoreStyle
from simulator.backend import ComputeBackend
from simulator.circuits.adc.adc import ADC
from simulator.circuits.dac.dac import DAC

xp = ComputeBackend()


class BitslicedCore(WrapperCore):
    """A core consisting of two or more arrays that implements bit-sliced weights
    At least two slices are required (otherwise, use balanced core).
    """

    def __init__(self, clipper_core_factory, params):
        """:param clipper_core_factory:
        :param params: all parameters
        :type params: Parameters
        :return:
        """
        WrapperCore.__init__(self, clipper_core_factory, params)

        Nslices = params.core.bit_sliced.num_slices

        self.balanced = params.core.bit_sliced.style == BitSlicedCoreStyle.BALANCED
        self.adc_per_ibit = params.xbar.adc.mvm.adc_per_ibit
        self.digital_offset = (
            self.params.core.offset.style == OffsetCoreStyle.DIGITAL_OFFSET
        )
        self.subtract_current_in_xbar = (
            self.params.core.balanced.subtract_current_in_xbar
        )
        self.interleaved_posneg = self.params.core.balanced.interleaved_posneg
        self.Gmin_norm = self.params.xbar.device.Gmin_norm

        # i = 0 is least significant bit
        self.Nslices = Nslices
        if self.balanced:
            self.core_slices = [
                [clipper_core_factory(), clipper_core_factory()] for i in range(Nslices)
            ]

            if self.params.simulation.fast_balanced:
                # Initialize list of matrices
                self.W_balanced = [None for i in range(Nslices)]

        else:
            self.core_slices = [[clipper_core_factory()] for i in range(Nslices)]

        self.adc_params = self.params.xbar.adc
        self.dac_params = self.params.xbar.dac
        self.Icol_max = self.params.xbar.array.Icol_max
        self.clip_Icol = self.params.xbar.array.Icol_max > 0

        # Create ADCs
        # Give the correct Nbits_reduction parameter to each slice
        self.adcs = [None] * Nslices
        for i in range(Nslices):
            self.adcs[i] = ADC(
                self.adc_params,
                self.dac_params,
                self.params.core,
                self.params.simulation,
                bitslice=i,
            )

        self.dac = DAC(self.dac_params, self.params.core)

        self.i_op = 0

    def _wrapper_set_matrix(self, matrix, weight_limits=None, error_mask=None):
        # Quantize weights here, and not in numeric_core
        # Wbits as set in inference_net is the number of weight bits per device, assuming a balanced core
        # In this case we add one to get the true number of weight bits (pos and neg)
        Wbits = self.params.core.weight_bits
        Nslices = self.params.core.bit_sliced.num_slices
        Wrange_xbar = self.params.xbar.device.Grange_norm

        if Wbits % Nslices == 0:
            Wbits_slice = int(Wbits / Nslices)
        elif self.balanced:
            Wbits_slice = np.ceil((Wbits - 1) / Nslices).astype(int)
        else:
            Wbits_slice = np.ceil(Wbits / Nslices).astype(int)

        # Shape of weight matrix before duplication
        self.W_shape = matrix.shape
        self.Wbits_slice = Wbits_slice
        self.Wmax = pow(2, Wbits)
        self.Woffset = pow(2, Wbits - 1)

        # Place matrix in the range (-0.5, 0.5)
        W = matrix / (2 * self.max)

        # First convert the matrix to the range (-2^(Wbits-1)-1, +2^(Wbits-1)-1)
        W *= self.Wmax
        W *= (pow(2, Wbits - 1) - 1) / pow(2, Wbits - 1)

        # Then convert the matrix to the range (0, 2^Wbits) by adding a bias
        if not self.balanced:
            W += self.Woffset

        # Quantize the weights to Wbits
        W = np.rint(W, out=W)

        # Decompose W into bit slices
        W_slices = [None for i in range(Nslices)]
        if not self.balanced:
            # Positive weights case
            W_slices[0] = W % pow(2, Wbits_slice)
            for i in range(1, Nslices):
                if i < Nslices - 1:
                    W = (W - W_slices[i - 1]) / pow(2, Wbits_slice)
                    W_slices[i] = W % pow(2, Wbits_slice)
                else:
                    W_slices[i] = (W - W_slices[i - 1]) / pow(2, Wbits_slice)
        else:
            # Negative weights case
            W_sign = np.sign(W)
            W_slices[0] = np.abs(W) % pow(2, Wbits_slice)
            for i in range(1, Nslices):
                if i < Nslices - 1:
                    W = (np.abs(W) - W_slices[i - 1]) / pow(2, Wbits_slice)
                    W_slices[i] = np.abs(W) % pow(2, Wbits_slice)
                else:
                    W_slices[i] = (np.abs(W) - W_slices[i - 1]) / pow(2, Wbits_slice)
            for i in range(Nslices):
                W_slices[i] = W_slices[i] * W_sign

        self.dac.set_limits(matrix)

        for i in range(Nslices):
            W_i = W_slices[i]

            # Now xbar weights in the range (0, 1)
            W_i /= pow(2, Wbits_slice) - 1

            if not self.balanced:
                if (
                    self.params.xbar.array.parasitics.enable
                    and self.params.xbar.array.parasitics.disable_slices[i]
                ):
                    self.core_slices[i][0].params.xbar.array.parasitics.enable = False

                # Unit column
                if not self.digital_offset:
                    if i < Nslices - 1:
                        G_unit = 0
                    else:
                        G_unit = ((2**self.Wbits_slice) // 2) / (
                            2**self.Wbits_slice - 1
                        )
                    W_i = np.insert(W_i, 0, G_unit, axis=0)

                if self.Gmin_norm > 0:
                    W_i = self.Gmin_norm + W_i * Wrange_xbar

                self.core_slices[i][0].set_matrix(W_i, error_mask=error_mask)

            else:
                if (
                    self.params.xbar.array.parasitics.enable
                    and self.params.xbar.array.parasitics.disable_slices[i]
                ):
                    self.core_slices[i][0].params.xbar.array.parasitics.enable = False
                    self.core_slices[i][1].params.xbar.array.parasitics.enable = False

                # Separate positive and negative
                W_i_pos = np.abs(W_i) * (W_i >= 0)
                W_i_neg = np.abs(W_i) * (W_i < 0)

                if self.Gmin_norm > 0:
                    W_i_pos = self.Gmin_norm + W_i_pos * Wrange_xbar
                    W_i_neg = self.Gmin_norm + W_i_neg * Wrange_xbar

                self.core_slices[i][0].set_matrix(W_i_pos, error_mask=error_mask)
                self.core_slices[i][1].set_matrix(W_i_neg, error_mask=error_mask)

                if self.params.simulation.fast_balanced:
                    self.W_balanced[i] = (
                        self.core_slices[i][0]._read_matrix()
                        - self.core_slices[i][1]._read_matrix()
                    )

            if self.adc_params.mvm.bits > 0:
                self.adcs[i].set_limits(W_i)

        # If profiling ADC inputs, initialize data structure here now that matrix dimensions are known
        # Currently assuming profiling is only done for MVMs
        if self.params.simulation.analytics.profile_adc_inputs:
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
            self.outputs_per_op = Nout_mvm
            if (
                self.params.simulation.convolution.conv_matmul
                and self.params.simulation.convolution.is_conv_core
            ):
                self.outputs_per_op *= self.params.simulation.convolution.Nwindows

            self.adc_inputs = xp.zeros(
                (Nslices, magbits, Nmvms * Nout_mvm),
                dtype=xp.float32,
            )

    def _wrapper_set_vmm_inputs(self, vector):
        vec_dac = self.dac.vmm.convert(vector)
        for i in range(self.Nslices):
            self.core_slices[i][0].set_vmm_inputs(vec_dac)
            if self.balanced and not self.params.simulation.fast_balanced:
                self.core_slices[i][1].set_vmm_inputs(vec_dac)

    def _wrapper_set_mvm_inputs(self, vector):
        vec_dac = self.dac.mvm.convert(vector)
        for i in range(self.Nslices):
            self.core_slices[i][0].set_mvm_inputs(vec_dac)
            if self.balanced and not self.params.simulation.fast_balanced:
                self.core_slices[i][1].set_mvm_inputs(vec_dac)

    def _wrapper_run_xbar_mvm(self):
        vector = self.core_slices[0][0].vector_mvm
        op = "mvm"
        return self.run_xbar_operation(op, vector)

    def _wrapper_run_xbar_vmm(self):
        vector = self.core_slices[0][0].vector_vmm
        op = "vmm"
        return self.run_xbar_operation(op, vector)

    def run_xbar_operation(self, op, vector):
        Wbits = self.params.core.weight_bits
        Nslices = self.params.core.bit_sliced.num_slices
        Wbits_slice = self.Wbits_slice

        function = "run_xbar_" + op
        getattr(self.adc_params, op)
        dac_params = getattr(self.dac_params, op)

        # Use the same DAC for all arrays in bitsliced core
        dac = getattr(self.dac, op)
        adcs = [None] * Nslices
        for i in range(Nslices):
            adcs[i] = getattr(self.adcs[i], op)

        #########

        # Whether positive and negative weights in balanced configuration are subtracted digitally
        digital_posneg = (
            self.balanced
            and not self.subtract_current_in_xbar
            and not self.interleaved_posneg
        )

        Wrange_xbar = self.params.xbar.device.Grange_norm
        output_slices = [None for i in range(Nslices)]
        if digital_posneg:
            output_slices_pos = [None for i in range(Nslices)]
            output_slices_neg = [None for i in range(Nslices)]

        ################################
        ##  ANALOG INPUT ENCODING
        ################################
        if not dac_params.input_bitslicing:
            for i in range(Nslices):
                if not self.balanced:
                    output_slices[i] = getattr(self.core_slices[i][0], function)()

                    if self.clip_Icol:
                        output_slices[i] = output_slices[i].clip(
                            -self.Icol_max,
                            self.Icol_max,
                        )

                else:
                    if self.params.simulation.fast_balanced:
                        if op == "mvm":
                            output_slices[i] = xp.dot(self.W_balanced[i], vector)
                        else:
                            output_slices[i] = xp.dot(vector, self.W_balanced[i])
                    else:
                        if not self.interleaved_posneg:
                            output_pos = getattr(self.core_slices[i][0], function)()
                            output_neg = getattr(self.core_slices[i][1], function)()

                            if self.clip_Icol:
                                output_pos = output_pos.clip(
                                    -self.Icol_max,
                                    self.Icol_max,
                                )
                                output_neg = output_neg.clip(
                                    -self.Icol_max,
                                    self.Icol_max,
                                )

                            if self.subtract_current_in_xbar:
                                output_slices[i] = output_pos - output_neg
                            else:
                                output_slices_pos[i] = output_pos
                                output_slices_neg[i] = output_neg

                        else:
                            output_slices[i] = getattr(
                                self.core_slices[i][0],
                                function,
                            )(core_neg=self.core_slices[i][1])
                            if self.clip_Icol:
                                output_slices[i] = output_slices[i].clip(
                                    -self.Icol_max,
                                    self.Icol_max,
                                )

                if self.params.simulation.analytics.profile_adc_inputs:
                    i1 = self.i_op * self.outputs_per_op
                    i2 = i1 + self.outputs_per_op
                    if not digital_posneg:
                        self.adc_inputs[i, 0, i1:i2] = xp.array(
                            output_slices[i].flatten(),
                            dtype=xp.float32,
                        )
                    else:
                        self.adc_inputs[i, 0, i1:i2] = xp.array(
                            xp.concatenate(
                                (
                                    output_slices_pos[i].flatten(),
                                    output_slices_neg[i].flatten(),
                                ),
                            ),
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
                for i in range(Nslices):
                    if not self.balanced:
                        output_i_k = getattr(self.core_slices[i][0], function)(
                            vector=vector_slices[k],
                        )

                        if self.clip_Icol:
                            output_i_k = output_i_k.clip(-self.Icol_max, self.Icol_max)

                    else:
                        if self.params.simulation.fast_balanced:
                            if op == "mvm":
                                output_i_k = xp.dot(
                                    self.W_balanced[i],
                                    vector_slices[k],
                                )
                            else:
                                output_i_k = xp.dot(
                                    vector_slices[k],
                                    self.W_balanced[i],
                                )

                        else:
                            if not self.interleaved_posneg:
                                output_i_k_pos = getattr(
                                    self.core_slices[i][0],
                                    function,
                                )(vector=vector_slices[k])
                                output_i_k_neg = getattr(
                                    self.core_slices[i][1],
                                    function,
                                )(vector=vector_slices[k])

                                if self.clip_Icol:
                                    output_i_k_pos = output_i_k_pos.clip(
                                        -self.Icol_max,
                                        self.Icol_max,
                                    )
                                    output_i_k_neg = output_i_k_neg.clip(
                                        -self.Icol_max,
                                        self.Icol_max,
                                    )

                                if self.subtract_current_in_xbar:
                                    output_i_k = output_i_k_pos - output_i_k_neg

                            else:
                                output_i_k = getattr(self.core_slices[i][0], function)(
                                    vector=vector_slices[k],
                                    core_neg=self.core_slices[i][1],
                                )

                                if self.clip_Icol:
                                    output_i_k = output_i_k.clip(
                                        -self.Icol_max,
                                        self.Icol_max,
                                    )

                    if self.params.simulation.analytics.profile_adc_inputs:
                        i1 = self.i_op * self.outputs_per_op
                        i2 = i1 + self.outputs_per_op
                        if not digital_posneg:
                            self.adc_inputs[i, k, i1:i2] = xp.array(
                                output_i_k.flatten(),
                                dtype=xp.float32,
                            )
                        else:
                            self.adc_inputs[i, k, i1:i2] = xp.array(
                                xp.concatenate(
                                    (
                                        output_i_k_pos.flatten(),
                                        output_i_k_neg.flatten(),
                                    ),
                                ),
                                dtype=xp.float32,
                            )

                    # ADC
                    if self.adc_per_ibit:
                        if not digital_posneg:
                            output_i_k = adcs[i].convert(output_i_k)
                        else:
                            output_i_k_pos = adcs[i].convert(output_i_k_pos)
                            output_i_k_neg = adcs[i].convert(output_i_k_neg)

                    # Accumulate results
                    if not digital_posneg:
                        if k == 0:
                            output_slices[i] = output_i_k.copy()
                        else:
                            output_slices[i] += output_i_k
                        output_slices[i] /= 2.0**slice_size
                    else:
                        if k == 0:
                            output_slices_pos[i] = output_i_k_pos.copy()
                            output_slices_neg[i] = output_i_k_neg.copy()
                        else:
                            output_slices_pos[i] += output_i_k_pos
                            output_slices_neg[i] += output_i_k_neg
                        output_slices_pos[i] /= 2.0**slice_size
                        output_slices_neg[i] /= 2.0**slice_size

            # Scaling correction
            for i in range(Nslices):
                if not digital_posneg:
                    output_slices[i] *= (2**slice_size) - 1
                    output_slices[i] *= 2 ** (slice_size * num_input_slices - magbits)
                    output_slices[i] *= pow(2, magbits) / (pow(2, magbits) - 1)
                else:
                    output_slices_pos[i] *= (2**slice_size) - 1
                    output_slices_pos[i] *= 2 ** (
                        slice_size * num_input_slices - magbits
                    )
                    output_slices_pos[i] *= pow(2, magbits) / (pow(2, magbits) - 1)
                    output_slices_neg[i] *= (2**slice_size) - 1
                    output_slices_neg[i] *= 2 ** (
                        slice_size * num_input_slices - magbits
                    )
                    output_slices_neg[i] *= pow(2, magbits) / (pow(2, magbits) - 1)

        self.i_op += 1

        # Clip and quantize result
        if not self.adc_per_ibit:
            if not digital_posneg:
                for i in range(Nslices):
                    output_slices[i] = adcs[i].convert(output_slices[i])
            else:
                for i in range(Nslices):
                    output_slices_pos[i] = adcs[i].convert(output_slices_pos[i])
                    output_slices_neg[i] = adcs[i].convert(output_slices_neg[i])
                    output_slices[i] = output_slices_pos[i] - output_slices_neg[i]
        else:
            if digital_posneg:
                for i in range(Nslices):
                    output_slices[i] = output_slices_pos[i] - output_slices_neg[i]

        # Account for finite on-off ratio
        if self.Gmin_norm > 0:
            if self.balanced or (not self.balanced and not self.digital_offset):
                for i in range(Nslices):
                    output_slices[i] /= Wrange_xbar
            else:
                # If using digital offset, a Gmin offset has to be subtracted, and the offset is input dependent
                if (
                    self.params.simulation.convolution.x_par > 1
                    or self.params.simulation.convolution.y_par > 1
                ):
                    x_par = self.params.simulation.convolution.x_par
                    y_par = self.params.simulation.convolution.y_par
                    Noutputs = self.W_shape[0]
                    x_reshape = vector.reshape(
                        (x_par * y_par, len(vector) // (x_par * y_par)),
                    )
                    Gmin_bias = self.Gmin_norm * np.sum(x_reshape, axis=1)
                    Gmin_bias = xp.repeat(Gmin_bias, Noutputs)
                else:
                    if len(vector.shape) == 1:
                        Gmin_bias = self.Gmin_norm * xp.sum(vector)
                    else:
                        if op == "mvm":
                            Gmin_bias = self.Gmin_norm * xp.sum(vector, axis=0)
                        else:
                            Gmin_bias = self.Gmin_norm * xp.sum(vector, axis=1)[:, None]
                for i in range(Nslices):
                    output_slices[i] = (output_slices[i] - Gmin_bias) / Wrange_xbar

        # Analog offset subtraction: done before aggregation
        if not self.balanced and not self.digital_offset:
            for i in range(Nslices):
                if (
                    self.params.simulation.convolution.x_par > 1
                    or self.params.simulation.convolution.y_par > 1
                ):
                    x_par = self.params.simulation.convolution.x_par
                    y_par = self.params.simulation.convolution.y_par
                    output_i = output_slices[i].reshape(
                        (x_par * y_par, len(output_slices[i]) // (x_par * y_par)),
                    )
                    for m in range(x_par * y_par):
                        output_i[m, 1:] = output_i[m, 1:] - output_i[m, 0]
                    output_slices[i] = output_i[:, 1:].flatten()
                else:
                    if len(vector.shape) == 1:
                        output_slices[i] = output_slices[i][1:] - output_slices[i][0]
                    else:
                        if op == "mvm":
                            output_slices[i] = (
                                output_slices[i][1:, :] - output_slices[i][0, :]
                            )
                        else:
                            output_slices[i] = (
                                output_slices[i][:, 1:] - output_slices[i][:, 0]
                            )

        # Aggregate bit slices
        output = output_slices[0]
        for i in range(1, Nslices):
            output += pow(2, i * Wbits_slice) * output_slices[i]

        output *= pow(2, Wbits_slice) - 1

        # Digital offset subtraction: done after aggregation
        if not self.balanced and self.digital_offset:
            if (
                self.params.simulation.convolution.x_par > 1
                or self.params.simulation.convolution.y_par > 1
            ):
                x_par = self.params.simulation.convolution.x_par
                y_par = self.params.simulation.convolution.y_par
                Noutputs = self.W_shape[0]
                x_reshape = vector.reshape(
                    (x_par * y_par, len(vector) // (x_par * y_par)),
                )
                offset = self.Woffset * xp.sum(x_reshape, axis=1)
                offset = xp.repeat(offset, Noutputs)
                output -= offset
            else:
                if len(vector.shape) == 1:
                    output -= self.Woffset * xp.sum(vector)
                else:
                    if op == "mvm":
                        output -= self.Woffset * xp.sum(vector, axis=0)
                    else:
                        output -= self.Woffset * xp.sum(vector, axis=1)[:, None]

        output /= self.Wmax
        output /= (pow(2, Wbits - 1) - 1) / pow(2, Wbits - 1)

        if self.Gmin_norm > 0:
            output *= Wrange_xbar

        return output

    def _wrapper_read_matrix(self):
        Nslices = self.params.core.bit_sliced.num_slices
        Wbits_slice = self.Wbits_slice
        if not self.balanced:
            W = self.core_slices[0][0]._read_matrix()
        else:
            W = (
                self.core_slices[0][0]._read_matrix()
                - self.core_slices[0][1]._read_matrix()
            )

        for i in range(1, Nslices):
            if not self.balanced:
                W += pow(2, i * Wbits_slice) * self.core_slices[i][0]._read_matrix()
            else:
                W += pow(2, i * Wbits_slice) * (
                    self.core_slices[i][0]._read_matrix()
                    - self.core_slices[i][1]._read_matrix()
                )

        W *= pow(2, Wbits_slice) - 1
        if not self.balanced:
            W -= self.Woffset
        W /= self.Wmax
        return W

    def _wrapper_save_matrix(self):
        if not self.balanced:
            W_list = np.concatenate(
                (
                    self.core_slices[0][0]._save_matrix(),
                    self.core_slices[1][0]._save_matrix(),
                ),
            )
        else:
            W_list = np.concatenate(
                (
                    self.core_slices[0][0]._save_matrix(),
                    self.core_slices[0][1]._save_matrix(),
                ),
            )
            W_list = np.concatenate((W_list, self.core_slices[1][0]._save_matrix()))
            W_list = np.concatenate((W_list, self.core_slices[1][1]._save_matrix()))
        if self.Nslices > 2:
            for i in range(2, self.Nslices):
                if not self.balanced:
                    W_list = np.concatenate(
                        (W_list, self.core_slices[i][0]._save_matrix()),
                    )
                else:
                    W_list = np.concatenate(
                        (W_list, self.core_slices[i][0]._save_matrix()),
                    )
                    W_list = np.concatenate(
                        (W_list, self.core_slices[i][1]._save_matrix()),
                    )
        return W_list

    def _wrapper_restore_matrix(self, matrix):
        raise ValueError("Not implemented")

    def expand_matrix(self, Ncopy):
        # Calls expand_matrix in the inner cores
        # Makes multiple copies of matrix to compute multiple MVMs in parallel
        for i in range(self.Nslices):
            if not self.balanced:
                self.core_slices[i][0].expand_matrix(Ncopy)
            else:
                if not self.params.simulation.fast_balanced:
                    self.core_slices[i][0].expand_matrix(Ncopy)
                    self.core_slices[i][1].expand_matrix(Ncopy)
                else:
                    if not self.params.simulation.convolution.weight_reorder:
                        Nx, Ny = self.W_balanced[i].shape
                        W_i_temp = self.W_balanced[i].copy()
                        self.W_balanced[i] = xp.zeros(
                            (Ncopy * Nx, Ncopy * Ny),
                            dtype=self.W_balanced[i].dtype,
                        )
                        for m in range(Ncopy):
                            x_start, x_end = m * Nx, (m + 1) * Nx
                            y_start, y_end = m * Ny, (m + 1) * Ny
                            self.W_balanced[i][
                                x_start:x_end,
                                y_start:y_end,
                            ] = W_i_temp.copy()
                    else:
                        self.W_balanced[i] = self.core_slices[i][0].weight_reorder(
                            self.W_balanced[i].copy(),
                        )

    def unexpand_matrix(self):
        # Calls unexpand_matrix in the inner cores
        for i in range(self.Nslices):
            if not self.balanced:
                self.core_slices[i][0].unexpand_matrix()
            else:
                if not self.params.simulation.fast_balanced:
                    self.core_slices[i][0].unexpand_matrix()
                    self.core_slices[i][1].unexpand_matrix()
                else:
                    self.W_balanced[i] = self.W_balanced[i][
                        : self.W_shape[0],
                        : self.W_shape[1],
                    ]
