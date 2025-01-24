#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np
from .wrapper_core import WrapperCore
from simulator.parameters.core_parameters import OffsetCoreStyle
from simulator.circuits.adc.adc import ADC
from simulator.circuits.dac.dac import DAC
from simulator.backend import ComputeBackend

xp = ComputeBackend()

import numpy.typing as npt
from typing import Callable
from . import NumericCore


class OffsetCore(WrapperCore):
    """An offset core consisting of a single inner core.

    The middle value in the dynamic range of the inner core is used to store :code:`0.0`.

    """

    def __init__(self, clipper_core_factory: Callable[[], NumericCore], params) -> None:
        WrapperCore.__init__(self, clipper_core_factory, params)

        self.core = clipper_core_factory()

        # Whether to subtract an offset value in digital or in analog
        self.digital_offset = (
            self.params.core.offset.style == OffsetCoreStyle.DIGITAL_OFFSET
        )
        self.adc_params = self.params.xbar.adc
        self.dac_params = self.params.xbar.dac
        self.device_params = params.xbar.device
        self.input_params = self.params.core.mapping.inputs
        self.analytics_params = self.params.simulation.analytics
        self.Icol_max = self.params.xbar.array.Icol_max
        self.clip_Icol = self.params.xbar.array.Icol_max > 0

        # Create ADC
        self.adc = ADC(
            self.adc_params,
            self.dac_params,
            self.params.core,
            self.params.simulation,
        )
        self.dac = DAC(self.dac_params, self.params.core)

        self.i_op = 0

    def _wrapper_set_matrix(
        self,
        matrix: npt.NDArray,
        weight_limits=None,
        error_mask=None,
    ):
        matrix_norm = matrix / (2 * self.max)

        # To get optimal accuracy with OffsetCore, we need to ensure that the # utilized cell
        # conductance levels is odd. This ensures that there will be a conductance level that
        # corresponds to a weight value of zero. Since the actual number of levels is even, we
        # have to make sure the bottom level is unused.
        # This is done by very slightly compressing the xbar conductance range.

        if self.device_params.cell_bits > 0:
            self.Wrange_xbar = (
                self.params.xbar.device.Grange_norm
                * (2 ** (self.device_params.cell_bits) - 2)
                / (2 ** (self.device_params.cell_bits) - 1)
            )
            self.Gmin_norm = self.device_params.Gmax_norm - self.Wrange_xbar
        else:
            self.Wrange_xbar = self.params.xbar.device.Grange_norm
            self.Gmin_norm = self.device_params.Gmin_norm

        if not self.digital_offset:
            # Zero-point column
            # This currently is only compatible with MVM, since it adds a column and not a row
            # For some reason, cupy does not have an insert method
            if self.params.simulation.useGPU:
                matrix_norm = xp.asarray(np.insert(matrix_norm.get(), 0, 0, axis=0))
            else:
                matrix_norm = np.insert(matrix_norm, 0, 0, axis=0)

        matrix_norm = self.Wrange_xbar * (matrix_norm + 0.5) + self.Gmin_norm
        self.W_shape = matrix_norm.shape
        self.core.set_matrix(matrix_norm, error_mask=error_mask)

        # ADC range options
        if self.adc_params.mvm.bits > 0 or self.adc_params.vmm.bits > 0:
            self.adc.set_limits(matrix)

        self.dac.set_limits(matrix)

        # Profiling of ADC inputs
        if (
            self.analytics_params.profile_adc_inputs
            and self.dac_params.mvm.input_bitslicing
        ):
            if self.dac_params.mvm.input_bitslicing:
                magbits = self.params.xbar.dac.mvm.bits
                if self.params.xbar.dac.mvm.signed:
                    magbits -= 1
            else:
                magbits = 1
            Nout_mvm = matrix.shape[0]
            Nmvms = self.analytics_params.ntest
            # For convolutions, the size of the second dimension will be further scaled on the first
            # mvm call, when the number of sliding windows per input is known
            self.adc_inputs = xp.zeros((magbits, Nmvms * Nout_mvm), dtype=xp.float32)
            self.last_adc_input = 0

    def _wrapper_set_vmm_inputs(self, vector: npt.NDArray):
        vec_dac = self.dac.vmm.convert(vector)
        self.core.set_vmm_inputs(vec_dac)

    def _wrapper_set_mvm_inputs(self, vector: npt.NDArray):
        vec_dac = self.dac.mvm.convert(vector)
        self.core.set_mvm_inputs(vec_dac)

    def _wrapper_run_xbar_vmm(self):
        vector = self.core.vector_vmm
        op = "vmm"
        return self.run_xbar_operation(op, vector)

    def _wrapper_run_xbar_mvm(self):
        op = "mvm"
        vector = self.core.vector_mvm
        return self.run_xbar_operation(op, vector)

    def run_xbar_operation(self, op: str, vector: npt.NDArray) -> npt.NDArray:
        """Run the specified operation (vmm or mvm) with the supplied vector.

        Args:
            op: Specifies if it is a vmm or mvm
            vector: Input vector

        Returns:
            npt.NDArray: The vector output of the operation
        """
        function = "run_xbar_" + op
        core_operation = getattr(self.core, function)
        adc_params = getattr(self.adc_params, op)
        dac_params = getattr(self.dac_params, op)
        adc = getattr(self.adc, op)
        dac = getattr(self.dac, op)

        # Update the dimensions of adc_inputs as soon as # windows is known
        if (
            self.analytics_params.profile_adc_inputs
            and self.last_adc_input == 0
            and self.params.simulation.convolution.is_conv_core
        ):
            self.adc_inputs = xp.zeros(
                (
                    self.adc_inputs.shape[0],
                    self.adc_inputs.shape[1]
                    * self.params.simulation.convolution.Nwindows,
                ),
                dtype=xp.float32,
            )

        ################################
        ##  ANALOG INPUT ENCODING
        ################################
        if not dac_params.input_bitslicing:
            output = core_operation(vector)
            if self.clip_Icol:
                output = output.clip(-self.Icol_max, self.Icol_max)

            # ADC input profiling
            if self.analytics_params.profile_adc_inputs:
                num_inputs = output.size
                i1 = self.last_adc_input
                i2 = self.last_adc_input + num_inputs
                self.adc_inputs[0, i1:i2] = xp.array(output.flatten(), dtype=xp.float32)

        ################################
        ##  INPUT BIT SLICING
        ################################
        else:
            # Input bit slicing (bit serial)
            vector_slices = dac.convert_sliced(vector)
            num_input_slices = len(vector_slices)
            slice_size = dac_params.slice_size
            magbits = dac_params.bits
            if dac_params.signed:
                magbits -= 1

            for k in range(num_input_slices):
                output_k = core_operation(vector_slices[k])

                # Clip the accumulated current on each column
                # This clipping is done by the analog integrator rather than by the ADC
                if self.clip_Icol:
                    output_k = output_k.clip(-self.Icol_max, self.Icol_max)

                if self.analytics_params.profile_adc_inputs:
                    num_inputs = output_k.size
                    i1 = self.last_adc_input
                    i2 = self.last_adc_input + num_inputs
                    self.adc_inputs[k, i1:i2] = xp.array(
                        output_k.flatten(), dtype=xp.float32,
                    )

                # ADC
                if adc_params.adc_per_ibit:
                    output_k = adc.convert(output_k)

                if k == 0:
                    output = output_k.copy()
                else:
                    output += output_k

                # Charge division or shift right
                output /= 2.0**slice_size

            # Scaling correction
            output *= (2**slice_size) - 1
            output *= 2 ** (slice_size * num_input_slices - magbits)
            output *= pow(2, magbits) / (pow(2, magbits) - 1)

        self.i_op += 1

        if self.analytics_params.profile_adc_inputs:
            self.last_adc_input += num_inputs

        ##### Quantize and subtract offset

        # clip and quantize result
        if not adc_params.adc_per_ibit:
            output = adc.convert(output)

        if self.digital_offset:
            # Subtract bias offset
            half = self.Gmin_norm + 0.5 * self.Wrange_xbar
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
                offset = half * xp.sum(x_reshape, axis=1)
                offset = xp.repeat(offset, Noutputs)
                output -= offset
            else:
                if len(vector.shape) == 1:
                    output -= half * xp.sum(vector)
                else:
                    if op == "mvm":
                        if len(vector.shape) == 3:
                            output -= half * xp.sum(vector, axis=1)[:, None, :]
                        else:
                            output -= half * xp.sum(vector, axis=0)
                    else:
                        if len(vector.shape) == 3:
                            output -= half * xp.sum(vector, axis=2)[:, :, None]
                        else:
                            output -= half * xp.sum(vector, axis=1)[:, None]
        else:
            if (
                self.params.simulation.convolution.x_par > 1
                or self.params.simulation.convolution.y_par > 1
            ):
                x_par = self.params.simulation.convolution.x_par
                y_par = self.params.simulation.convolution.y_par
                output = output.reshape((x_par * y_par, len(output) // (x_par * y_par)))
                for m in range(x_par * y_par):
                    output[m, 1:] = output[m, 1:] - output[m, 0]
                output = output[:, 1:].flatten()
            else:
                if len(vector.shape) == 1:
                    output = output[1:] - output[0]
                else:
                    if op == "mvm":
                        if len(output.shape) == 3:
                            output = output[:, 1:, :] - output[:, 0, :][:, None, :]
                        elif len(output.shape) < 3:
                            output = output[1:, :] - output[0, :]
                    else:
                        if len(output.shape) == 3:
                            output = output[:, :, 1:] - output[:, :, 0][:, :, None]
                        elif len(output.shape) < 3:
                            output = output[:, 1:] - output[:, 0]

        return output

    def _wrapper_read_matrix(self):
        output = self.core._read_matrix()
        output = output.copy()
        if not self.digital_offset:
            output_shifted = output[1:, :]
            offset = output[0, :]
            output_shifted = (output_shifted - self.Gmin_norm) / self.Wrange_xbar
            offset = (offset - self.Gmin_norm) / self.Wrange_xbar
            output = output_shifted - offset
        else:
            output = (output - self.Gmin_norm) / self.Wrange_xbar - 0.5

        # Unexpand the matrix
        if self.params.simulation.disable_fast_matmul:
            output = output[: self.W_shape[0], : self.W_shape[1]]
        output *= 2 * self.max
        return output

    def _wrapper_save_matrix(self):
        output = self.core._save_matrix()
        return output.copy()

    def _wrapper_restore_matrix(self, matrix):
        return self.core._restore_matrix(matrix)

    def expand_matrix(self, Ncopy):
        # Calls expand_matrix in the inner cores
        # Makes multiple copies of matrix to compute multiple MVMs in parallel
        self.core.expand_matrix(Ncopy)

    def unexpand_matrix(self):
        # Calls unexpand_matrix in the inner cores
        self.core.unexpand_matrix()
