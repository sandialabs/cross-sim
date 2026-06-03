#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from abc import ABCMeta
from . import ICore
from simulator.parameters.core_parameters import CoreStyle, BitSlicedCoreStyle
from simulator.devices.device import Device
from simulator.circuits.array import (
    NoninterleavedInputSourceArray,
    NoninterleavedSeparateSourceArray,
    InterleavedInputSourceArray,
    InterleavedSeparateSourceArray,
    UncoupledNonlinearArray,
)
import numpy.typing as npt
import typing

from simulator.backend import ComputeBackend

xp = ComputeBackend()  # Represents either cupy or numpy


class NumericCore(ICore, metaclass=ABCMeta):
    """Performs purely numeric simulations of crossbar matrix multiplication."""

    def __init__(self, params):
        """Initializes the NumericCore.

        Args:
            params: Parameters for the simulation.
        """
        self.matrix = None
        self.vector_vmm = None
        self.vector_mvm = None
        self.params = params

        # Device and DAC created in numeric core
        # ADC belongs to wrapper core
        self.device = Device.create_device(params.xbar.device)

        if self.params.core.style != CoreStyle.OFFSET and not (
            self.params.core.style == CoreStyle.BITSLICED
            and self.params.core.bit_sliced.style == BitSlicedCoreStyle.OFFSET
        ):
            self.interleaved = self.params.core.balanced.interleaved_posneg
        else:
            self.interleaved = False
        self.current_from_input = self.params.xbar.array.parasitics.current_from_input
        self.read_noise_params = self.params.xbar.device.read_noise

        self.simulate_parasitics = self.params.xbar.array.parasitics.enable and (
            self.params.xbar.array.parasitics.Rp_row > 0
            or self.params.xbar.array.parasitics.Rp_col > 0
            or self.params.xbar.array.parasitics.Rp_row_terminal > 0
            or self.params.xbar.array.parasitics.Rp_col_terminal > 0
        )

        # Create parasitics solvers
        self.circuit_solver_mvm = None
        self.circuit_solver_vmm = None
        if self.simulate_parasitics:
            # These solvers can model both parasitic resistance and I-V
            # nonlinearity
            if not self.interleaved and self.current_from_input:
                self.circuit_solver_mvm = NoninterleavedInputSourceArray(
                    params, device=self.device
                )
                self.circuit_solver_vmm = NoninterleavedInputSourceArray(
                    params, device=self.device
                )
            elif not self.interleaved and not self.current_from_input:
                self.circuit_solver_mvm = NoninterleavedSeparateSourceArray(
                    params, device=self.device
                )
                self.circuit_solver_vmm = NoninterleavedSeparateSourceArray(
                    params, device=self.device
                )
            elif self.interleaved and self.current_from_input:
                self.circuit_solver_mvm = InterleavedInputSourceArray(
                    params, device=self.device
                )
                self.circuit_solver_vmm = InterleavedInputSourceArray(
                    params, device=self.device
                )
            elif self.interleaved and not self.current_from_input:
                self.circuit_solver_mvm = InterleavedSeparateSourceArray(
                    params, device=self.device
                )
                self.circuit_solver_vmm = InterleavedSeparateSourceArray(
                    params, device=self.device
                )
        elif self.params.xbar.device.nonlinear_IV.enable:
            # This solver models I-V nonlinearity only
            self.circuit_solver_mvm = UncoupledNonlinearArray(
                params, device=self.device
            )
            self.circuit_solver_vmm = UncoupledNonlinearArray(
                params, device=self.device
            )

    def set_matrix(self, matrix, error_mask=None):
        """Sets the value of the conductance matrix.

        Args:
            matrix: Conductance matrix to set
            error_mask: Boolean mask with the same shape as matrix to indicate
                which values of the matrix should have errors applied.
                Defaults to None.
        """
        if self.params.simulation.useGPU:
            self.matrix = xp.array(matrix)
        else:
            self.matrix = matrix

        # Apply weight error
        matrix_copy = self.matrix.copy()
        matrix_error = self.device.apply_write_error(matrix_copy)
        if not error_mask:
            self.matrix = matrix_error
        else:
            self.matrix = matrix_copy
            self.matrix[error_mask] = matrix_error[error_mask]

        # If using lumped read noise, compute the noise variance matrix here
        if self.read_noise_params.enable and self.read_noise_params.lumped:
            self.noise_var_matrix = self.device.read_noise_variance(self.matrix)

    def set_vmm_inputs(self, vector):
        """Sets the inputs that will be used in vector matrix multiplication.

        Args:
            vector: Input vector to set.
        """
        self.vector_vmm = vector

    def set_mvm_inputs(self, vector):
        """Sets the inputs that will be used in matrix vector multiplication.

        Args:
            vector: Input vector to set.
        """
        self.vector_mvm = vector

    def run_xbar_vmm(
        self,
        vector: typing.Optional[npt.NDArray] = None,
        core_neg: "NumericCore" = None,
    ) -> npt.NDArray:
        """Simulates a vector matrix multiplication using the crossbar.

        Args:
            vector: Vector to use. If no vector is specified then the input
                vector for mvm currently set is used instead. Defaults to None.
            core_neg: For use when interleaved option is True.

        Returns:
            npt.NDArray: Result of the matrix vector multiply using the crossbar
        """
        if vector is None:
            vector = self.vector_vmm

        row_in = False

        # Apply read noise (unique noise on each call)
        matrix = self.read_noise_matrix(vector=vector, row_in=row_in)
        matrix_neg, noise_var_neg = None, None
        if core_neg is not None:
            matrix_neg = core_neg.read_noise_matrix(vector=vector, row_in=row_in)
            if self.read_noise_params.enable and self.read_noise_params.lumped:
                noise_var_neg = core_neg.noise_var_matrix

        circuit_solver = self.circuit_solver_vmm
        op_pair = (vector, matrix)

        return self.run_xbar_operation(
            matrix,
            vector,
            op_pair,
            circuit_solver,
            row_in,
            matrix_neg,
            noise_var_neg,
        )

    def run_xbar_mvm(
        self,
        vector: typing.Optional[npt.NDArray] = None,
        core_neg: "NumericCore" = None,
    ) -> npt.NDArray:
        """Simulates a matrix vector multiplication using the crossbar.

        Args:
            vector: Vector to use. If no vector is specified then the input
                vector for mvm currently set is used instead. Defaults to None.
            core_neg: For use when interleaved option is True.

        Returns:
            npt.NDArray: Result of the matrix vector multiply using the crossbar
        """
        if vector is None:
            vector = self.vector_mvm

        row_in = True

        # Apply read noise (unique noise on each call)
        matrix = self.read_noise_matrix(vector=vector, row_in=row_in)
        matrix_neg, noise_var_neg = None, None
        if core_neg is not None:
            matrix_neg = core_neg.read_noise_matrix(vector=vector, row_in=row_in)
            if self.read_noise_params.enable and self.read_noise_params.lumped:
                noise_var_neg = core_neg.noise_var_matrix

        circuit_solver = self.circuit_solver_mvm
        op_pair = (matrix, vector)

        return self.run_xbar_operation(
            matrix,
            vector,
            op_pair,
            circuit_solver,
            row_in,
            matrix_neg,
            noise_var_neg,
        )

    def run_xbar_operation(  # noqa:C901
        self,
        matrix,
        vector,
        op_pair,
        circuit_solver,
        row_in,
        matrix_neg,
        noise_var_neg,
    ):
        """A generalized functino to perform cross bar multiplications."""
        input_dim = len(vector.shape)

        if (
            self.simulate_parasitics or self.params.xbar.device.nonlinear_IV.enable
        ) and vector.any():
            result = circuit_solver.iterative_solve(
                matrix.copy(),
                vector,
                row_in=row_in,
                matrix_neg=matrix_neg,
            )

        elif len(matrix.shape) > 2:
            # Matmul read noise without parasitics
            #   Note: the different handling of 2D and 3D input is a consequence
            #   of how the dimensions need to be ordered for the baseline
            #   xp.matmul() call
            if input_dim == 2:
                if row_in:
                    vector = vector.transpose()
                vector = vector[:, :, xp.newaxis]
            if input_dim == 3 and not row_in:
                vector = xp.transpose(vector, (0, 2, 1))

            vector = xp.transpose(vector, (1, 2, 0))
            vector = xp.tile(vector, (matrix.shape[0], 1, 1, 1))
            if matrix_neg is not None:
                # Interleaved
                result = xp.sum(matrix * vector, axis=1) - xp.sum(
                    matrix_neg * vector, axis=1
                )
            else:
                # Non-interleaved
                result = xp.sum(matrix * vector, axis=1)
            result = xp.transpose(result, (2, 0, 1))

            if input_dim == 2:
                result = result[:, :, 0]
                if row_in:
                    result = result.transpose()
            if input_dim == 3 and not row_in:
                result = xp.transpose(result, (0, 2, 1))

        else:
            if matrix_neg is not None:
                # Interleaved without parasitics: identical to normal balanced
                # core operation
                if row_in:
                    result = xp.matmul(*op_pair) - xp.matmul(matrix_neg, vector)
                else:
                    result = xp.matmul(*op_pair) - xp.matmul(vector, matrix_neg)
            else:
                # Compute using matrix vector dot product
                result = xp.matmul(*op_pair)

        # Apply lumped read noise
        result = self.apply_lumped_read_noise(vector, result, noise_var_neg)

        return result

    def _read_matrix(self):
        return self.matrix.copy()

    def _save_matrix(self):
        return self.matrix.copy()

    def _restore_matrix(self, matrix):
        self.matrix = matrix.copy()

    def read_noise_matrix(self, vector=None, row_in=True) -> npt.NDArray:
        """Applies noise to a weight matrix.

        This accounts for whether the matrix inclues replicated weights.
        """
        # If input is 1D, just apply read noise to the conductance matrix.
        # Also use this path if read noise is off or lumped read noise is used.
        if (
            len(vector.shape) == 1
            or not self.read_noise_params.enable
            or self.read_noise_params.model == "IdealDevice"
            or self.read_noise_params.lumped
        ):
            return self.device.read_noise(self.matrix.copy())

        # Batched read noise mode (used only if fast_matmul = True)
        # If input is 2D or 3D, make a copy of the conductance matrix for
        # each MVM so that read noise can be applied independently (but
        # sampled in parallel) for each MVM
        # The resulting 4D matrix has the same shape as the matrix that
        # would otherwise be created in the parasitics solver, so the
        # replicated and noised matrix can be passed directly into the
        # parasitics solver
        else:
            if len(vector.shape) == 2:
                input_index = 1 if row_in else 0
                expanded_matrix = xp.tile(
                    self.matrix[:, :, None], (1, 1, vector.shape[input_index])
                )[:, :, xp.newaxis, :]
            else:
                if row_in:
                    expanded_matrix = xp.tile(
                        self.matrix[:, :, None, None],
                        (1, 1, vector.shape[2], vector.shape[0]),
                    )
                else:
                    expanded_matrix = xp.tile(
                        (self.matrix.T)[:, :, None, None],
                        (1, 1, vector.shape[1], vector.shape[0]),
                    )
            return self.device.read_noise(expanded_matrix)

    def apply_lumped_read_noise(self, vector, output, noise_var_neg) -> npt.NDArray:
        """Applies lumped read noise on an array of dot products.

        Args:
            vector: Inputs to the MVM or VMM
            output: Dot product outputs of the MVM or VMM without read noise
            noise_var_neg: Matrix of read noise variances for the negative
                weights core, if interleaved_posneg=True.

        Returns:
            npt.NDArray: Dot product outputs with read noise added
        """
        if not (self.read_noise_params.enable and self.read_noise_params.lumped):
            return output

        # Compute the lumped read noise as a weighted quadrature sum of errors
        # with the input vector
        sigma_outputs = xp.sqrt(self.noise_var_matrix @ (vector**2))

        # Add lumped read noise to outputs
        output += sigma_outputs * xp.random.normal(size=output.shape).astype(
            output.dtype
        )

        # If interleaved, compute and add negative weights lumped read noise
        if noise_var_neg is not None:
            sigma_outputs_neg = xp.sqrt(noise_var_neg @ (vector**2))
            output += sigma_outputs_neg * xp.random.normal(size=output.shape).astype(
                output.dtype
            )

        return output
