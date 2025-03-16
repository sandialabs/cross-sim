#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from abc import ABCMeta
from . import ICore
from simulator.parameters.core_parameters import CoreStyle
from simulator.devices.device import Device
from simulator.circuits.array import *
import numpy.typing as npt
import typing

from simulator.backend import ComputeBackend

xp = ComputeBackend()  # Represents either cupy or numpy


class NumericCore(ICore, metaclass=ABCMeta):
    """An inner :py:class:`.ICore` that performs purely-numeric calculations."""

    def __init__(self, params):
        self.matrix = None
        self.vector_vmm = None
        self.vector_mvm = None
        self.params = params

        # Device and DAC created in numeric core
        # ADC belongs to wrapper core
        self.device = Device.create_device(params.xbar.device)

        if self.params.core.style != CoreStyle.OFFSET:
            self.interleaved = self.params.core.balanced.interleaved_posneg
        else:
            self.interleaved = False
        self.current_from_input = self.params.xbar.array.parasitics.current_from_input

        if self.params.simulation.fast_matmul:
            self.Ncopy = 1
        else:
            self.Ncopy = (
                self.params.simulation.convolution.x_par
                * self.params.simulation.convolution.y_par
            )
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
            if not self.interleaved and self.current_from_input:
                self.circuit_solver_mvm = NonInterleaved_InputSource_Array(params)
                self.circuit_solver_vmm = NonInterleaved_InputSource_Array(params)
            elif not self.interleaved and not self.current_from_input:
                self.circuit_solver_mvm = NonInterleaved_SeparateSource_Array(params)
                self.circuit_solver_vmm = NonInterleaved_SeparateSource_Array(params)
            elif self.interleaved and self.current_from_input:
                self.circuit_solver_mvm = Interleaved_InputSource_Array(params)
                self.circuit_solver_vmm = Interleaved_InputSource_Array(params)
            elif self.interleaved and not self.current_from_input:
                self.circuit_solver_mvm = Interleaved_SeparateSource_Array(params)
                self.circuit_solver_vmm = Interleaved_SeparateSource_Array(params)

    def set_matrix(self, matrix, error_mask=None):
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

    def set_vmm_inputs(self, vector):
        self.vector_vmm = vector

    def set_mvm_inputs(self, vector):
        self.vector_mvm = vector

    def run_xbar_vmm(
        self,
        vector: typing.Optional[npt.NDArray] = None,
        core_neg: "NumericCore" = None,
    ) -> npt.NDArray:

        if vector is None:
            vector = self.vector_vmm

        row_in = False

        # Apply read noise (unique noise on each call)
        matrix = self.read_noise_matrix(vector=vector, row_in=row_in)
        if self.interleaved:
            matrix_neg = core_neg.read_noise_matrix(vector=vector, row_in=row_in)
        else:
            matrix_neg = None

        circuit_solver = self.circuit_solver_vmm
        op_pair = (vector, matrix)

        return self.run_xbar_operation(
            matrix,
            vector,
            op_pair,
            circuit_solver,
            row_in,
            matrix_neg,
        )

    def run_xbar_mvm(
        self,
        vector: typing.Optional[npt.NDArray] = None,
        core_neg: "NumericCore" = None,
    ) -> npt.NDArray:

        if vector is None:
            vector = self.vector_mvm

        row_in = True

        # Apply read noise (unique noise on each call)
        matrix = self.read_noise_matrix(vector=vector, row_in=row_in)
        if self.interleaved:
            matrix_neg = core_neg.read_noise_matrix(vector=vector, row_in=row_in)
        else:
            matrix_neg = None

        circuit_solver = self.circuit_solver_mvm
        op_pair = (matrix, vector)

        return self.run_xbar_operation(
            matrix,
            vector,
            op_pair,
            circuit_solver,
            row_in,
            matrix_neg,
        )

    def run_xbar_operation(
        self,
        matrix,
        vector,
        op_pair,
        circuit_solver,
        row_in,
        matrix_neg,
    ):
        input_dim = len(vector.shape)

        if self.simulate_parasitics and vector.any():

            # Only create parasitics mask as needed, to avoid unnecessary storage of the VMM mask
            # if only the MVM mask is needed, and vice versa
            if self.Ncopy > 1 and not circuit_solver.useMask:
                circuit_solver._create_parasitics_mask(self.matrix_original, self.Ncopy)

            result = circuit_solver.iterative_solve(
                matrix.copy(),
                vector,
                row_in=row_in,
                matrix_neg=matrix_neg,
            )

        elif len(matrix.shape) > 2:
            # Matmul read noise without parasitics
            #   Note: the different handling of 2D and 3D input is a consequence of how
            #   the dimensions need to be ordered for the baseline xp.matmul() call
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
                result = xp.sum(matrix * vector, axis=1) - xp.sum(matrix_neg * vector, axis=1)
            else:
                # Non-interleaved
                result = xp.sum(matrix * vector, axis=1)
            result = xp.transpose(result, (2, 0, 1))

            if input_dim == 2:
                result = result[:,:,0]
                if row_in:
                    result = result.transpose()
            if input_dim == 3 and not row_in:
                result = xp.transpose(result, (0, 2, 1))

        else:
            if matrix_neg is not None:
                # Interleaved without parasitics: identical to normal balanced core operation
                if row_in:
                    result = xp.matmul(*op_pair) - xp.matmul(matrix_neg, vector)
                else:
                    result = xp.matmul(*op_pair) - xp.matmul(vector, matrix_neg)
            else:
                # Compute using matrix vector dot product
                result = xp.matmul(*op_pair)

        return result

    def _read_matrix(self):
        return self.matrix.copy()

    def _save_matrix(self):
        return self.matrix.copy()

    def _restore_matrix(self, matrix):
        self.matrix = matrix.copy()

    def read_noise_matrix(self, vector=None, row_in=True) -> npt.NDArray:
        """Applies noise to a weight matrix, accounting for whether the matrix inclues replicated weights."""

        if self.Ncopy == 1:

            # If input is 1D, just apply read noise to the conductance matrix
            if (
                len(vector.shape) == 1
                or not self.params.xbar.device.read_noise.enable
                or self.params.xbar.device.read_noise.model == "IdealDevice"
            ):
                return self.device.read_noise(self.matrix.copy())

            # Batched read noise mode (used only if fast_matmul = True)
            # If input is 2D or 3D, make a copy of the conductance matrix for each
            # MVM so that read noise can be applied independently (but sampled in
            # parallel) for each MVM
            # The resulting 4D matrix has the same shape as the matrix that would
            # otherwise be created in the parasitics solver, so the replicated and noised
            # matrix can be passed directly into the parasitics solver
            else:
                if len(vector.shape) == 2:
                    input_index = (1 if row_in else 0)
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

        # If doing a circuit simulation, must keep the full sized block diagonal matrix
        elif self.Ncopy > 1 and self.simulate_parasitics:
            noisy_matrix = self.device.read_noise(self.matrix.copy())
            circuit_solver = (self.circuit_solver_mvm if row_in else self.circuit_solver_vmm)
            if not circuit_solver.useMask:
                circuit_solver._create_parasitics_mask(self.matrix_original, self.Ncopy)
            noisy_matrix *= circuit_solver.mask

        # If not parasitic and Ncopy > 1
        else:
            if not self.params.xbar.device.read_noise.enable:
                return self.matrix
            else:
                noisy_matrix = self.device.read_noise(self.matrix_dense.copy())
                Nx, Ny = self.matrix_original.shape
                for m in range(self.Ncopy):
                    x_start, y_start = m * Nx, m * Ny
                    x_end, y_end = x_start + Nx, y_start + Ny
                    self.matrix[x_start:x_end, y_start:y_end] = noisy_matrix[m, :, :]
                noisy_matrix = self.matrix

        return noisy_matrix

    def expand_matrix(self, Ncopy):
        """Makes a big matrix containing M copies of the weight matrix so that multiple VMMs can be computed in parallel, SIMD style
        Off-diagonal blocks of this matrix are all zero
        If noise is enabled, additionally create a third matrix that contains all the nonzero elements of this big matrix
        Intended for GPU use only, designed for neural network inference.
        """
        # Keep a copy of original matrix, both for construction of the expanded matrix and as a backup for later restoration if needed

        Nx, Ny = self.matrix.shape

        # Keep a copy of the original un-expanded matrix so that it can be restored with unexpand_matrix
        self.matrix_original = self.matrix.copy()

        if not self.params.xbar.device.read_noise.enable:
            self.matrix = xp.zeros(
                (Ncopy * Nx, Ncopy * Ny),
                dtype=self.matrix.dtype,
            )
            for m in range(Ncopy):
                x_start, x_end = m * Nx, (m + 1) * Nx
                y_start, y_end = m * Ny, (m + 1) * Ny
                self.matrix[x_start:x_end, y_start:y_end] = self.matrix_original

        else:
            # Block diagonal matrix for running MVMs
            self.matrix = xp.zeros((Ncopy * Nx, Ncopy * Ny), dtype=self.matrix.dtype)
            # Dense matrix with the same number of non-zeros as the block diagonal for applying read noise
            self.matrix_dense = xp.zeros((Ncopy, Nx, Ny), dtype=self.matrix.dtype)
            for m in range(Ncopy):
                x_start, x_end = m * Nx, (m + 1) * Nx
                y_start, y_end = m * Ny, (m + 1) * Ny
                self.matrix[x_start:x_end, y_start:y_end] = self.matrix_original
                self.matrix_dense[m, :, :] = self.matrix_original

    def unexpand_matrix(self):
        """Undo the expansion operation in expand_matrix."""
        if hasattr(self, "matrix_original"):
            self.matrix = self.matrix_original.copy()
            self.matrix_dense = None
