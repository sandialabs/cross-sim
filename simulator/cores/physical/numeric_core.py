#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import numpy.typing as npt

from simulator.cores.interfaces.icore_internal import ICore, ICoreInternal
from simulator.parameters.crosssim import CrossSimParameters
from simulator.parameters.core import CoreMappingParameters
from simulator.circuits.array_simulator import (
    solve_mvm_circuit,
    mvm_parasitics,
    mvm_parasitics_interleaved,
)
from simulator.circuits.dac.dac import DAC
from simulator.circuits.adc.adc import ADC
from simulator.devices.device import Device
from simulator.backend import ComputeBackend
from simulator.parameters.simulation import MultiplicationFunction

xp: np = ComputeBackend()  # Represents either cupy or numpy
log = logging.getLogger(__name__)


class NumericCore(ICoreInternal):
    """Performs purely numeric simulations of crossbar matrix multiplication.

    Contains additional attribtutes and functions in addition to the ICore
    interface

    Attributes:
        adc: Analog to digital converter. Not used by the class, but made
            accessible to any parent core that interacts with a physical core
        dac: Digital to analog converter. Not used by the class, but made
            accessible to any parent core that interacts with a physical core
    """

    adc: ADC | None = None
    dac: DAC | None = None

    def __init__(
        self,
        params: CrossSimParameters,
        parent: ICore | None = None,
        key: str | None = None,
    ):
        """Initializes the NumericCore.

        Note:
            The core's adc/dac is not initialized on object instantiation. These
            objects are created when the appropriate parent core calls make_core

        Args:
            params: Parameters for the simulation.
            parent: Parent core, if applicable. Defaults to None.
            key: The core's key in the parent's subcore dictionary.
        """
        self.params = params
        self.parent = parent
        self.key = key

        self.matrix = None
        self.vector_vmm = None
        self.vector_mvm = None
        self.par_mask = None
        if (
            self.params.simulation.multiplication_function
            == MultiplicationFunction.MATMUL
        ):
            self.mult_func = xp.matmul
        else:
            self.mult_func = xp.dot

        self.xbar_params = self.params.xbar.match(self.identifier)
        self._mapping = self.generate_mapping()

        # Device created in numeric core
        # ADC and DAC are created by a parent core and given to child numeric
        # cores
        self.device = Device(self.xbar_params.device)

        # TODO: Balanced param is called interleaved_posneg
        #       Either change name in parent core or change key in child core
        self.interleaved = getattr(parent, "interleaved", False)

        self.Ncopy = 1
        if not self.params.simulation.fast_matmul:
            self.Ncopy = (
                self.params.simulation.convolution.x_par
                * self.params.simulation.convolution.y_par
            )
        self.Rp_row = self.xbar_params.array.parasitics.Rp_row
        self.Rp_col = self.xbar_params.array.parasitics.Rp_col
        self.simulate_parasitics = self.xbar_params.array.parasitics.enable and (
            self.Rp_col > 0 or self.Rp_row > 0
        )

        # Set parasitics solver functions
        # Convention: row_in == True for MVM and row_in = False for VMM
        if not self.simulate_parasitics:
            circuit_solver = None
        elif self.interleaved:
            circuit_solver = mvm_parasitics_interleaved
        else:
            circuit_solver = mvm_parasitics
        self.circuit_solver_mvm = circuit_solver
        self.circuit_solver_vmm = circuit_solver

    def set_matrix(
        self,
        matrix: npt.ArrayLike,
        apply_errors: bool = True,
        error_mask: npt.ArrayLike | None = None,
    ):
        """Sets the value of the conductance matrix.

        Args:
            matrix: Conductance matrix to set
            apply_errors: Whether to apply errors when setting the matrix.
                This option is independent of the "enable" option for the
                models found in DeviceParameters. Defaults to True.
            error_mask: Boolean mask with the same shape as matrix to indicate
                which values of the matrix should have errors applied.
                Defaults to None.
        """
        self.matrix = xp.asarray(matrix)
        self._check_matrix_bounds(self.matrix)

        # If simulating parasitics with SW packing, create a mask here
        if self.Ncopy > 1 and self.simulate_parasitics:
            self._create_par_mask()

        # Apply weight error
        if not apply_errors:
            return
        matrix_copy = self.matrix.copy()
        # TODO: Why are we making a copy?
        # Does apply write error modify in place?
        matrix_error = self.device.apply_write_error(matrix_copy)
        if error_mask is None:
            self.matrix = matrix_error
        else:
            self.matrix = matrix_copy
            self.matrix[error_mask] = matrix_error[error_mask]

    def set_vmm_inputs(self, vector: npt.ArrayLike):
        """Sets the inputs that will be used in vector matrix multiplication.

        Args:
            vector: Input vector to set.
        """
        self.vector_vmm = xp.asarray(vector)

    def set_mvm_inputs(self, vector: npt.ArrayLike):
        """Sets the inputs that will be used in matrix vector multiplication.

        Args:
            vector: Input vector to set.
        """
        self.vector_mvm = xp.asarray(vector)

    def run_xbar_vmm(
        self,
        vector: npt.ArrayLike | None = None,
        core_neg: NumericCore | None = None,
    ) -> npt.NDArray:
        """Simulates a vector matrix multiplication using the crossbar.

        Args:
            vector: Vector to use. If no vector is specified then the input
                vector for mvm currently set is used instead. Defaults to None.
            core_neg: For use when interleaved option is True.

        Returns:
            npt.NDArray: Result of the matrix vector multiply using the crossbar
        """
        # apply read noise
        matrix = self.read_noise_matrix()
        if self.interleaved:
            matrix_neg = core_neg.read_noise_matrix()
        else:
            matrix_neg = None

        if vector is not None:
            self.set_vmm_inputs(vector=vector)
        elif self.vector_vmm is None:
            # TODO: Create custom exceptions.
            raise RuntimeError(
                "VMM input never set, cannot perform xbar multiplication. "
                "Call core.set_vmm_inputs prior to multiplication.",
            )

        vector = self.vector_vmm
        circuit_solver = self.circuit_solver_vmm
        row_in = False
        op_pair = (vector, matrix)

        return self.run_xbar_operation(
            matrix=matrix,
            vector=vector,
            op_pair=op_pair,
            circuit_solver=circuit_solver,
            row_in=row_in,
            matrix_neg=matrix_neg,
        )

    def run_xbar_mvm(
        self,
        vector: npt.ArrayLike | None = None,
        core_neg: NumericCore | None = None,
    ) -> npt.NDArray:
        """Simulates a matrix vector multiplication using the crossbar.

        Args:
            vector: Vector to use. If no vector is specified then the input
                vector for mvm currently set is used instead. Defaults to None.
            core_neg: For use when interleaved option is True.

        Returns:
            npt.NDArray: Result of the matrix vector multiply using the crossbar
        """
        # Apply read noise (unique noise on each call)
        matrix = self.read_noise_matrix()
        if self.interleaved:
            matrix_neg = core_neg.read_noise_matrix()
        else:
            matrix_neg = None

        # Load input vector
        if vector is not None:
            self.set_mvm_inputs(vector=vector)
        elif self.vector_mvm is None:
            raise RuntimeError(
                "MVM input never set, cannot perform xbar multiplication. "
                "Call core.set_mvm_inputs prior to multiplication.",
            )
        vector = self.vector_mvm
        circuit_solver = self.circuit_solver_mvm
        row_in = True
        op_pair = (matrix, vector)

        return self.run_xbar_operation(
            matrix=matrix,
            vector=vector,
            op_pair=op_pair,
            circuit_solver=circuit_solver,
            row_in=row_in,
            matrix_neg=matrix_neg,
        )

    def run_xbar_operation(
        self,
        matrix: npt.NDArray,
        vector: npt.NDArray,
        op_pair: tuple[npt.NDArray, npt.NDArray],
        circuit_solver: Callable,
        row_in: bool,
        matrix_neg: npt.NDArray | None,
    ) -> npt.NDArray:
        """A generalized functino to perform cross bar multiplications."""
        if self.simulate_parasitics and vector.any():
            useMask = self.Ncopy > 1
            result = solve_mvm_circuit(
                circuit_solver=circuit_solver,
                vector=vector,
                matrix=matrix.copy(),
                simulation_params=self.params.simulation,
                xbar_params=self.xbar_params,
                interleaved=self.interleaved,
                matrix_neg=matrix_neg,
                useMask=useMask,
                mask=self.par_mask,
                row_in=row_in,
            )
        elif matrix_neg is not None:
            # Interleaved without parasitics: identical to normal
            # balanced core operation
            if row_in:
                result = self.mult_func(*op_pair) - self.mult_func(matrix_neg, vector)
            else:
                result = self.mult_func(*op_pair) - self.mult_func(vector, matrix_neg)
        else:
            # Compute using matrix vector dot product
            result = self.mult_func(*op_pair)

        # This transpose is only needed when dot (not rdot) is performed.
        # When op_pair[1].ndim is 3, that means it was dot, when op_pair[0].ndim
        # is 3, that means it was rdot
        if self.mult_func == xp.dot and op_pair[1].ndim == 3:
            result = result.transpose((1, 0, 2))
        return result

    def read_matrix(self, apply_errors: bool = True) -> npt.NDArray:
        """Read the matrix set by simulation.

        Note that if the matrix was written with errors enabled, then reading
        with apply_errors=False may still produce a matrix different than the
        value it was originally set with.

        Args:
            apply_errors: If True, the matrix will be read using the error model
                that was configured. If False, the matrix will be read without
                using the error models for reading the matrix. Defaults ot True.
        """
        if apply_errors:
            return self.read_noise_matrix()
        else:
            return self.matrix.copy()

    def read_noise_matrix(self) -> npt.NDArray:
        """Returns the stored matrix including read noise.

        Applies noise to a weight matrix, accounting for whether the matrix
        includes replicated weights
        """
        # Default code path
        if self.Ncopy == 1:
            # TODO: Why are we copying here?
            #       Does this modify the original matrix?
            #       If not, we should really think this through
            #       We don't want to copy the matrix to new memory on
            #           *every* multiply
            return self.device.read_noise(self.matrix.copy())

        # If doing a circuit simulation,
        # must keep the full sized block diagonal matrix
        elif self.Ncopy > 1 and self.simulate_parasitics:
            noisy_matrix = self.device.read_noise(self.matrix.copy())
            noisy_matrix *= self.par_mask

        # If not parasitic and Ncopy > 1
        else:
            if not self.xbar_params.device.read_noise.enable:
                return self.matrix
            else:
                # TODO: Is self.matrix_dense guaranteed to exist?
                #       It is not initialized (even as None) in __init__
                noisy_matrix = self.device.read_noise(self.matrix_dense.copy())
                Nx, Ny = self.matrix_original.shape
                for m in range(self.Ncopy):
                    x_start, y_start = m * Nx, m * Ny
                    x_end, y_end = x_start + Nx, y_start + Ny
                    self.matrix[x_start:x_end, y_start:y_end] = noisy_matrix[m, :, :]
                noisy_matrix = self.matrix

        return noisy_matrix

    def expand_matrix(self, Ncopy: int):
        """Expands the matrix to contain copies of itself.

        Allows parallel multiplications.

        Makes a big matrix containing M copies of the weight matrix so that
        multiple VMMs can be computed in parallel, SIMD style
        Off-diagonal blocks of this matrix are all zero
        If noise is enabled, additionally create a third matrix that contains
        all the nonzero elements of this big matrix
        Intended for GPU use only, designed for neural network inference
        """
        # Keep a copy of original matrix, both for construction of the expanded
        # matrix and as a backup for later restoration if needed

        Nx, Ny = self.matrix.shape

        # Keep a copy of the original un-expanded matrix so that it can be
        # restored with unexpand_matrix
        self.matrix_original = self.matrix.copy()

        if not self.xbar_params.device.read_noise.enable:
            if self.params.simulation.convolution.weight_reorder:
                self.matrix = self.weight_reorder(self.matrix_original.copy())
            else:
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
            # Dense matrix with the same number of non-zeros as the block
            # diagonal for applying read noise
            self.matrix_dense = xp.zeros((Ncopy, Nx, Ny), dtype=self.matrix.dtype)
            # TODO: Matrix dense is a conditionally defined attribute, fix
            # (@Curtis, 23-11-02)
            for m in range(Ncopy):
                x_start, x_end = m * Nx, (m + 1) * Nx
                y_start, y_end = m * Ny, (m + 1) * Ny
                self.matrix[x_start:x_end, y_start:y_end] = self.matrix_original
                self.matrix_dense[m, :, :] = self.matrix_original

    def weight_reorder(self, matrix_original: npt.NDArray) -> npt.NDArray:
        """Reorders weights for slide window packing.

        Utility function used to implement weight reordering for sliding window
        packing. This function is also used by higher cores if fast_balanced is
        equal to True

        Args:
            matrix_original: Matrix to be reordered

        Returns:
            npt.NDArray: Reordered matrix
        """
        Kx = self.params.simulation.convolution.Kx
        Ky = self.params.simulation.convolution.Ky
        Nic = self.params.simulation.convolution.Nic
        Noc = self.params.simulation.convolution.Noc
        stride = self.params.simulation.convolution.stride
        x_par = self.params.simulation.convolution.x_par  # parallel windows in x
        y_par = self.params.simulation.convolution.y_par  # parallel windows in y
        x_par_in = (x_par - 1) * stride + Kx
        y_par_in = (y_par - 1) * stride + Ky

        matrix = xp.zeros(
            (x_par * y_par * Noc, x_par_in * y_par_in * Nic),
            dtype=matrix_original.dtype,
        )
        m = 0
        for ix in range(x_par):
            for iy in range(y_par):
                for ixx in range(Kx):
                    for iyy in range(Ky):
                        # 1: Which elements of the flattened input should be
                        #    indexed for this 2D point?
                        x_coord = stride * ix + ixx
                        y_coord = stride * iy + iyy
                        row_xy = x_coord * y_par_in + y_coord
                        x_start = row_xy
                        x_end = row_xy + Nic * x_par_in * y_par_in
                        # 2: Which elements of the weight matrix are used for
                        #    this point?
                        Wx_coord = ixx * Ky + iyy
                        W_start = Wx_coord
                        W_end = Wx_coord + Nic * Kx * Ky
                        y_start, y_end = m * Noc, (m + 1) * Noc
                        matrix[
                            y_start:y_end,
                            x_start : x_end : (x_par_in * y_par_in),
                        ] = matrix_original[:, W_start : W_end : (Kx * Ky)].copy()
                m += 1

        return matrix

    def unexpand_matrix(self):
        """Undo the expansion operation in expand_matrix."""
        self.matrix = self.matrix_original.copy()
        self.matrix_dense = None

    def _check_matrix_bounds(self, matrix: npt.NDArray):
        """Raises a ValueError if matrix is outside of expected bounds."""
        gmin = self.device.Gmin_norm
        gmax = self.device.Gmax_norm
        if xp.min(matrix) < gmin or xp.max(matrix) > gmax:
            parent_fast_balanced = getattr(self.parent, "fast_balanced", False)
            if not parent_fast_balanced:
                msg = "Conductance matrix set outside of expected bounds."
                log.error(msg)
                raise ValueError(msg)
            else:
                # We expect this can happen in fast balance
                msg = (
                    "Conductance matrix set outside of expected bounds "
                    "with fast_balance enabled."
                )
                log.info(msg)

    def _create_par_mask(self):
        """If simulating parasitics with SW packing, create a mask here."""
        Nx, Ny = self.matrix.shape
        self.par_mask = xp.zeros(
            shape=(self.Ncopy * Nx, self.Ncopy * Ny),
            dtype=self.matrix.dtype,
        )
        for m in range(self.Ncopy):
            x_start, x_end = m * Nx, (m + 1) * Nx
            y_start, y_end = m * Ny, (m + 1) * Ny
            self.par_mask[x_start:x_end, y_start:y_end] = 1
        self.par_mask = self.par_mask > 1e-9

    def generate_mapping(self) -> CoreMappingParameters:
        """Generate a mapping for the core."""
        gmin = self.xbar_params.device.Gmin_norm
        gmax = self.xbar_params.device.Gmax_norm
        mapping = CoreMappingParameters(
            weights={"min": gmin, "max": gmax, "clipping": False},
            mvm={"min": -1.0, "max": 1.0, "clipping": False},
            vmm={"min": -1.0, "max": 1.0, "clipping": False},
            # TODO:
            # What would we put for inputs here?
            # Surely that's something that matters?
            # For now, I'm just going to assume the defaults of (-1, 1)
            # @Curtis
            # 2023-11-22
        )
        return mapping
