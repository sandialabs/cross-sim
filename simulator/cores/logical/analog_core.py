#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""A Numpy-like interface for interacting with analog MVM operations.

AnalogCore is the primary interface for CrossSim analog MVM simulations.
AnalogCores behave as if they are Numpy arrays for 1 and 2 dimensional matrix
multiplication (left and right), transpostion and slice operations. This is
intended to provide drop-in compatibility with existing numpy matrix
manipulation code. AnalogCore only implements functions which make physical
sense for analog MVM array, for instance element-wise operations are
intentionally not supported as they don't have an obvious physical
implementation.

Internally AnalogCore may partition a single matrix across multiple physical
arrays based on the input data types and specified parameters.
"""

from __future__ import annotations

import logging
import warnings

import numpy.typing as npt

import simulator.cores.utils as core_utils
from simulator.backend import ComputeBackend
from simulator.parameters.crosssim import CrossSimParameters
from simulator.parameters.core.analog_core import (
    AnalogCoreParameters,
    OutputDTypeStrategy,
    PartitionMode,
)
from simulator.cores.interfaces.icore import ICore
import simulator.cores.logical.utils as logical_utils
from simulator.parameters.mapping import CoreMappingParameters

xp = ComputeBackend()
log = logging.getLogger(__name__)


class AnalogCore(ICore):
    """Primary simulation object for Analog MVM.

    AnalogCore provides a numpy-like interface for matrix multiplication
    operations using analog MVM arrays. AnalogCore should be the primary
    interface for algorithms. AnalogCore internally contains multiple physical
    cores (which may corrospond to multiple discrete arrays in the case of
    balanced and bit-sliced systems). AnalogCore may also expand the provided
    matrix as needed for analog computation, for instance when using complex
    numbers.

    Attributes:
        matrix:
            Numpy-like array to be represented by this core. This sets the size
            of the array (which may be larger than a single physical array).
        params:
            A CrossSimParameters object or list of CrossSimParameters objects
            specifying the properties of the constructed AnalogCores. For
            simulations where a single matrix will be split across multiple
            physical cores this must be a list of length equal to the number of
            underlying cores.
        empty_matrix:
            Bool indicating whether to initialize the array from the input
            matrix. For creating arrays where the data isn't known yet.
    """

    core_params: AnalogCoreParameters
    complex_valued: bool
    fast_matmul: bool
    weight_clipping: bool
    rslice: slice | None = None
    cslice: slice | None = None
    nrow: int
    ncol: int
    ndim: int
    dtype: npt.DTypeLike
    shape: tuple[int, int]
    output_dtype_strategy: OutputDTypeStrategy

    # Forces numpy to use rmatmat for right multiplies
    # Alternative is subclassing ndarray which has some annoying potential side
    # effects.
    # cupy uses __array_priority__ = 100 so need to be 1 more than that
    __array_priority__ = 101

    def __init__(
        self,
        matrix: npt.ArrayLike,
        params: CrossSimParameters,
        empty_matrix=False,
    ) -> None:
        """Initializes an AnalogCore object with the provided dimension and
        parameters.

        Args:
            matrix: Matrix to initialize the array size and (optionally) data
            params: Parameters object or objects specifying the behavior of the
                object.
            empty_matrix: Bool to skip initializing array data from input matrix
        Raises:
            ValueError: Parameter object is invalid for the configuration.
        """
        params.validate()
        # Print a message if logging is not configured.
        logical_utils.check_logging(ignore_check=params.simulation.ignore_logging_check)
        super().__init__(
            xsim_parameters=params,
            core_parameters=params.core,
            parent=None,
        )

        self._reset()
        matrix = self._process_matrix(matrix)
        self._calculate_partition()

        if not empty_matrix:
            self.set_matrix(
                matrix=matrix,
                apply_errors=True,
                error_mask=None,
            )

    def _process_matrix(self, matrix: npt.ArrayLike) -> npt.NDArray:
        """Configures internal variables of the core to match the matrix.
        Returns the matrix in a form usable for the core.

        Args:
            matrix: The array to be processed.

        Raises:
            ValueError: Raised if the matrix is not 2 dimensional.

        Returns:
            npt.NDArray: The processed matrix.
        """
        matrix: npt.NDArray = xp.asarray(matrix)
        if matrix.ndim != 2:
            raise ValueError("AnalogCore must 2 dimensional")

        # Double # rows and columns for complex matrix
        nrow, ncol = matrix.shape
        if self.complex_valued:
            nrow *= 2
            ncol *= 2

        self.nrow = nrow
        self.ncol = ncol
        self.ndim = matrix.ndim
        self.dtype = matrix.dtype
        self._shape = matrix.shape
        return matrix

    def set_matrix(
        self,
        matrix: npt.ArrayLike,
        apply_errors: bool = True,
        error_mask: tuple[slice, slice] | None = None,
    ) -> None:
        """Sets the matrix that AnalogCore will use.

        Transform the input matrix as needed for programming to analog arrays
        including complex expansion, clipping, and matrix partitioning. Calls
        the set_matrix() methods of the underlying core objects

        Args:
            matrix: Matrix value to set.
            apply_errors: Whether to apply errors when setting the matrix.
                This option is independent of the "enable" option for the
                models found in DeviceParameters. Defaults to True.
            error_mask: Boolean mask with the same shape as matrix to indicate
                which values of the matrix should have errors applied.
                Defaults to None.
        """
        matrix = xp.asarray(matrix)
        if self.shape is None:
            self._process_matrix(matrix)

        if self.shape != matrix.shape:
            raise ValueError("Matrix shape must match AnalogCore shape")

        if (
            matrix.dtype == xp.complex64 or matrix.dtype == xp.complex128
        ) and not self.complex_valued:
            raise ValueError(
                (
                    "If setting complex-valued matrices, "
                    "please set core.complex_matrix = True"
                ),
            )

        self._shape = matrix.shape
        self.dtype = matrix.dtype

        # Break up complex matrix into real and imaginary quadrants
        mcopy = logical_utils.reshaped_complex_matrix(
            matrix=matrix,
            is_complex=self.complex_valued,
        )

        # For partial matrix updates new values must be inside the previous
        # range. If the values would exceed this range then you would have to
        # reprogram all matrix values based on the new range, so instead we will
        # clip and warn
        if error_mask:
            mat_max = xp.max(matrix)
            mat_min = xp.min(matrix)

            # Adding an epsilon here to avoid erroreous errors
            if mat_max > (self.max + self._eps) or mat_min < (self.min - self._eps):
                warnings.warn(
                    (
                        "Partial matrix update contains values outside of weight "
                        "range. These values will be clipped. To remove this wanring, "
                        "set the weight range to contain the full range of expected "
                        "parital matrix updates."
                    ),
                    category=RuntimeWarning,
                    stacklevel=2,
                )

        # Clip the matrix values
        # This is done at this level so that matrix partitions are not
        # separately clipped using different limits
        # Need to update the params on the individual cores
        # Only set percentile limits if we're writng the full matrix
        else:
            # TODO: This seems error prone.
            #       self.min and self.max are conditionally defined
            #       (@Curtis, 2023-11-17)
            if self.mapping.weights.percentile:
                self.mapping.weights.update_mapping(mcopy)
            self.min = self.mapping.weights.min
            self.max = self.mapping.weights.max

        if self.weight_clipping:
            mcopy = mcopy.clip(self.min, self.max)

        self.matrix_col_sum = matrix.sum(axis=0)
        self.matrix_row_sum = matrix.sum(axis=1)

        for row_partition_bounds in self.row_partition_bounds:
            for col_partition_bounds in self.col_partition_bounds:
                self._scale_and_set_partition(
                    matrix=mcopy,
                    row_partition_bounds=row_partition_bounds,
                    col_partition_bounds=col_partition_bounds,
                    apply_errors=apply_errors,
                    error_mask=error_mask,
                )

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
        matrix = xp.zeros((self.nrow, self.ncol))
        for row, row_start, row_end in self.row_partition_bounds:
            for col, col_start, col_end in self.col_partition_bounds:
                subcore = self.subcores[(row, col)]
                matrix[row_start:row_end, col_start:col_end] = self.scale_weights(
                    weights=subcore.read_matrix(apply_errors),
                    source=subcore.mapping.weights,
                    target=self.mapping.weights,
                )

        if not self.complex_valued:
            return matrix
        else:
            Nx, Ny = matrix.shape[0] // 2, matrix.shape[1] // 2
            m_real = matrix[0:Nx, 0:Ny]
            m_imag = matrix[Nx:, 0:Ny]
            return m_real + 1j * m_imag

    def set_mvm_inputs(self, vector: npt.NDArray):
        """Sets the inputs that will be used in matrix vector multiplication.

        Args:
            vector: Input vector to set.
        """
        mapping = self.mapping.mvm
        clipping_enabled = mapping.clipping
        implicitly_initialized = mapping._implicitly_initialized
        vec_out_of_range = vector.max() > mapping.max or vector.min() < mapping.min
        if clipping_enabled and implicitly_initialized and vec_out_of_range:
            warnings.warn(
                "MVM input mapping was implicitly defined and input value is "
                "outside of bounds. Input value will be clipped, which may "
                "not be desired.",
                UserWarning,
                stacklevel=2,
            )
        super().set_mvm_inputs(vector=vector)

    def set_vmm_inputs(self, vector: npt.NDArray):
        """Sets the inputs that will be used in matrix vector multiplication.

        Args:
            vector: Input vector to set.
        """
        mapping = self.mapping.vmm
        clipping_enabled = mapping.clipping
        implicitly_initialized = mapping._implicitly_initialized
        vec_out_of_range = vector.max() > mapping.max or vector.min() < mapping.min
        if clipping_enabled and implicitly_initialized and vec_out_of_range:
            warnings.warn(
                "VMM input mapping was implicitly defined and input value is "
                "outside of bounds. Input value will be clipped, which may "
                "not be desired.",
                UserWarning,
                stacklevel=2,
            )
        super().set_vmm_inputs(vector=vector)

    def matvec(self, vec: npt.ArrayLike) -> npt.NDArray:
        """Perform matrix-vector (Ax = b) multiply on programmed vector (1D).

        Primary simulation function for 1D inputs. Transforms the vector for
        analog simulation and calls the underlying core simulation functions for
        each sub-core. Without errors this should be identical to
        ``A.matmul(vec)`` or ``A @ vec`` where A is the numpy array programmed
        with set_matrix().

        Args:
            vec: 1D Numpy-like array to be multiplied.

        Returns:
            1D Numpy-like array result of matrix-vector multiplication.
        """
        # If complex, concatenate real and imaginary part of input
        vec = xp.asarray(vec)

        if vec.shape != (self.shape[1],) and vec.shape != (self.shape[1], 1):
            raise ValueError(
                "Operands could not be broadcast together with ",
                f"shapes {self.shape}, {vec.shape}",
            )

        if self.complex_valued:
            vec_real = xp.real(vec)
            vec_imag = xp.imag(vec)
            vcopy = xp.concatenate((vec_real, vec_imag))
        else:
            vcopy = vec.copy()

        if self.mvm_input_percentile_scaling:
            self.mapping.mvm.update_mapping(vcopy)

        output = xp.zeros(self.nrow)
        for row, row_start, row_end in self.row_partition_bounds:
            for col, col_start, col_end in self.col_partition_bounds:
                vec_in = vcopy[col_start:col_end]
                self.set_mvm_inputs(vector=vec_in)
                out = self.subcores[(row, col)].run_xbar_mvm()
                scaled_output = self.subcores[(row, col)].scale_mvm_output(
                    x=out,
                    source=self.subcores[(row, col)].mapping,
                    target=self.mapping,
                )
                output[row_start:row_end] += scaled_output

        # If complex, compose real and imaginary
        if self.complex_valued:
            N = int(len(output) / 2)
            output_real = output[:N]
            output_imag = output[N:]
            output = output_real + 1j * output_imag

        output_dtype = self.output_dtype_resolver(
            matrix_dtype=self.dtype,
            input_dtype=vec.dtype,
        )
        return output.astype(output_dtype)

    def matmat(self, mat: npt.ArrayLike) -> npt.NDArray:
        """Perform right matrix-matrix (AX=B) multiply on programmed 2D matrix.

        Primary simulation function for 2D inputs. Transforms the matrix for
        analog simulation and calls the underlying core simulation functions for
        each sub-core. Without errors this should be identical to
        ``A.matmul(mat)`` or ``A @ mat`` where A is the numpy array programmed
        with set_matrix().

        Args:
            mat: 2D Numpy-like array to be multiplied.

        Returns:
            2D Numpy-like array result of matrix-matrix multiplication.
        """
        mat = xp.asarray(mat)
        mcopy = []

        if self.complex_valued:
            smat = mat.reshape(-1, *mat.shape[-2:])
            mcopy += [
                [xp.real(smat[i])] + [xp.imag(smat[i])] for i in range(smat.shape[0])
            ]
            mcopy = xp.vstack(mcopy).reshape(
                (*mat.shape[:-2], mat.shape[-2] * 2, mat.shape[-1]),
            )
        else:
            mcopy = mat.copy()

        if self.mvm_input_percentile_scaling:
            self.mapping.mvm.update_mapping(mcopy)

        mat_3d = mcopy.reshape(-1, *mcopy.shape[-2:])
        output_3d = xp.zeros((mat_3d.shape[0], self.nrow, mcopy.shape[-1]))
        for row, row_start, row_end in self.row_partition_bounds:
            for col, col_start, col_end in self.col_partition_bounds:
                mat_in = mat_3d[:, col_start:col_end, :]
                self.set_mvm_inputs(vector=mat_in)
                out = self.subcores[(row, col)].run_xbar_mvm()
                scaled_output = self.subcores[(row, col)].scale_mvm_output(
                    x=out,
                    source=self.subcores[(row, col)].mapping,
                    target=self.mapping,
                )
                output_3d[:, row_start:row_end, :] += scaled_output
        output = output_3d.reshape(*mcopy.shape[:-2], self.nrow, mcopy.shape[-1])

        if self.complex_valued:
            output = output.reshape(-1, output.shape[-2], output.shape[-1])
            answer = (
                output[:, : int(self.nrow // 2)] + 1j * output[:, int(self.nrow // 2) :]
            )
            answer = answer.reshape(*mat.shape[:-2], self.shape[0], mat.shape[-1])
        else:
            answer = output

        output_dtype = self.output_dtype_resolver(
            matrix_dtype=self.dtype,
            input_dtype=mat.dtype,
        )
        return answer.astype(output_dtype)

    def vecmat(self, vec: npt.ArrayLike) -> npt.NDArray:
        """Perform vector-matrix (xA = b) multiply on programmed vector (1D).

        Primary simulation function for 1D inputs. Transforms the vector for
        analog simulation and calls the underlying core simulation functions for
        each sub-core. Without errors this should be identical to
        ``vec.matmul(A)`` or ``vec @ A`` where A is the numpy array programmed
        with set_matrix().

        Args:
            vec: 1D Numpy-like array to be multiplied.

        Returns:
            1D Numpy-like array result of vector-matrix multiplication.
        """
        vec = xp.asarray(vec)

        if vec.shape != (self.shape[0],) and vec.shape != (1, self.shape[0]):
            raise ValueError(
                "Operands could not be broadcast together with ",
                f"shapes {self.shape}, {vec.shape}",
            )

        if self.complex_valued:
            vec_real = xp.real(vec)
            vec_imag = xp.imag(vec)
            vcopy = xp.concatenate((vec_imag, vec_real))
        else:
            vcopy = vec.copy()

        if self.vmm_input_percentile_scaling:
            self.mapping.vmm.update_mapping(vcopy)

        output = xp.zeros(self.ncol)
        for row, row_start, row_end in self.row_partition_bounds:
            for col, col_start, col_end in self.col_partition_bounds:
                vec_in = vcopy[row_start:row_end]
                self.set_vmm_inputs(vector=vec_in)
                out = self.subcores[(row, col)].run_xbar_vmm()
                scaled_output = self.subcores[(row, col)].scale_vmm_output(
                    x=out,
                    source=self.subcores[(row, col)].mapping,
                    target=self.mapping,
                )
                output[col_start:col_end] += scaled_output

        if self.complex_valued:
            N = int(len(output) / 2)
            output_real = output[N:]
            output_imag = output[:N]
            output = output_real + 1j * output_imag

        output_dtype = self.output_dtype_resolver(
            matrix_dtype=self.dtype,
            input_dtype=vec.dtype,
        )
        return output.astype(output_dtype)

    def rmatmat(self, mat: npt.ArrayLike) -> npt.NDArray:
        """Perform left matrix-matrix (XA=B) multiply on programmed matrix (2D).

        Primary simulation function for 2D inputs. Transforms the matrix for
        analog simulation and calls the underlying core simulation functions for
        each sub-core.  Without errors this should be identical to
        ``mat.matmul(A)`` or ``mat @ A`` where A is the numpy array programmed
        with set_matrix().

        Args:
            mat: 2D Numpy-like array to be multiplied.

        Returns:
            2D Numpy-like array result of matrix-matrix multiplication.
        """
        mat = xp.asarray(mat)
        mcopy = []

        if self.shape[0] != mat.shape[-1]:
            raise ValueError(
                "Operands could not be broadcast together with ",
                f"shapes {self.shape}, {mat.shape}",
            )

        # TODO: investigate if memory gets copied with these intermediate
        # values (mat_real, mat_imag, mat_3d, output_3d) when using
        # numpy/cupy/any other backend we may use
        if self.complex_valued:
            smat = mat.reshape(-1, mat.shape[-2], mat.shape[-1])
            mcopy += [
                [xp.hstack((xp.real(smat[i]), -xp.imag(smat[i])))]
                for i in range(smat.shape[0])
            ]
            mcopy = xp.vstack(mcopy).reshape((*mat.shape[:-1], 2 * mat.shape[-1]))
        else:
            mcopy = mat.copy()

        if self.vmm_input_percentile_scaling:
            self.mapping.vmm.update_mapping(mcopy)

        mat_3d = mcopy.reshape(-1, *mcopy.shape[-2:])
        output_3d = xp.zeros((*mat_3d.shape[:-1], self.ncol))
        for row, row_start, row_end in self.row_partition_bounds:
            for col, col_start, col_end in self.col_partition_bounds:
                mat_in = mat_3d[:, :, row_start:row_end]
                self.set_vmm_inputs(vector=mat_in)
                out = self.subcores[(row, col)].run_xbar_vmm()
                scaled_output = self.subcores[(row, col)].scale_vmm_output(
                    x=out,
                    source=self.subcores[(row, col)].mapping,
                    target=self.mapping,
                )
                output_3d[:, :, col_start:col_end] += scaled_output
        output = output_3d.reshape(*mcopy.shape[:-1], self.ncol)

        if self.complex_valued:
            output = output.reshape(-1, output.shape[-2], output.shape[-1])
            answer = (
                output[:, :, : int(self.ncol // 2)]
                - 1j * output[:, :, int(self.ncol // 2) :]
            )
            answer = answer.reshape((*mat.shape[:-1], self.shape[-1]))
        else:
            answer = output

        output_dtype = self.output_dtype_resolver(
            matrix_dtype=self.dtype,
            input_dtype=mat.dtype,
        )
        return answer.astype(output_dtype)

    def matmul(self, x: npt.ArrayLike) -> npt.NDArray:
        """Numpy-like np.matmul function for N-D inputs.

        Performs an N-D matrix dot product with the programmed matrix.
        For >=2D inputs this will decompose the matrix into a series for
        1D inputs or use a (generally faster) matrix-matrix approximation
        if possible given the simulation parameters. In the error free
        case this should be identical to ``np.matmul(A, x)`` or ``A @ x``
        where A is the numpy array programmed with set_matrix().

        Args:
            x: An N-D numpy-like array to be multiplied.

        Returns:
            An N-D numpy-like array result.
        """
        x = xp.asarray(x)

        # Technically ndim=2 (N,1) inputs are also "vectors" but by they
        # require a different output shape which is handled correctly if
        # they go through the matmat path instead.
        if x.ndim == 1:
            return self.matvec(x)
        else:
            # Stacking fails for shape (X, 0), revert to matmat to handle this
            # Empty matix is weird so ignoring user preference here
            if not (self.fast_matmul or x.shape[1] == 0):
                original_shape = x.shape
                while x.ndim > 3:
                    new_shape = (-1,) + x.shape[-2:]
                    x = x.reshape(new_shape)

                # Reshape 3D into 2D while maintaining dimension for matvec
                if x.ndim == 3:
                    x = x.transpose(1, 0, 2).reshape(x.shape[1], -1)

                output = xp.hstack(
                    [self.matvec(col).reshape(-1, 1) for col in x.T],
                )

                # Reshape 2D into 3D
                if len(original_shape) >= 3:
                    output = output.reshape(
                        self.shape[0],
                        -1,
                        original_shape[-1],
                    ).transpose(1, 0, 2)

                if len(original_shape) > 3:
                    new_shape = (
                        *original_shape[:-2],
                        self.shape[0],
                        original_shape[-1],
                    )
                    output = output.reshape(new_shape)

                return output
            else:
                return self.matmat(x)

    def rmatmul(self, x: npt.ArrayLike) -> npt.NDArray:
        """Numpy-like np.matmul function for N-D inputs.

        Performs an N-D matrix dot product with the programmed matrix.
        For >=2D inputs this will decompose the matrix into a series for
        1D inputs or use a (generally faster) matrix-matrix approximation
        if possible given the simulation parameters. In the error free
        case this should be identical to ``np.matmul(x, A)`` or ``x @ A``
        where A is the numpy array programmed with set_matrix().

        Args:
            x: An N-D numpy-like array to be multiplied.

        Returns:
            An N-D numpy-like array result.
        """
        x = xp.asarray(x)

        # As with dot, sending all ndim == 2 to matmat fixes some
        # shape inconsistency when compared to numpy
        if x.ndim == 1:
            return self.vecmat(x)
        else:
            # Stacking fails for shape (0, X), revert to matmat to handle this
            # Empty matix is weird so ignoring user preference here
            if not (self.fast_matmul or x.shape[0] == 0):
                original_shape = x.shape
                x = x.reshape(-1, original_shape[-1])
                output = xp.vstack([self.vecmat(row) for row in x])
                if len(original_shape) > 2:
                    new_shape = (*original_shape[:-1], self.shape[-1])
                    output = output.reshape(new_shape)
                return output
            else:
                return self.rmatmat(x)

    def dot(self, x: npt.ArrayLike) -> npt.NDArray:
        """Numpy-like ndarray.dot function for N-D inputs.

        Performs an N-D matrix dot product with the programmed matrix. For >=2D
        inputs this will decompose the matrix into a series for 1D inputs or
        use a (generally faster) matrix-matrix approximation if possible given
        the simulation parameters. In the error free case this should be
        identical to ``A.dot(x)`` or ``np.dot(A,x)`` where A is the numpy array
        programmed with set_matrix().

        Args:
            x: An N-D numpy-like array to be multiplied.

        Returns:
            An N-D numpy-like array result.
        """
        x = xp.asarray(x)

        # Technically ndim=2 (N,1) inputs are also "vectors" but by they
        # require a different output shape which is handled correctly if
        # they go through the matmat path instead.
        if x.ndim == 1:
            return self.matvec(x)
        else:
            # Stacking fails for shape (X, 0), revert to matmat which handles
            # this. Empty matix is weird so ignoring user preference here
            if not (self.fast_matmul or x.shape[1] == 0):
                original_shape = x.shape
                while x.ndim > 3:
                    new_shape = (-1,) + x.shape[-2:]
                    x = x.reshape(new_shape)

                # Reshape 3D into 2D while maintaining dimension for matvec
                if x.ndim == 3:
                    x = x.transpose(1, 0, 2).reshape(x.shape[1], -1)
                output = xp.hstack(
                    [
                        self.matvec(
                            col.reshape(
                                -1,
                            ),
                        ).reshape(-1, 1)
                        for col in x.T
                    ],
                )

                if len(original_shape) > 2:
                    new_shape = (
                        self.shape[0],
                        *original_shape[:-2],
                        original_shape[-1],
                    )
                    output = output.reshape(new_shape)

                return output
            else:
                result = self.matmat(x)
                return result.transpose(-2, *range(result.ndim - 2), -1)

    def rdot(self, x: npt.ArrayLike) -> npt.NDArray:
        """Numpy-like ndarray.dot() function for N-D inputs.

        Performs an N-D matrix dot product with the programmed matrix. For >=2D
        inputs this will decompose the matrix into a series for 1D inputs or
        use a (generally faster) matrix-matrix approximation if possible given
        the simulation parameters. In the error free case this should be
        identical to ``x.dot(A)`` or ``np.dot(x, A)`` where A is the numpy array
        programmed with set_matrix().

        Args:
            x: An N-D numpy-like array to be multiplied.

        Returns:
            An N-D numpy-like array result.
        """
        x = xp.asarray(x)

        # As with dot, sending all ndim == 2 to matmat fixes some shape
        # inconsistency when compared to numpy
        if x.ndim == 1:
            return self.vecmat(x)
        else:
            # Stacking fails for shape (0, X), revert to matmat which handles
            # this. Empty matix is weird so ignoring user preference here
            if not (self.fast_matmul or x.shape[0] == 0):
                original_shape = x.shape
                x = x.reshape(-1, original_shape[-1])
                output = xp.vstack([self.vecmat(row) for row in x])
                if len(original_shape) > 2:
                    new_shape = (*original_shape[:-1], self.shape[-1])
                    output = output.reshape(new_shape)
                return output
            else:
                return self.rmatmat(x)

    def mat_multivec(self, vec):
        """Perform matrix-vector multiply on multiple analog vectors packed into
        the "vec" object. A single MVM op in the simulation models multiple MVMs
        in the physical hardware.

        The "vec" object will be reshaped into the following 2D shape:
            (Ncopy, N)
        where Ncopy is the number of input vectors packed into the MVM
        simulation and N is the length of a single input vector

        Args:
            vec: ...

        Raises:
            NotImplementedError: ...
            ValueError: ...

        Returns:
            NDArray: ...
        """
        if self.complex_valued:
            raise NotImplementedError(
                "MVM packing not supported for complex-valued MVMs",
            )

        if self.Ncores == 1:
            return self.matvec(vec.flatten(), bypass_dimcheck=True)

        else:
            Ncopy = (
                self.params.simulation.convolution.x_par
                * self.params.simulation.convolution.y_par
            )
            if vec.size != Ncopy * self.ncol:
                raise ValueError("Packed vector size incompatible with core parameters")
            if vec.shape != (Ncopy, self.ncol):
                vec = vec.reshape((Ncopy, self.ncol))

            output = xp.zeros((Ncopy, self.nrow))
            for i in range(self.num_cores_col):
                output_i = xp.zeros((Ncopy, self.nrow))
                i_start = xp.sum(self.NcolsVec[:i])
                i_end = xp.sum(self.NcolsVec[: i + 1])
                vec_i = vec[:, i_start:i_end].flatten()
                for j in range(self.num_cores_row):
                    j_start = xp.sum(self.NrowsVec[:j])
                    j_end = xp.sum(self.NrowsVec[: j + 1])
                    output_ij = self.cores[j][i].run_xbar_mvm(vec_i.copy())
                    output_i[:, j_start:j_end] = output_ij.reshape(
                        (Ncopy, j_start - j_end),
                    )
                output += output_i
            output_dtype = self.output_dtype_resolver(
                matrix_dtype=self.dtype,
                input_dtype=vec.dtype,
            )
            return output.flatten().astype(output_dtype)

    def transpose(self) -> AnalogCore:
        """Returns a transposed view of the core."""
        from simulator.cores.logical.transposed_core import TransposedCore

        return TransposedCore(parent=self)

    @property
    def T(self) -> AnalogCore:
        """Returns a transposed view of the core."""
        return self.transpose()

    def __getitem__(self, item):
        """Returns a masked view of the core."""
        from simulator.cores.logical.masked_core import MaskedCore

        if not isinstance(item, tuple):
            item = (item, slice(None, None, None))
        full_mask, flatten = logical_utils.is_full_mask(self.shape, self.ndim, item)
        if full_mask:
            return self
        return MaskedCore(self, *item, flatten)

    def __setitem__(self, key, value):
        """Sets a value on a masked view of the core."""
        if not isinstance(key, tuple):
            key = (key, slice(None, None, None))
        full_mask, _ = logical_utils.is_full_mask(self.shape, self.ndim, key)
        rslice, cslice = key
        expanded_mat = self.read_matrix(apply_errors=False)
        expanded_mat[rslice, cslice] = xp.asarray(value)
        error_mask = None if full_mask else (rslice, cslice)
        self.set_matrix(expanded_mat, apply_errors=True, error_mask=error_mask)

    def _calculate_partition(self):
        """Calculates how the core will partition matrices.

        Assumes that _process_matrix() has been called.

        Raises:
            ValueError: Raised if the amount of subcores does not match the
                amount of partitions.
        """
        # Determine number of cores
        NrowsMax, NcolsMax = self.params.core.max_partition_size
        if NrowsMax > 0:
            self.Ncores = (self.ncol - 1) // NrowsMax + 1
        else:
            self.Ncores = 1

        if NcolsMax > 0:
            self.Ncores *= (self.nrow - 1) // NcolsMax + 1
        else:
            self.Ncores *= 1

        self.num_cores_row = self.Ncores // ((self.ncol - 1) // NrowsMax + 1)
        self.num_cores_col = self.Ncores // self.num_cores_row

        logical_utils.verify_partition_scheme(
            keys=self.subcores.keys(),
            partition_mode=self.partition_mode,
            expected_shape=(self.num_cores_row, self.num_cores_col),
        )

        self.NrowsVec, self.row_partition_bounds = logical_utils.partition(
            num_cores=self.num_cores_row,
            num_elements=self.nrow,
            max_partition=NcolsMax,
            partition_priority=self.row_partition_priority,
            partition_strategy=self.row_partition_strategy,
        )
        self.NcolsVec, self.col_partition_bounds = logical_utils.partition(
            num_cores=self.num_cores_col,
            num_elements=self.ncol,
            max_partition=NrowsMax,
            partition_priority=self.col_partition_priority,
            partition_strategy=self.col_partition_strategy,
        )

        if self.partition_mode == PartitionMode.AUTO:
            default_param = self.core_params.subcores["default"]
            subcore_params = {}
            for row, _, _ in self.row_partition_bounds:
                for col, _, _ in self.col_partition_bounds:
                    subcore_params[(row, col)] = default_param.copy()
            self.core_params.subcores = subcore_params
            self.subcores = core_utils.make_subcores(
                xsim_parameters=self.params,
                core_parameters=self.core_params,
                parent=self,
            )
            core_utils.add_adc_to_core(
                xsim_parameters=self.params,
                core_parameters=self.core_params,
                core=self,
            )
            core_utils.add_dac_to_core(
                xsim_parameters=self.params,
                core_parameters=self.core_params,
                core=self,
            )

    def _scale_and_set_partition(
        self,
        matrix: npt.NDArray,
        row_partition_bounds,
        col_partition_bounds,
        apply_errors: bool,
        error_mask: tuple[slice, slice],
    ):
        """Get error mask, scale weights, and set a partition of the matrix."""
        row, row_start, row_end = row_partition_bounds
        col, col_start, col_end = col_partition_bounds
        subcore = self.subcores[(row, col)]
        matrix_partition = matrix[row_start:row_end, col_start:col_end]
        if error_mask:
            emask = logical_utils.clean_error_mask(self.shape, *error_mask)
            error_mask = logical_utils.set_error_mask(
                error_mask=emask,
                row_start=row_start,
                col_start=col_start,
                row_end=row_end,
                col_end=col_end,
            )
        scaled_matrix_partition = self.scale_weights(
            weights=matrix_partition,
            source=self.mapping.weights,
            target=subcore.mapping.weights,
        )
        subcore.set_matrix(
            matrix=scaled_matrix_partition,
            apply_errors=apply_errors,
            error_mask=error_mask,
        )

    def _reset(self):
        """Initializes AnalogCore attributes from parameters."""
        # Floating point epsilons come up occassionally so just store it here
        self._eps = xp.finfo(float).eps

        core_utils.add_adc_to_core(
            xsim_parameters=self.params,
            core_parameters=self.core_params,
            core=self,
        )
        core_utils.add_dac_to_core(
            xsim_parameters=self.params,
            core_parameters=self.core_params,
            core=self,
        )

        self.mvm_input_percentile_scaling = self.mapping.mvm.percentile is not None
        self.vmm_input_percentile_scaling = self.mapping.vmm.percentile is not None

        self.complex_valued = (
            self.core_params.complex_matrix or self.core_params.complex_input
        )
        self.fast_matmul = self.params.simulation.fast_matmul
        self.weight_clipping = self.mapping.weights.clipping
        self.output_dtype_strategy = self.params.core.output_dtype_strategy
        self.output_dtype_resolver = logical_utils.create_output_dtype_resolver(
            strategy=self.output_dtype_strategy,
        )

        partitioning_params = self.core_params.partitioning
        self.partition_mode = partitioning_params.partition_mode
        self.col_partition_priority = partitioning_params.col_partition_priority
        self.row_partition_priority = partitioning_params.row_partition_priority
        self.col_partition_strategy = partitioning_params.col_partition_strategy
        self.row_partition_strategy = partitioning_params.row_partition_strategy

        self.nrow = None
        self.ncol = None
        self.ndim = None
        self.dtype = None
        self._shape = None

    def __matmul__(self, other: npt.ArrayLike) -> npt.NDArray:
        """Return self@value."""
        other = xp.asarray(other)
        return self.matmul(other)

    def __rmatmul__(self, other: npt.ArrayLike) -> npt.NDArray:
        """Return value@self."""
        other = xp.asarray(other)
        return self.rmatmul(other)

    def __repr__(self) -> str:
        """Returns repr(self)."""
        prefix = "AnalogCore("
        mid = xp.array2string(
            self.read_matrix(apply_errors=False),
            separator=", ",
            prefix=prefix,
        )
        suffix = ")"
        return prefix + mid + suffix

    def __str__(self) -> str:
        """Return str(self)."""
        return self.read_matrix(apply_errors=False).__str__()

    def __array__(self) -> npt.NDArray:
        """Return np.array(self)."""
        return self.read_matrix(apply_errors=False)

    def generate_mapping(self) -> CoreMappingParameters:
        """Generate a mapping for the core."""
        return self.core_params.mapping.copy()
