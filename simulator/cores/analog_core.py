#
# Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government
# retains certain rights in this software.
#
# See LICENSE for full license details
#

"""A Numpy-like interface for interacting with analog MVM operations.

AnalogCore is the primary interface for CrossSim analog MVM simulations. AnalogCores
behave as if they are Numpy arrays for 1 and 2 dimensional matrix multiplication (left
and right), transpostion and slice operations. This is intended to provide drop-in
compatibility with existing numpy matrix manipulation code. AnalogCore only implements
functions which make physical sense for analog MVM array, for instance element-wise
operations are intentionally not supported as they don't have an obvious physical
implementation.

Internally AnalogCore may partition a single matrix across multiple physical arrays
based on the input data types and specified parameters.
"""

from __future__ import annotations

import numpy as np
from warnings import warn
from simulator.parameters import CrossSimParameters

from . import BalancedCore, OffsetCore, BitslicedCore, NumericCore

from simulator.backend import ComputeBackend
from simulator.parameters.core_parameters import (
    PartitionStrategy,
    CoreStyle,
    BalancedCoreStyle,
    OutputDType,
)


import numpy.typing as npt

xp = ComputeBackend()


class AnalogCore:
    """Primary simulation object for Analog MVM.

    AnalogCore provides a numpy-like interface for matrix multiplication operations
    using analog MVM arrays. AnalogCore should be the primary interface for algorithms.
    AnalogCore internally contains multiple physical cores (which may correspond to
    multiple discrete arrays in the case of balanced and bit-sliced systems). AnalogCore
    may also expand the provided matrix as needed for analog computation, for instance
    when using complex numbers.

    Attributes:
        params:
            The primary CrossSimParameters object setting the configuration for the
            AnalogCore. If a core has multiple CrossSimParameters (multiple physical
            cores) the first core in the list is the primary parameter.
        Ncores:
            An integer count of the number of physical cores needed to represent the
            programmed matrix.
        cores:
            A 2D list of WrapperCore objects representing the the physical cores
            representing the programmed matrix. There are Ncores WrapperCores in cores.
        complex_valued:
            A bool indicating whether the core supports complex valued operations.
        ndim: An integer indicating the number of dimensions of the matrix. Always 2.
        shape: A tuple of integers indicating the shape of the programmed matrix
        nrow:
            The number of physical rows used to represent the programmed matrix. May be
            greater than the rows in the matrix for transformations such as to handle
            complex-valued matrices.
        ncol:
            The number of physical columns used to represent the programmed matrix.
            May be greater than the columns in the matrix for transformations such as
            to handle complex-valued matrices.
        dtype: Data type of the programmed matrix, typically np.float32.

    """

    # Forces numpy to use rmatmat for right multiplies
    # Alternative is subclassing ndarray which has some annoying potential side effects
    # cupy uses __array_priority__ = 100 so need to be 1 more than that
    __array_priority__ = 101

    def __init__(
        self,
        matrix: npt.ArrayLike,
        params: CrossSimParameters | list[CrossSimParameters],
        empty_matrix=False,
    ) -> None:
        """Initializes an AnalogCore object with the provided dimension and parameters.

        Args:
            matrix: Matrix to initialize the array size and (optionally) data
            params: Parameters object or objects specifying the behavior of the object.
            empty_matrix: Bool to skip initializing array data from input matrix
        Raises:
            ValueError: Parameter object is invalid for the configuration.
        """
        # Set primary params object and ensure inputs are valid
        if isinstance(params, CrossSimParameters):
            self.params = params.copy()
        elif isinstance(params, list):
            if all(isinstance(p, CrossSimParameters) for p in params):
                self.params = params[0].copy()
            else:
                raise ValueError(
                    "All params objects in list must be CrossSimParameters",
                )
        else:
            raise ValueError(
                "Params must be CrossSimParameters or list of CrossSimParameters",
            )

        # Initialize the compute backend
        gpu_id = (
            self.params.simulation.gpu_id if self.params.simulation.useGPU else None
        )
        xp.__init__(self.params.simulation.useGPU, gpu_id)

        matrix = xp.asarray(matrix)

        # Floating point epsilons come up occasionally so just store it here
        self._eps = xp.finfo(float).eps

        # params used in AnalogCore
        self.complex_valued = (
            self.params.core.complex_matrix or self.params.core.complex_input
        )
        self.fast_matmul = self.params.simulation.fast_matmul
        self.shape = matrix.shape
        self.weight_clipping = self.params.core.mapping.weights.clipping

        self.mvm_input_percentile_scaling = (
            self.params.core.mapping.inputs.mvm.percentile is not None
        )
        self.vmm_input_percentile_scaling = (
            self.params.core.mapping.inputs.vmm.percentile is not None
        )

        # set the output type for this core
        if self.params.core.output_dtype == OutputDType.NATIVE:
            self.output_type = self._native_dtype
        elif self.params.core.output_dtype == OutputDType.MATRIX:
            self.output_type = self._matrix_dtype
        elif self.params.core.output_dtype == OutputDType.INPUT:
            self.output_type = self._input_dtype
        elif self.params.core.output_dtype == OutputDType.FLOAT32:
            self.output_type = self._explicit_dtype
            self._output_type = xp.float32
        elif self.params.core.output_dtype == OutputDType.FLOAT64:
            self.output_type = self._explicit_dtype
            self._output_type = xp.float64
        elif self.params.core.output_dtype == OutputDType.FLOAT16:
            self.output_type = self._explicit_dtype
            self._output_type = xp.float16
        elif self.params.core.output_dtype == OutputDType.INT64:
            self.output_type = self._explicit_dtype
            self._output_type = xp.int64
        elif self.params.core.output_dtype == OutputDType.INT32:
            self.output_type = self._explicit_dtype
            self._output_type = xp.int32
        elif self.params.core.output_dtype == OutputDType.INT16:
            self.output_type = self._explicit_dtype
            self._output_type = xp.int16
        elif self.params.core.output_dtype == OutputDType.INT8:
            self.output_type = self._explicit_dtype
            self._output_type = xp.int8

        # This protects from the case where AnalogCore is a 1D vector which breaks
        # complex equivalent expansion. This could probably be fixed but it is a
        # sufficiently unusual case that just throw an error for now.
        if self.complex_valued and (
            matrix.ndim == 1 or any(i == 1 for i in matrix.shape)
        ):
            raise ValueError("AnalogCore must 2 dimensional")

        # AnalogCore has slice objects to simplify core operation stacking
        self.rslice = None
        self.cslice = None

        # TODO: temporary setting nrow and ncol for compatibility
        # Change when fixing row/col confusion
        nrow, ncol = matrix.shape

        # Double # rows and columns for complex matrix
        if self.complex_valued:
            nrow *= 2
            ncol *= 2

        # Determine number of cores
        NrowsMax = self.params.core.rows_max
        NcolsMax = self.params.core.cols_max
        if NrowsMax > 0:
            self.Ncores = (ncol - 1) // NrowsMax + 1
        else:
            self.Ncores = 1

        if NcolsMax > 0:
            self.Ncores *= (nrow - 1) // NcolsMax + 1
        else:
            self.Ncores *= 1

        # Check that Ncores is compatible with the number of params objects
        params_ = params
        # Just duplicate the params object if the user only passed in a single params.
        # If the list lengths don't match however, e.g. user intentionally passed in a
        # list but it is the wrong size, that might indicate a config problem so error.
        if self.Ncores > 1 and isinstance(params_, CrossSimParameters):
            params_ = [params] * self.Ncores
        elif self.Ncores == 1 and isinstance(params_, list):
            raise ValueError("Multiple params objects provided for single-core layer")
        elif (
            self.Ncores > 1
            and isinstance(params_, list)
            and len(params_) != self.Ncores
        ):
            raise ValueError(
                "Number of params objects provided does not match number of cores",
            )

        self.col_partition_priority = (
            self.params.core.mapping.weights.col_partition_priority
        )
        self.row_partition_priority = (
            self.params.core.mapping.weights.row_partition_priority
        )
        self.col_partition_strategy = (
            self.params.core.mapping.weights.col_partition_strategy
        )
        self.row_partition_strategy = (
            self.params.core.mapping.weights.row_partition_strategy
        )

        # Create single cores
        if self.Ncores == 1:
            self.cores = [[self._make_core(params)]]
            self.num_cores_row = 1
            self.num_cores_col = 1
            self.NrowsVec = [nrow]
            self.NcolsVec = [ncol]

        else:
            self.num_cores_row = self.Ncores // ((ncol - 1) // NrowsMax + 1)
            self.num_cores_col = self.Ncores // self.num_cores_row
            self.cores = [
                [
                    self._make_core(params_[r * self.num_cores_col + c])
                    for c in range(self.num_cores_col)
                ]
                for r in range(self.num_cores_row)
            ]

            # Partition the matrix across the sub cores with the following priority:
            # 1) If rows/cols can be partition evenly partition evenly
            # 2) If row/col_partition priority is defined check if nrow/ncol is a
            #   multiple of N*max/partition so all but one core has Nmax*
            #   This is useful (with col_partition_priority = [2, 4]) so that
            #   convolution blocks in depthwise convolutions are not partition.
            #   Maybe not totally robust
            # 3) Otherwise partition based on row/col_partition_strategy (max or even)

            if nrow % self.num_cores_row == 0:
                self.NrowsVec = (nrow // self.num_cores_row) * xp.ones(
                    self.num_cores_row,
                    dtype=xp.int32,
                )
            else:
                # prio_partition = True in (
                #   ((nrow % (self.NrowsMax / div)) == 0)
                #   for div in self.row_partition_priority
                # )
                prio_partition = True in (
                    ((nrow % (NcolsMax / div)) == 0)
                    for div in self.row_partition_priority
                )

                if (
                    prio_partition
                    or self.row_partition_strategy == PartitionStrategy.MAX
                ):
                    # rows_per_core = NrowsMax
                    rows_per_core = NcolsMax
                else:
                    rows_per_core = np.round(nrow / self.num_cores_row).astype(np.int32)
                self.NrowsVec = rows_per_core * np.ones(
                    self.num_cores_row,
                    dtype=np.int32,
                )
                self.NrowsVec[-1] = nrow - (self.num_cores_row - 1) * rows_per_core

            if ncol % self.num_cores_col == 0:
                self.NcolsVec = (ncol // self.num_cores_col) * np.ones(
                    self.num_cores_col,
                    dtype=np.int32,
                )
            else:
                # prio_partition = True in (
                #   ((ncol % (self.NcolsMax / div)) == 0)
                #   for div in self.col_partition_priority
                # )
                prio_partition = True in (
                    ((ncol % (NrowsMax / div)) == 0)
                    for div in self.col_partition_priority
                )

                if (
                    prio_partition
                    or self.col_partition_strategy == PartitionStrategy.MAX
                ):
                    # cols_per_core = NcolsMax
                    cols_per_core = NrowsMax
                else:
                    cols_per_core = np.round(ncol / self.num_cores_col).astype(np.int32)

                self.NcolsVec = cols_per_core * np.ones(
                    self.num_cores_col,
                    dtype=np.int32,
                )
                self.NcolsVec[-1] = ncol - (self.num_cores_col - 1) * cols_per_core

            # Precompute a list of row/col partition information (id, start, end)
            # This is used to aggreagate partitions in every other function
            self.row_partition_bounds: list[tuple[int, int, int]] = [
                (r, int(np.sum(self.NrowsVec[:r])), int(np.sum(self.NrowsVec[: r + 1])))
                for r in range(self.num_cores_row)
            ]
            self.col_partition_bounds: list[tuple[int, int, int]] = [
                (c, int(np.sum(self.NcolsVec[:c])), int(np.sum(self.NcolsVec[: c + 1])))
                for c in range(self.num_cores_col)
            ]

        self.nrow = nrow
        self.ncol = ncol
        self.dtype = None
        self.ndim = 2

        if not empty_matrix:
            self.set_matrix(matrix)

    def set_matrix(
        self,
        matrix: npt.ArrayLike,
        verbose: bool = False,
        error_mask: tuple[slice, slice] | None = None,
    ) -> None:
        """Programs a matrix into the AnalogCore.

        Transform the input matrix as needed for programming to analog arrays including
        complex expansion, clipping, and matrix partitioning. Calls the set_matrix()
        methoods of the underlying core objects.

        Args:
            matrix: Numpy ndarray to be programmed into the array.
            verbose: Boolean flag to enable verbose print statements.
            error_mask: Tuple of slices for setting parts of the matrix

        Raises:
            ValueError: Matrix is not valid for the input parameters.
        """
        matrix = xp.asarray(matrix)
        if self.shape != matrix.shape:
            raise ValueError("Matrix shape must match AnalogCore shape")

        if verbose:
            print("Min/Max matrix values", xp.min(matrix), xp.max(matrix))

        if (
            matrix.dtype == xp.complex64 or matrix.dtype == xp.complex128
        ) and not self.complex_valued:
            raise ValueError(
                (
                    "If setting complex-valued matrices, "
                    "please set core.complex_matrix = True"
                ),
            )

        self.dtype = matrix.dtype

        # Now that we've captured the data type, convert integers to float32s
        if xp.issubdtype(self.dtype, xp.integer):
            matrix = matrix.astype(xp.float32)

        # Break up complex matrix into real and imaginary quadrants
        if self.complex_valued:
            Nx, Ny = matrix.shape
            matrix_real = xp.real(matrix)
            matrix_imag = xp.imag(matrix)
            mcopy = xp.zeros((2 * Nx, 2 * Ny), dtype=matrix_real.dtype)
            mcopy[0:Nx, 0:Ny] = matrix_real
            mcopy[Nx:, 0:Ny] = matrix_imag
            mcopy[0:Nx, Ny:] = -matrix_imag
            mcopy[Nx:, Ny:] = matrix_real
        else:
            mcopy = matrix.copy()

        # For partial matrix updates new values must be inside the previous range.
        # If the values would exceed this range then you would have to reprogram all
        # matrix values based on the new range, so instead we will clip and warn
        if error_mask:
            mat_max = xp.max(matrix)
            mat_min = xp.min(matrix)

            # Adding an epsilon here to avoid erroreous errors
            if mat_max > (self.max + self._eps) or mat_min < (self.min - self._eps):
                print(mat_max, self.params.core.mapping.weights.max)
                print(mat_min, self.params.core.mapping.weights.min)
                warn(
                    (
                        "Partial matrix update contains values outside of weight "
                        "range. These values will be clipped. To remove this warning, "
                        "set the weight range to contain the full range of expected "
                        "partial matrix updates."
                    ),
                    category=RuntimeWarning,
                    stacklevel=2,
                )

        # Warn about a small numeric error if using BALANCED and TWO_SIDED
        if (
            self.params.core.style == CoreStyle.BALANCED
            and self.params.core.balanced.style == BalancedCoreStyle.TWO_SIDED
            and self.params.xbar.device.cell_bits > 0
            and self.dtype != xp.int8
        ):
            warn(
                (
                    "When using BALANCED core with the TWO_SIDED style, there may "
                    "be a small numerical error due to a misalignment in the quantized"
                    "conductance levels. This will be fixed in a future update. To remove "
                    "this warning, use the ONE_SIDED style."
                ),
                category=RuntimeWarning,
                stacklevel=2,
            )

        # Clip the matrix values
        # This is done at this level so that matrix partitions are not separately
        # clipped using different limits
        # Need to update the params on the individual cores
        # Only set percentile limits if we're writng the full matrix
        weight_limits = None
        if not error_mask:
            if self.params.core.mapping.weights.percentile:
                self.min, self.max = self._set_limits_percentile(
                    self.params.core.mapping.weights,
                    mcopy,
                    reset=True,
                )
            else:
                self.min = self.params.core.mapping.weights.min
                self.max = self.params.core.mapping.weights.max
            weight_limits = (self.min, self.max)

        if self.weight_clipping:
            mcopy = mcopy.clip(self.min, self.max)

        for i in range(self.num_cores_row):
            for j in range(self.num_cores_col):
                self.cores[i][
                    j
                ].params.core.mapping.weights.min = self.params.core.mapping.weights.min
                self.cores[i][
                    j
                ].params.core.mapping.weights.max = self.params.core.mapping.weights.max

        if self.Ncores == 1:
            self.cores[0][0].set_matrix(
                mcopy,
                weight_limits=weight_limits,
                error_mask=error_mask,
            )
        else:
            for row, row_start, row_end in self.row_partition_bounds:
                for col, col_start, col_end in self.col_partition_bounds:
                    error_mask_ = error_mask
                    if error_mask:
                        if error_mask[0].step and error_mask[0].step < 0:
                            row_mask = slice(
                                error_mask[0].start - row_start,
                                None,
                                error_mask[0].step,
                            )
                        else:
                            row_mask = slice(
                                error_mask[0].start - row_start,
                                error_mask[0].stop - row_start,
                                error_mask[0].step,
                            )

                        if error_mask[1].step and error_mask[1].step < 0:
                            col_mask = slice(
                                error_mask[1].start - col_start,
                                None,
                                error_mask[1].step,
                            )
                        else:
                            col_mask = slice(
                                error_mask[1].start - col_start,
                                error_mask[1].stop - col_start,
                                error_mask[1].step,
                            )
                        error_mask_ = (
                            slice(*row_mask.indices(row_end - row_start)),
                            slice(*col_mask.indices(col_end - col_start)),
                        )
                    mat_prog = mcopy[row_start:row_end, col_start:col_end]
                    self.cores[row][col].set_matrix(
                        mat_prog,
                        weight_limits=weight_limits,
                        error_mask=error_mask_,
                    )

    def get_matrix(self) -> npt.NDArray:
        """Returns the programmed matrix with weight errors and clipping applied.

        The programmed matrix is converted into original input matrix format, e.g.,
        a single matrix with real or complex inputs, but with analog-specific
        non-idealities applied. Currently this is clipping and programming-time weight
        errors (programming and drift).

        Returns:
            A Numpy array of the programmed matrix.

        """
        if self.Ncores == 1:
            matrix = self.cores[0][0]._read_matrix()
        else:
            matrix = xp.zeros((self.nrow, self.ncol))
            for row, row_start, row_end in self.row_partition_bounds:
                for col, col_start, col_end in self.col_partition_bounds:
                    matrix[row_start:row_end, col_start:col_end] = xp.asarray(
                        self.cores[row][col]._read_matrix(),
                    )

        if not self.complex_valued:
            return self._convert_output_type(matrix, self.dtype)
        else:
            Nx, Ny = matrix.shape[0] // 2, matrix.shape[1] // 2
            m_real = matrix[0:Nx, 0:Ny]
            m_imag = matrix[Nx:, 0:Ny]
            return m_real + 1j * m_imag

    def matvec(self, vec: npt.ArrayLike) -> npt.NDArray:
        """Perform matrix-vector (Ax = b) multiply on programmed vector (1D).

        Primary simulation function for 1D inputs. Transforms the vector for analog
        simulation and calls the underlying core simulation functions for each
        sub-core. Without errors this should be identical to ``A.matmul(vec)`` or
        ``A @ vec`` where A is the numpy array programmed with set_matrix().

        Args:
            vec: 1D Numpy-like array to be multiplied.
            bypass: If True, bypasses call to check_dimensions()

        Returns:
            1D Numpy-like array result of matrix-vector multiplication.
        """
        # If complex, concatenate real and imaginary part of input
        vec = self._ensure_data_format(vec)

        if vec.shape != (self.shape[1],) and vec.shape != (self.shape[1], 1):
            raise ValueError(
                f"Dimension mismatch: {self.shape}, {vec.shape}",
            )

        if self.complex_valued:
            vec_real = xp.real(vec)
            vec_imag = xp.imag(vec)
            vcopy = xp.concatenate((vec_real, vec_imag))
        else:
            vcopy = vec.copy()

        input_range = None
        if self.mvm_input_percentile_scaling:
            input_range = self._set_limits_percentile(
                self.params.core.mapping.inputs.mvm,
                vcopy,
                reset=True,
            )

        if self.Ncores == 1:
            output = self.cores[0][0].run_xbar_mvm(vcopy, input_range)

        else:
            output = xp.zeros(self.nrow)
            for row, row_start, row_end in self.row_partition_bounds:
                for col, col_start, col_end in self.col_partition_bounds:
                    vec_in = vcopy[col_start:col_end]
                    output[row_start:row_end] += self.cores[row][col].run_xbar_mvm(
                        vec_in,
                        input_range,
                    )

        # If complex, compose real and imaginary
        if self.complex_valued:
            N = int(len(output) / 2)
            output_real = output[:N]
            output_imag = output[N:]
            output = output_real + 1j * output_imag

        return self._convert_output_type(output, self.output_type(vec.dtype))

    def matmat(self, mat: npt.ArrayLike) -> npt.NDArray:
        """Perform right matrix-matrix (AX = B) multiply on programmed matrix.

        Primary simulation function for >=2D inputs. Transforms the matrix for analog
        simulation and calls the underlying core simulation functions for each
        sub-core.  Without errors this should be identical to ``A.matmul(mat)`` or
        ``A @ mat`` where A is the numpy array programmed with set_matrix().

        Args:
            mat: >=2D Numpy-like array to be multiplied.

        Returns:
            >=2D Numpy-like array result of matrix-matrix multiplication.
        """
        mat = self._ensure_data_format(mat)

        if self.shape[1] != mat.shape[-2]:
            raise ValueError(
                f"Dimension mismatch: {self.shape}, {mat.shape}",
            )

        if self.complex_valued:
            mat_real = xp.real(mat)
            mat_imag = xp.imag(mat)
            mcopy = xp.vstack((mat_real, mat_imag))
        else:
            mcopy = mat.copy()

        input_range = None
        if self.mvm_input_percentile_scaling:
            input_range = self._set_limits_percentile(
                self.params.core.mapping.inputs.mvm,
                mcopy,
                reset=True,
            )

        # Numpy handles N-D inputs as a stack of matrices of the trailing 2
        # dimensions. This is the key to understanding the code below. For the
        # indexing we start from the 2D case and then build a stack.
        if self.Ncores == 1:
            output = self.cores[0][0].run_xbar_mvm(mcopy, input_range)

        else:
            # The output is a stack of matrices based with a shape based on the
            # Matmul of input trailing 2 dims and the 2 stored. Therefore, to
            # size the output, it is the leading dimensions of the input, and
            # the size derived from standard output sizing. In the 2D case
            # there is no leading dimension, so it is just nrow, mat.shape[1].
            # Recall that nrow is self.shape[0] (modulo complex values), so
            # this naturally follows from basic matrix behavior.
            output = xp.zeros((*mat.shape[:-2], self.nrow, mat.shape[-1]))
            # The indexing is attempting to select N-D chunks from mat_in and
            # output. So we want to take all the leading dimensions as given
            # and then select part of the 2nd to last dimension, and the full
            # final dimension. The `...` symbol means take all leading
            # dimensions before the specified dimensions (last 2). In the 2D
            # case this is equivalent to [start:end] which shorthands
            # [start:end, :]. We need to be explicit about the trailing
            # dimension here so the `...` captures the correct number of
            # leading dimensions.
            for row, row_start, row_end in self.row_partition_bounds:
                for col, col_start, col_end in self.col_partition_bounds:
                    mat_in = mcopy[..., col_start:col_end, :]
                    output[..., row_start:row_end, :] += self.cores[row][
                        col
                    ].run_xbar_mvm(
                        mat_in,
                        input_range,
                    )

        if self.complex_valued:
            output_real = output[: int(self.nrow // 2)]
            output_imag = output[int(self.nrow // 2) :]
            output = output_real + 1j * output_imag

        return self._convert_output_type(output, self.output_type(mat.dtype))

    def vecmat(self, vec: npt.ArrayLike) -> npt.NDArray:
        """Perform vector-matrix (xA = b) multiply on programmed vector (1D).

        Primary simulation function for 1D inputs. Transforms the vector for analog
        simulation and calls the underlying core simulation functions for each
        sub-core. Without errors this should be identical to ``vec.matmul(A)`` or
        ``vec @ A`` where A is the numpy array programmed with set_matrix().

        Args:
            vec: 1D Numpy-like array to be multiplied.

        Returns:
            1D Numpy-like array result of vector-matrix multiplication.
        """
        vec = self._ensure_data_format(vec)

        if vec.shape != (self.shape[0],) and vec.shape != (1, self.shape[0]):
            raise ValueError(
                f"Dimension mismatch: {vec.shape}, {self.shape} ",
            )

        if self.complex_valued:
            vec_real = xp.real(vec)
            vec_imag = xp.imag(vec)
            vcopy = xp.concatenate((vec_imag, vec_real))
        else:
            vcopy = vec.copy()

        input_range = None
        if self.vmm_input_percentile_scaling:
            input_range = self._set_limits_percentile(
                self.params.core.mapping.inputs.vmm,
                vcopy,
                reset=True,
            )

        if self.Ncores == 1:
            output = self.cores[0][0].run_xbar_vmm(vcopy, input_range)

        else:
            output = xp.zeros(self.ncol)
            for row, row_start, row_end in self.row_partition_bounds:
                for col, col_start, col_end in self.col_partition_bounds:
                    vec_in = vcopy[row_start:row_end]
                    output[col_start:col_end] += self.cores[row][col].run_xbar_vmm(
                        vec_in,
                        input_range,
                    )

        if self.complex_valued:
            N = int(len(output) / 2)
            output_real = output[N:]
            output_imag = output[:N]
            output = output_real + 1j * output_imag

        return self._convert_output_type(output, self.output_type(vec.dtype))

    def rmatmat(self, mat: npt.ArrayLike) -> npt.NDArray:
        """Perform left matrix-matrix (XA = B) multiply on programmed matrix.

        Primary simulation function for >=2D inputs. Transforms the matrix for analog
        simulation and calls the underlying core simulation functions for each
        sub-core.  Without errors this should be identical to ``mat.matmul(A)`` or
        ``mat @ A`` where A is the numpy array programmed with set_matrix().

        Args:
            mat: >=2D Numpy-like array to be multiplied.

        Returns:
            >=2D Numpy-like array result of matrix-matrix multiplication.
        """
        mat = self._ensure_data_format(mat)

        if self.shape[0] != mat.shape[-1]:
            raise ValueError(
                f"Dimension mismatch: {mat.shape}, {self.shape}",
            )

        if self.complex_valued:
            mat_real = xp.real(mat)
            mat_imag = xp.imag(mat)
            mcopy = xp.hstack((mat_real, -mat_imag))
        else:
            mcopy = mat.copy()

        input_range = None
        if self.vmm_input_percentile_scaling:
            input_range = self._set_limits_percentile(
                self.params.core.mapping.inputs.vmm,
                mcopy,
                reset=True,
            )

        # For explanation of >2D see matmat, this is conceptually the same just
        # with the dimensions flipped. Actually simpler because we don't need
        # the transposes.
        if self.Ncores == 1:
            output = self.cores[0][0].run_xbar_vmm(mcopy, input_range)

        else:
            output = xp.zeros((*mat.shape[:-1], self.ncol))
            for row, row_start, row_end in self.row_partition_bounds:
                for col, col_start, col_end in self.col_partition_bounds:
                    mat_in = mcopy[..., :, row_start:row_end]
                    output[..., :, col_start:col_end] += self.cores[row][
                        col
                    ].run_xbar_vmm(
                        mat_in,
                        input_range,
                    )

        if self.complex_valued:
            output_real = output[:, : int(self.ncol // 2)]
            output_imag = output[:, int(self.ncol // 2) :]
            output = output_real - 1j * output_imag

        return self._convert_output_type(output, self.output_type(mat.dtype))

    def matmul(self, x: npt.ArrayLike) -> npt.NDArray:
        """Numpy-like np.matmul function for N-D inputs.

        Performs an N-D matrix dot product with the programmed matrix. For >=2D
        inputs this will decompose the matrix into a series for 1D inputs or use a
        (generally faster) matrix-matrix approximation if possible given the simulation
        parameters. In the error free case this should be identical to
        ``np.matmul(A, x)`` or ``A @ x`` where A is the numpy array programmed with
        set_matrix().

        Args:
            x: An N-D numpy-like array to be multiplied.

        Returns:
            An N-D numpy-like array result.
        """
        x = self._ensure_data_format(x)

        # Technically ndim=2 (N,1) inputs are also "vectors" but by they require a
        # different output shape which is handled correctly if they go through the
        # matmat path instead.
        if x.ndim == 1:
            return self.matvec(x)
        else:
            # Stacking fails for shape (X, 0), revert to matmat which handles this
            # Empty matix is weird so ignoring user preference here
            if not (self.fast_matmul or x.shape[1] == 0):
                return xp.hstack(
                    [
                        self.matvec(
                            col.reshape(
                                -1,
                            ),
                        ).reshape(-1, 1)
                        for col in x.T
                    ],
                )
            else:
                return self.matmat(x)

    def rmatmul(self, x: npt.ArrayLike) -> npt.NDArray:
        """Numpy-like np.matmul function for N-D inputs.

        Performs an N-D matrix dot product with the programmed matrix. For >=2D
        inputs this will decompose the matrix into a series for 1D inputs or use a
        (generally faster) matrix-matrix approximation if possible given the simulation
        parameters. In the error free case this should be identical to
        ``np.matmul(x, A)`` or ``x @ A`` where A is the numpy array programmed with
        set_matrix().

        Args:
            x: An N-D numpy-like array to be multiplied.

        Returns:
            An N-D numpy-like array result.
        """
        x = self._ensure_data_format(x)

        # As with dot, sending all ndim == 2 to matmat fixes some shape inconsistency
        # when compared to numpy
        if x.ndim == 1:
            return self.vecmat(x)
        else:
            # Stacking fails for shape (0, X), revert to matmat which handles this
            # Empty matix is weird so ignoring user preference here
            if not (self.fast_matmul or x.shape[0] == 0):
                return xp.vstack([self.vecmat(row) for row in x])
            else:
                return self.rmatmat(x)

    def dot(self, x: npt.ArrayLike) -> npt.NDArray:
        """Numpy-like ndarray.dot function for N-D inputs.

        Performs an N-D matrix dot product with the programmed matrix. For >=2D
        inputs this will decompose the matrix into a series for 1D inputs or use a
        (generally faster) matrix-matrix approximation if possible given the simulation
        parameters. In the error free case this should be identical to ``A.dot(x)`` or
        ``np.dot(A,x)`` where A is the numpy array programmed with set_matrix().

        Args:
            x: An N-D numpy-like array to be multiplied.

        Returns:
            An N-D numpy-like array result.
        """
        x = self._ensure_data_format(x)

        # Technically ndim=2 (N,1) inputs are also "vectors" but by they require a
        # different output shape which is handled correctly if they go through the
        # matmat path instead.
        if x.ndim == 1:
            return self.matvec(x)
        else:
            # Stacking fails for shape (X, 0), revert to matmat which handles this
            # Empty matix is weird so ignoring user preference here
            if not (self.fast_matmul or x.shape[1] == 0):
                return xp.hstack(
                    [
                        self.matvec(
                            col.reshape(
                                -1,
                            ),
                        ).reshape(-1, 1)
                        for col in x.T
                    ],
                )
            else:
                return self.matmat(x).transpose(
                    (x.ndim - 2, *np.arange(0, x.ndim - 2), x.ndim - 1),
                )

    def rdot(self, x: npt.ArrayLike) -> npt.NDArray:
        """Numpy-like ndarray.dot() function for N-D inputs.

        Performs an N-D matrix dot product with the programmed matrix. For >=2D
        inputs this will decompose the matrix into a series for 1D inputs or use a
        (generally faster) matrix-matrix approximation if possible given the simulation
        parameters. In the error free case this should be identical to ``x.dot(A)`` or
        ``np.dot(x, A)`` where A is the numpy array programmed with set_matrix().

        Args:
            x: An N-D numpy-like array to be multiplied.

        Returns:
            An N-D numpy-like array result.
        """
        x = self._ensure_data_format(x)

        # As with dot, sending all ndim == 2 to matmat fixes some shape inconsistency
        # when compared to numpy
        if x.ndim == 1:
            return self.vecmat(x)
        else:
            # Stacking fails for shape (0, X), revert to matmat which handles this
            # Empty matix is weird so ignoring user preference here
            if not (self.fast_matmul or x.shape[0] == 0):
                return xp.vstack([self.vecmat(row) for row in x])
            else:
                return self.rmatmat(x)

    def mat_multivec(self, vec: npt.ArrayLike) -> npt.NDArray:
        """Perform matrix-vector multiply on multiple analog vectors packed into the
        "vec" object. A single MVM op in the simulation models multiple MVMs in the
        physical hardware.

        The "vec" object will be reshaped into the following 2D shape: (Ncopy, N)
        where Ncopy is the number of input vectors packed into the MVM simulation
        and N is the length of a single input vector

        Args:
            vec: ...

        Raises:
            NotImplementedError: ...
            ValueError: ...

        Returns:
            NDArray: ...
        """
        vec = self._ensure_data_format(vec)

        if self.complex_valued:
            raise NotImplementedError(
                "MVM packing not supported for complex-valued MVMs",
            )

        # For consistency use run_xbar_mvm for both single and multiple cores
        # This means functions using mat_multivec can't use input scaling but this is
        # currently used exclusively for convolution which isn't user facing.
        if self.Ncores == 1:
            return self.cores[0][0].run_xbar_mvm(vec.flatten())

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
                i_start = np.sum(self.NcolsVec[:i]).astype(int)
                i_end = np.sum(self.NcolsVec[: i + 1]).astype(int)
                vec_i = vec[:, i_start:i_end].flatten()
                for j in range(self.num_cores_row):
                    j_start = int(np.sum(self.NrowsVec[:j]))
                    j_end = int(np.sum(self.NrowsVec[: j + 1]))
                    output_ij = self.cores[j][i].run_xbar_mvm(vec_i.copy())
                    output_i[:, j_start:j_end] = output_ij.reshape(
                        (Ncopy, j_start - j_end),
                    )
                output += output_i
            return self._convert_output_type(
                output.flatten(),
                self.output_type(vec.dtype),
            )

    def transpose(self) -> AnalogCore:
        return TransposedCore(parent=self)

    T = property(transpose)

    def __getitem__(self, item):
        rslice, cslice, full_mask, flatten = self._create_mask(item)
        if full_mask:
            return self
        return MaskedCore(self, rslice, cslice, flatten)

    def __setitem__(self, key, value):
        rslice, cslice, full_mask, _ = self._create_mask(key)
        expanded_mat = xp.asarray(self.get_matrix())
        expanded_mat[rslice, cslice] = xp.asarray(value)
        error_mask = None if full_mask else (rslice, cslice)
        self.set_matrix(expanded_mat, error_mask=error_mask)

    def _create_mask(self, item) -> tuple[slice, slice, bool, bool]:
        """Converts an input item int, slice, tuple of int/slices into a tuple of slices."""
        if not isinstance(item, tuple):
            # Single value passed, convert to tuple then pad with empty slice
            item = (item, slice(None, None, None))
        if not all(isinstance(i, (int, slice)) for i in item):
            raise TypeError("Index must be int, slice or tuple of those types")
        if len(item) > 2:
            # Case of length one is accounted for above
            raise ValueError("Index must be of length 1 or 2")

        # Numpy flattens arrays if any of the slices are integers
        flatten = any(isinstance(i, int) for i in item)

        # Tracks if the mask covers the full matrix, if so we can just ignore the mask
        full_mask = False

        # For an example with a negative step size and None initialized as start and
        # stop for slice, indices incorrectly makes the stop index -1 when trying to
        # encapsulate the whole range of what is being sliced.
        #
        # For example, for a slice x = slice(None, None, -1), slice(*x.indices(5))
        # results in slice(4,-1,-1), which is incorrect, because the slice always
        # results in an empty value. It must be noted that stop is not inclusive,
        # and indices is attempting to include the 0th index by making the stop index
        # one lower. This is flawed due to a -1 index being also the len() - 1 index
        # of some structure. Changing to None ensures the encapsulation of the whole
        # structure.
        #
        # NOTE: we may need to add to each conditional 'self.*_slice.stop < 0' in the
        # case where y in .indices(y) is less than the total size of the structure.

        rslice, cslice = item
        if isinstance(rslice, int):
            if rslice < 0:
                rslice = slice(self.shape[0] + rslice, self.shape[0] + rslice + 1)
            else:
                rslice = slice(rslice, rslice + 1)
        else:
            rslice = slice(*rslice.indices(self.shape[0]))
            if rslice.step < 0:
                rslice = slice(rslice.start, None, rslice.step)

            full_mask = (
                len(range(*rslice.indices(self.shape[0]))) == self.shape[0]
                and rslice.step > 0
            )

        if self.ndim == 1:
            cslice = None
        else:
            if isinstance(cslice, int):
                if cslice < 0:
                    cslice = slice(self.shape[1] + cslice, self.shape[1] + cslice + 1)
                else:
                    cslice = slice(cslice, cslice + 1)
            else:
                cslice = slice(*cslice.indices(self.shape[1]))
                if cslice.step < 0:
                    cslice = slice(cslice.start, None, cslice.step)

            full_mask &= (
                len(range(*cslice.indices(self.shape[1]))) == self.shape[1]
                and cslice.step > 0
            )

        return (rslice, cslice, full_mask, flatten)

    # Output dtype methods
    def _native_dtype(self, input_dtype: npt.DTypeLike) -> npt.DTypeLike:
        return xp.promote_types(self.dtype, input_dtype)

    def _matrix_dtype(self, input_dtype: npt.DTypeLike) -> npt.DTypeLike:
        return self.dtype

    def _input_dtype(self, input_dtype: npt.DTypeLike) -> npt.DTypeLike:
        return input_dtype

    def _explicit_dtype(self, input_dtype: npt.DTypeLike) -> npt.DTypeLike:
        return self._output_type

    def _convert_output_type(self, output, output_dtype):
        if xp.issubdtype(output_dtype, xp.integer):
            output = xp.rint(output)
        return output.astype(output_dtype)

    def _ensure_data_format(self, array):
        array = xp.asarray(array)
        if xp.issubdtype(self.dtype, xp.integer):
            array = array.astype(xp.float32)
        return array

    @staticmethod
    def _set_limits_percentile(constraints, input_, reset=False):
        """Set the min and max of the params object based on input data using
        the percentile option, if the min and max have not been set yet
        If min and max are already set, this function does nothing if reset=False
        constraints must have the following params:
            min: float
            max: float
            percentile: float.
        """
        input_ = xp.asarray(input_)

        if (constraints.min is None or constraints.max is None) or reset:
            if constraints.percentile >= 1.0:
                X_max = xp.max(xp.abs(input_))
                X_max *= constraints.percentile
                min_ = -X_max
                max_ = X_max

            elif constraints.percentile < 1.0:
                X_posmax = xp.percentile(input_, 100 * constraints.percentile)
                X_negmax = xp.percentile(input_, 100 - 100 * constraints.percentile)
                X_max = xp.max(xp.abs(xp.array([X_posmax, X_negmax])))
                min_ = -X_max
                max_ = X_max

        # Ensure min_ and max_ aren't the same for uniform inputs
        if min_ == max_:
            eps = xp.finfo(float).eps
            min_ -= eps
            max_ += eps
        return (min_, max_)

    @staticmethod
    def _make_core(params):
        def inner_factory():
            return NumericCore(params)

        def inner_factory_independent():
            new_params = params.copy()
            return NumericCore(new_params)

        # set the outer core type
        if params.core.style == CoreStyle.OFFSET:
            return OffsetCore(inner_factory, params)

        elif params.core.style == CoreStyle.BALANCED:
            return BalancedCore(inner_factory, params)

        elif params.core.style == CoreStyle.BITSLICED:
            return BitslicedCore(inner_factory_independent, params)

        else:
            raise ValueError(
                "Core type "
                + str(params.core.style)
                + " is unknown: should be OFFSET, BALANCED, or BITSLICED",
            )

    def __matmul__(self, other: npt.ArrayLike) -> npt.NDArray:
        other = xp.asarray(other)
        return self.matmul(other)

    def __rmatmul__(self, other: npt.ArrayLike) -> npt.NDArray:
        other = xp.asarray(other)
        return self.rmatmul(other)

    def __repr__(self) -> str:
        prefix = "AnalogCore("
        mid = np.array2string(self.get_matrix(), separator=", ", prefix=prefix)
        suffix = ")"
        return prefix + mid + suffix

    def __str__(self) -> str:
        return self.get_matrix().__str__()

    def __array__(self) -> npt.NDArray:
        return self.get_matrix()


class TransposedCore(AnalogCore):
    def __init__(self, parent):
        self.parent = parent

        self.shape = tuple(reversed(self.parent.shape))
        self.ndim = 2

    @property
    def rslice(self):
        return self.parent.rslice

    @property
    def cslice(self):
        return self.parent.cslice

    @property
    def fast_matmul(self):
        return self.parent.fast_matmul

    # Mostly needed because of how some tests are written, potentially could be removed
    @property
    def cores(self):
        return self.parent.cores

    def transpose(self) -> AnalogCore:
        return self.parent

    T = property(transpose)

    def get_matrix(self) -> npt.NDArray:
        return self.parent.get_matrix().T

    def set_matrix(
        self,
        matrix: npt.ArrayLike,
        verbose: bool = False,
        error_mask: tuple[slice, slice] | None = None,
    ) -> None:
        matrix = xp.asarray(matrix)
        self.parent.set_matrix(matrix.T, verbose=verbose, error_mask=error_mask)

    def matvec(self, vec: npt.ArrayLike) -> npt.NDArray:
        vec = xp.asarray(vec)

        if vec.shape != (self.shape[1],) and vec.shape != (self.shape[1], 1):
            raise ValueError(
                f"Dimension mismatch: {self.shape}, {vec.shape}",
            )

        return self.parent.vecmat(vec)

    def matmat(self, mat: npt.ArrayLike) -> npt.NDArray:
        mat = xp.asarray(mat)
        if mat.ndim != 2:
            raise ValueError(
                f"Expected 2d matrix, got {mat.ndim}d input",
            )

        if self.shape[1] != mat.shape[0]:
            raise ValueError(
                f"Dimension mismatch: {self.shape}, {mat.shape}",
            )

        return self.parent.rmatmat(mat.T).T

    def vecmat(self, vec: npt.ArrayLike) -> npt.NDArray:
        vec = xp.asarray(vec)
        if vec.shape != (self.shape[0],) and vec.shape != (1, self.shape[0]):
            raise ValueError(
                f"Dimension mismatch: {vec.shape}, {self.shape} ",
            )

        return self.parent.matvec(vec)

    def rmatmat(self, mat: npt.ArrayLike) -> npt.NDArray:
        mat = xp.asarray(mat)

        if mat.ndim != 2:
            raise ValueError(
                f"Expected 2d matrix, got {mat.ndim}d input",
            )

        if self.shape[0] != mat.shape[1]:
            raise ValueError(
                f"Dimension mismatch: {mat.shape}, {self.shape}",
            )

        return self.parent.matmat(mat.T).T

    def __repr__(self) -> str:
        prefix = "TransposedCore("
        mid = np.array2string(self.get_matrix(), separator=", ", prefix=prefix)
        suffix = ")"
        return prefix + mid + suffix


class MaskedCore(AnalogCore):
    def __init__(self, parent, rslice, cslice, flatten):
        self.parent = parent
        self.rslice = rslice
        self.cslice = cslice

        rows = len(range(*rslice.indices(parent.shape[0])))

        cols = 0
        if self.parent.ndim == 2:
            cols = len(range(*cslice.indices(parent.shape[1])))

        self.shape = (rows, cols)
        self.ndim = 2
        if flatten:
            self.shape = (np.max(self.shape),)
            self.ndim = 1

    @property
    def fast_matmul(self) -> bool:
        return self.parent.fast_matmul

    # Mostly needed because of how some tests are written, potentially could be removed
    @property
    def cores(self):
        return self.parent.cores

    def transpose(self) -> AnalogCore:
        # Numpy defines the transpose of a 1D matrix as itself
        if self.ndim == 1:
            return self
        else:
            return TransposedCore(parent=self)

    T = property(transpose)

    def get_matrix(self) -> npt.NDArray:
        if self.ndim == 1 or self.parent.ndim == 1:
            return self.parent.get_matrix()[self.rslice].flatten()
        else:
            return self.parent.get_matrix()[self.rslice, self.cslice]

    def set_matrix(self, matrix: npt.ArrayLike, verbose: bool = False, error_mask=None):
        # TODO: Do we need to do anything with error_mask here?
        expanded_mat = xp.asarray(self.parent.get_matrix())
        expanded_mat[self.rslice, self.cslice] = xp.asarray(matrix)
        self.parent.set_matrix(
            expanded_mat,
            verbose=verbose,
            error_mask=(self.rslice, self.cslice),
        )

    def matvec(self, other: npt.ArrayLike) -> npt.NDArray:
        other = xp.asarray(other)

        if other.shape != (self.shape[1],) and other.shape != (self.shape[1], 1):
            raise ValueError(
                f"Dimension mismatch: {self.shape}, {other.shape}",
            )

        vec_in = xp.zeros(self.parent.shape[1], dtype=other.dtype)
        vec_in[self.cslice] = other.flatten()

        vec_out = self.parent.matvec(vec_in)
        return vec_out[self.rslice]

    def matmat(self, other: npt.ArrayLike) -> npt.NDArray:
        other = xp.asarray(other)

        if other.ndim != 2:
            raise ValueError(
                f"Expected 2d matrix, got {other.ndim}d input",
            )

        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Dimension mismatch: {self.shape}, {other.shape}",
            )

        # For row slices we're just ignoring the outputs corrosponding to the
        #   out-of-slice rows
        # For col slices we're just leaving empty entires in the input matrix
        #   corrosponding to missing rows
        mat_in = xp.zeros((self.parent.shape[1], other.shape[1]), dtype=other.dtype)
        for i in range(self.parent.shape[1])[self.cslice]:
            mat_in[i] = other[(i - self.cslice.start) // self.cslice.step]
        mat_out = self.parent.matmat(mat_in)
        return mat_out[self.rslice]

    def vecmat(self, other: npt.ArrayLike) -> npt.NDArray:
        other = xp.asarray(other)

        if other.shape != (self.shape[0],) and other.shape != (1, self.shape[0]):
            raise ValueError(
                f"Dimension mismatch: {other.shape}, {self.shape} ",
            )

        vec_in = xp.zeros(self.parent.shape[0], dtype=other.dtype)
        vec_in[self.rslice] = other.flatten()

        vec_out = self.parent.vecmat(vec_in)
        return vec_out[self.cslice]

    def rmatmat(self, other: npt.ArrayLike) -> npt.NDArray:
        other = xp.asarray(other)

        if other.ndim != 2:
            raise ValueError(
                f"Expected 2d matrix, got {other.ndim}d input",
            )

        if self.shape[0] != other.shape[1]:
            raise ValueError(
                f"Dimension mismatch: {other.shape}, {self.shape}",
            )

        mat_in = xp.zeros((other.shape[0], self.parent.shape[0]), dtype=other.dtype)
        for i in range(self.parent.shape[0])[self.rslice]:
            mat_in.T[i] = other.T[(i - self.rslice.start) // self.rslice.step]

        mat_out = self.parent.rmatmat(mat_in)
        return mat_out[:, self.cslice]

    def __repr__(self) -> str:
        prefix = "MaskedCore("
        mid = np.array2string(self.get_matrix(), separator=", ", prefix=prefix)
        suffix = ")"
        return prefix + mid + suffix
