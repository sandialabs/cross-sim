#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from __future__ import annotations

import logging
import textwrap
import warnings
from typing import Protocol

import numpy as np
import numpy.typing as npt

from simulator.backend.compute import ComputeBackend
from simulator.parameters.core.analog_core import (
    PartitionStrategy,
    PartitionMode,
    OutputDTypeStrategy,
)

xp: np = ComputeBackend()
log = logging.getLogger(__name__)


def check_logging(ignore_check: bool = False):
    """Checks if logging has been configured. If not, warns user."""
    if ignore_check is False:
        return

    logging_configured = logging.getLogger().hasHandlers()
    if not logging_configured:
        message = textwrap.dedent(
            """
            Logging is not configured, some critical warnings and error messages may not be shown.

            Logging can be enabled by adding the following to your code:

                import logging
                logging.basicConfig(level=logging.WARNING)

                # Your code here
                # ...

            To surpress this warning, set ignore_logging_check=True in simulation parameters

            For more information on how to configure the logger, please reference the
            official Python documentation:
                https://docs.python.org/3/library/logging.html
            """,  # noqa: E501
        )
        warnings.warn(message=message, category=RuntimeWarning, stacklevel=1)


def reshaped_complex_matrix(matrix: npt.NDArray, is_complex: bool) -> npt.NDArray:
    """Reshapes matrix to a usable form if it is complex valued."""
    if is_complex:
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
    return mcopy


def is_full_mask(
    shape: tuple[int, ...],
    ndim: int,
    item: slice | int,
) -> tuple[bool, int]:
    """Checks if the mask being applied to the core spans the full matrix."""
    if not all(isinstance(i, (int, slice)) for i in item):
        raise TypeError("Index must be int, slice or tuple of those types")
    if len(item) > 2:
        # Case of length one is accounted for above
        raise ValueError("Index must be of length 1 or 2")
    if len(shape) > 2:
        raise ValueError("Shape can not be greater than 2.")

    # Numpy flattens arrays if any of the slices are integers
    flatten = sum(1 for i in item if isinstance(i, int))

    # Tracks if the mask covers the full matrix,
    # if so we can just ignore the mask
    full_mask = False
    rslice, cslice = item
    if not isinstance(rslice, int):
        full_mask = (
            len(range(*rslice.indices(shape[0]))) == shape[0]
            and rslice.step is not None
            and rslice.step > 0
        )

    if ndim != 1:
        if isinstance(cslice, int):
            if cslice < 0:
                cslice = slice(shape[1] + cslice, shape[1] + cslice + 1)
            else:
                cslice = slice(cslice, cslice + 1)

        full_mask &= (
            len(range(*cslice.indices(shape[1]))) == shape[1]
            and cslice.step is not None
            and cslice.step > 0
        )

    return (full_mask, flatten)


def partition(
    num_cores: int,
    num_elements: int,
    max_partition: int,
    partition_priority: list[int],
    partition_strategy: PartitionStrategy,
) -> tuple[npt.NDArray, list[tuple[int, int, int]]]:
    """Returns how to partition the matrix across the subcores.

    Uses the following priority:
    1) If rows/cols can be partition evenly partition evenly
    2) If row/col_partition priority is defined check if nrow/ncol is
        a multiple of N*max/partition so all but one core has Nmax*
        This is useful (with col_partition_priority = [2, 4]) so that
        convolution blocks in depthwise convolutions are not partition.
        Maybe not totally robust
    3) Otherwise partition based on row/col_partition_strategy
        (max or even)

    Args:
        num_cores: The number of cores to use.
        num_elements: The number of elements to partition
        max_partition: The maximum size of the partition
        partition_priority: Ordered list of divisors to check
        partition_strategy: Strategy to use while creating partitions

    Returns:
        A tuple with the following types:
            xp.array: Array containing how many elements are in each partition
                (e.g. 20 elements, max size 8, with EQUAL strategy would give
                 a vector [7, 7, 6]
                )
            list[tuple[int, int, int]]: List of tuples for each partition in the
                form partition_index, partition_slice_start, partition_slice_end
    """
    if num_elements % num_cores == 0:
        partition_vec = (num_elements // num_cores) * xp.ones(
            num_cores,
            dtype=xp.int32,
        )
    else:
        prio_partition = True in (
            ((num_elements % (max_partition / div)) == 0) for div in partition_priority
        )

        if prio_partition or partition_strategy == PartitionStrategy.MAX:
            elements_per_core = max_partition
        else:
            elements_per_core = xp.round(num_elements / num_cores).astype(xp.int32)

        partition_vec = elements_per_core * xp.ones(
            num_cores,
            dtype=xp.int32,
        )
        partition_vec[-1] = num_elements - (num_cores - 1) * elements_per_core

    # Might not need these now that the cores are stored in a 1D array
    # Precompute a list of row/col partition information
    # as (id, start, end)
    # This is used to aggreagate partitions in every other function
    partition_bounds: list[tuple[int, int, int]] = [
        (
            e,
            xp.sum(partition_vec[:e]),
            xp.sum(partition_vec[: e + 1]),
        )
        for e in range(num_cores)
    ]

    return partition_vec, partition_bounds


class _OutputDTypeResolver(Protocol):
    """Protocol used by to type hint output dtype resolvers."""

    def __call__(
        self,
        matrix_dtype: npt.DTypeLike,
        input_dtype: npt.DTypeLike,
    ) -> npt.DTypeLike:
        """Resolves the output DType used by CrossSim.

        The output dtype resolver used is dependent on the strategy specified in
        simulation parameters. The following strategies are available:
          - Native: Use the backend's native type dtype promotion system.
          - Matrix: Use the dtype of the matrix.
          - Input: Use the dtype of the input.
          - Float16: Resolve as a float16
          - Float32: Resolve as a float32
          - Float64: Resolve as a float64

        Args:
            matrix_dtype: DType of the weight matrix.
            input_dtype: DType of the input.

        Returns:
            npt.DTypeLike: DType to be used for CrossSim's output dtype.
        """
        pass


def create_output_dtype_resolver(strategy: OutputDTypeStrategy) -> _OutputDTypeResolver:
    """Creates a function that resolves CrossSim's output dtype.

    Args:
        strategy: The strategy to use to resolve the output dtype.

    Returns:
        _OutputDTypeResolver: Function that returns a DType, given a matrix and
            input dtype.
    """
    # set the output type for this core
    if strategy == OutputDTypeStrategy.NATIVE:
        return lambda matrix_dtype, input_dtype: xp.promote_types(
            matrix_dtype,
            input_dtype,
        )
    elif strategy == OutputDTypeStrategy.MATRIX:
        return lambda matrix_dtype, input_dtype: matrix_dtype
    elif strategy == OutputDTypeStrategy.INPUT:
        return lambda matrix_dtype, input_dtype: input_dtype
    elif strategy == OutputDTypeStrategy.FLOAT16:
        return lambda matrix_dtype, input_dtype: xp.float16
    elif strategy == OutputDTypeStrategy.FLOAT32:
        return lambda matrix_dtype, input_dtype: xp.float32
    elif strategy == OutputDTypeStrategy.FLOAT64:
        return lambda matrix_dtype, input_dtype: xp.float64
    else:
        raise NotImplementedError(f"Unknown output dtype strategy {strategy}")


def clean_error_mask(
    shape: tuple[int, int],
    rslice: slice | int,
    cslice: slice | int,
) -> tuple[int, int]:
    """Cleans an error mask for use of logical cores."""
    if isinstance(rslice, int):
        if rslice < 0:
            rslice = slice(shape[0] + rslice, shape[0] + rslice + 1)
        else:
            rslice = slice(rslice, rslice + 1)
    else:
        rslice = slice(*rslice.indices(shape[0]))
    if isinstance(cslice, int):
        if cslice < 0:
            cslice = slice(shape[1] + cslice, shape[1] + cslice + 1)
        else:
            cslice = slice(cslice, cslice + 1)
    else:
        cslice = slice(*cslice.indices(shape[1]))
    return rslice, cslice


def set_error_mask(
    error_mask: tuple[slice, slice] | npt.ArrayLike,
    row_start: int,
    col_start: int,
    row_end: int,
    col_end: int,
) -> tuple[slice, slice]:
    """Internal function to set the error mask of a sub core."""
    row_mask = slice(
        error_mask[0].start - row_start if error_mask[0].start else None,
        (
            None
            if error_mask[0].step and error_mask[0].step < 0
            else error_mask[0].stop - row_start
        ),
        error_mask[0].step if error_mask[0].step else None,
    )
    col_mask = slice(
        error_mask[1].start - col_start if error_mask[1].start else None,
        (
            error_mask[1].stop - col_start
            if error_mask[1].step and error_mask[1].step >= 0
            else None
        ),
        error_mask[1].step if error_mask[1].step else None,
    )
    mask = (
        slice(*row_mask.indices(row_end - row_start)),
        slice(*col_mask.indices(col_end - col_start)),
    )
    return mask


def verify_partition_scheme(
    keys: list,
    partition_mode: PartitionMode,
    expected_shape: tuple[int, int],
):
    """Verifies that the partiiton scheme is compatible with provided subcores.

    Args:
        keys: A list of the keys of the subcores
        partition_mode: the partition mode of the core
        expected_shape: A shape that provides use with the number of cols,
            number of rows, and the number of cores (rows*cols)

    Raises:
        ValueError: if partition_mode is auto and multiple subcores provided
        ValueError: if partition_mode is coordinate and the # of partitioned
            matricies doesn't match the number of provided subcores
        ValueError: if partition mode is coordinate and the keys aren't all
            tuples of length 2
        ValueError: if partition mode is coordinate and the provided row coord
            is out of bounds
        ValueError: if partition mode is coordinate and the provided col coord
            is out of bounds
    """
    num_cores_row, num_cores_col = expected_shape
    num_cores = num_cores_row * num_cores_col
    if partition_mode == PartitionMode.AUTO:
        if list(keys) != ["default"]:
            raise ValueError(
                f"AnalogCore is allocating subcores to partitions in AUTO mode, which "
                f"requires a single key 'default' when defining subcores. "
                f"Found the the following subcore keys instead: {list(keys)}"
                f"\n\n"
                f"To fix this, define a single subcore with the key 'default', or "
                f"switch to COORDINATE mode",
            )

    elif partition_mode == PartitionMode.COORDINATE:
        if len(keys) != num_cores:
            raise ValueError(
                f"AnalogCore is allocating subcores in COORDINATE mode, which expects "
                f"one subcore for each partition with keys in (row, col) format. "
                f"The matrix has been partitioned into a "
                f"{num_cores_row} x {num_cores_col} grid, expecting a total of "
                f"{num_cores} subcores to be defined, "
                f"found a total of {len(keys)} instead.\n\n"
                f"Adjust the number of subcores defined to match the grid, or update "
                f"the AnalogCore's partitioning settings to fit the subcores provided.",
            )

        # Constructing the error message here is a bit involved
        # as it is user friendly to show all invalid keys at once
        # as compared to erroring out at the first invalid key that
        # is found
        bad_keys = []
        for key in keys:
            if not isinstance(key, tuple) or len(key) != 2:
                bad_keys.append((key, "Incorrect format"))
                continue
            if not (0 <= key[0] < num_cores_row):
                bad_keys.append((key, "Row out of bounds"))
            if not (0 <= key[1] < num_cores_col):
                bad_keys.append((key, "Col out of bounds"))
        if len(bad_keys) > 0:
            error_header = (
                "AnalogCore is allocating subcores in COORDINATE mode, which expects "
                "one subcore for each partition with keys in (row, col) format. "
                "\n\n"
                "However, the following keys are invalid:"
            )
            error_body = "\n".join(
                [f"\t{reason:<16}: {key = }" for key, reason in bad_keys],
            )
            explanations = (
                (
                    "Incorrect format",
                    "Keys must be in the form (row, col)",
                ),
                (
                    "Row out of bounds",
                    f"The matrix has been partitioned into {num_cores_row} rows. "
                    f"The row coordinate, r, must satisfy 0 <= r < {num_cores_row}",
                ),
                (
                    "Col out of bounds",
                    f"The matrix has been partitioned into {num_cores_col} rows. "
                    f"The col coordinate, c, must satisfy 0 <= c < {num_cores_col}",
                ),
            )
            error_tail = "Explanation of possible key failures:\n\t"
            error_tail += "\n\t".join(
                f"{error_type:<20}: {explanation}"
                for error_type, explanation in explanations
            )
            error_msg = "\n\n".join([error_header, error_body, error_tail])
            raise ValueError(error_msg)
