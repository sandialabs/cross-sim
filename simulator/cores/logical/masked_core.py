#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""A Numpy-like interface for interacting with analog MVM operations.

MaskedCore is the primary interface for AnalogCore masking. AnalogCores
behave as if they are Numpy arrays for 1 and 2 dimensional matrix
multiplication (left and right), transpostion and slice operations. This is
intended to provide an interface to interact with masked and sliced AnalogCores
"""

from __future__ import annotations

import numpy.typing as npt
from simulator.cores.logical.transposed_core import TransposedCore
from simulator.cores.logical.analog_core import AnalogCore
from simulator.parameters.mapping import CoreMappingParameters

from simulator.backend import ComputeBackend

xp = ComputeBackend()


class MaskedCore(AnalogCore):
    """Primary simulation object for Masked AnalogCores."""

    def __init__(
        self,
        parent: AnalogCore,
        rslice: slice | int,
        cslice: slice | int,
        flatten: int,
    ):
        """Initializes a masked core.

        This class should never be constructed directly. Instead, apply a mask
        to an analog core.
        """
        self.parent = parent
        self.rslice = rslice
        self.cslice = cslice

        if isinstance(rslice, int):
            rows = 1
        else:
            rows = len(range(*rslice.indices(parent.shape[0])))

        cols = 0
        if self.parent.ndim == 2:
            if isinstance(cslice, int):
                cols = 1
            else:
                cols = len(range(*cslice.indices(parent.shape[1])))

        self.ndim = self.parent.ndim - flatten
        self._shape = (rows, cols)
        for _ in range(flatten):
            self._shape = (xp.max(self.shape),)

    @property
    def mapping(self) -> CoreMappingParameters:
        """Returns the mapping parameters for the core."""
        return self.parent.mapping

    @property
    def fast_matmul(self) -> bool:
        """Returns whether fast matmul will be used."""
        return self.parent.fast_matmul

    def transpose(self) -> AnalogCore:
        """Returns a transposed view of the core."""
        # Numpy defines the transpose of a 1D matrix as itself
        if self.ndim == 1:
            return self
        else:
            return TransposedCore(parent=self)

    @property
    def T(self) -> AnalogCore:
        """Returns a transposed view of the core."""
        return self.transpose()

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
        if self.ndim == 1 or self.parent.ndim == 1:
            return self.parent.read_matrix(apply_errors=apply_errors)[
                self.rslice
            ].flatten()
        else:
            return self.parent.read_matrix(apply_errors=apply_errors)[
                self.rslice,
                self.cslice,
            ]

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
        # TODO: Do we need to do anything with error_mask here?
        expanded_mat = self.parent.read_matrix(apply_errors=apply_errors)
        expanded_mat[self.rslice, self.cslice] = xp.asarray(matrix)
        self.parent.set_matrix(
            matrix=expanded_mat,
            apply_errors=apply_errors,
            error_mask=(self.rslice, self.cslice),
        )

    def matvec(self, other: npt.ArrayLike) -> npt.NDArray:
        """Perform matrix-vector (Ax = b) multiply on programmed vector (1D).

        Primary simulation function for 1D inputs. Transforms the vector for
        analog simulation and calls the underlying core simulation functions for
        each sub-core. Without errors this should be identical to
        ``A.matmul(vec)`` or ``A @ vec`` where A is the numpy array programmed
        with set_matrix().

        Args:
            other: 1D Numpy-like array to be multiplied.

        Returns:
            1D Numpy-like array result of matrix-vector multiplication.
        """
        other = xp.asarray(other)

        if other.shape != (self.shape[1],) and other.shape != (self.shape[1], 1):
            raise ValueError(
                "Operands could not be broadcast together with ",
                f"shapes {self.shape}, {other.shape}",
            )

        vec_in = xp.zeros(self.parent.shape[1], dtype=other.dtype)
        vec_in[self.cslice] = other.flatten()

        vec_out = self.parent.matvec(vec_in)
        return vec_out[self.rslice]

    def matmat(self, other: npt.ArrayLike) -> npt.NDArray:
        """Perform right matrix-matrix (AX=B) multiply on programmed 2D matrix.

        Primary simulation function for 2D inputs. Transforms the matrix for
        analog simulation and calls the underlying core simulation functions for
        each sub-core. Without errors this should be identical to
        ``A.matmul(mat)`` or ``A @ mat`` where A is the numpy array programmed
        with set_matrix().

        Args:
            other: 2D Numpy-like array to be multiplied.

        Returns:
            2D Numpy-like array result of matrix-matrix multiplication.
        """
        other = xp.asarray(other)

        if self.shape[1] != other.shape[-2]:
            raise ValueError(
                "Operands could not be broadcast together with ",
                f"shapes {self.shape}, {other.shape}",
            )

        # For row slices we're just ignoring the outputs corrosponding to the
        #   out-of-slice rows
        # For col slices we're just leaving empty entires in the input matrix
        #   corrosponding to missing rows
        mat_in = xp.zeros(
            (*other.shape[:-2], self.parent.shape[1], other.shape[-1]),
            dtype=other.dtype,
        )
        mat_in[..., self.cslice, :] = other
        mat_out = self.parent.matmat(mat_in)
        mat_product = mat_out[..., self.rslice, :]
        return mat_product

    def vecmat(self, other: npt.ArrayLike) -> npt.NDArray:
        """Perform vector-matrix (xA = b) multiply on programmed vector (1D).

        Primary simulation function for 1D inputs. Transforms the vector for
        analog simulation and calls the underlying core simulation functions for
        each sub-core. Without errors this should be identical to
        ``vec.matmul(A)`` or ``vec @ A`` where A is the numpy array programmed
        with set_matrix().

        Args:
            other: 1D Numpy-like array to be multiplied.

        Returns:
            1D Numpy-like array result of vector-matrix multiplication.
        """
        other = xp.asarray(other)

        if other.shape != (self.shape[0],) and other.shape != (1, self.shape[0]):
            raise ValueError(
                "Operands could not be broadcast together with ",
                f"shapes {self.shape}, {other.shape}",
            )

        vec_in = xp.zeros(self.parent.shape[0], dtype=other.dtype)
        vec_in[self.rslice] = other.flatten()

        vec_out = self.parent.vecmat(vec_in)
        return vec_out[self.cslice]

    def rmatmat(self, other: npt.ArrayLike) -> npt.NDArray:
        """Perform left matrix-matrix (XA=B) multiply on programmed matrix (2D).

        Primary simulation function for 2D inputs. Transforms the matrix for
        analog simulation and calls the underlying core simulation functions for
        each sub-core.  Without errors this should be identical to
        ``mat.matmul(A)`` or ``mat @ A`` where A is the numpy array programmed
        with set_matrix().

        Args:
            other: 2D Numpy-like array to be multiplied.

        Returns:
            2D Numpy-like array result of matrix-matrix multiplication.
        """
        other = xp.asarray(other)

        if self.shape[0] != other.shape[-1]:
            raise ValueError(
                "Operands could not be broadcast together with ",
                f"shapes {self.shape}, {other.shape}",
            )

        mat_in = xp.zeros((*other.shape[:-1], self.parent.shape[0]), dtype=other.dtype)
        mat_in[..., self.rslice] = other
        mat_out = self.parent.rmatmat(mat_in)
        mat_product = mat_out[..., self.cslice]
        return mat_product

    def __repr__(self) -> str:
        """Returns repr(self)."""
        prefix = "MaskedCore("
        mid = xp.array2string(
            self.read_matrix(apply_errors=False),
            separator=", ",
            prefix=prefix,
        )
        suffix = ")"
        return prefix + mid + suffix
