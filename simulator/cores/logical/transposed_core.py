#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""A Numpy-like interface for interacting with analog MVM operations.

TransposedCore is the primary interface for AnalogCore transposing.
AnalogCores behave as if they are Numpy arrays for 1 and 2 dimensional matrix
multiplication (left and right), transpostion and slice operations. This is
intended to provide an interface to interact with Transposed AnalogCores
"""

from __future__ import annotations

import numpy.typing as npt
from simulator.cores.logical.analog_core import AnalogCore
from simulator.backend import ComputeBackend
from simulator.parameters.mapping import CoreMappingParameters

xp = ComputeBackend()


class TransposedCore(AnalogCore):
    """Primary simulation object for Transposed AnalogCores."""

    def __init__(self, parent: AnalogCore):
        """Initializes a transposed core.

        This class should never be constructed directly. Instead, call
        .transpose() or .T on an analog core.
        """
        self.parent = parent

        self._shape = tuple(reversed(self.parent.shape))
        self.ndim = 2

    @property
    def mapping(self) -> CoreMappingParameters:
        """Returns the mapping parameters for the core."""
        core_mapping_class = self.parent.mapping.__class__
        kwargs = self.parent.mapping.as_dict()
        mvm = kwargs.pop("mvm")
        vmm = kwargs.pop("vmm")
        kwargs["mvm"] = vmm
        kwargs["vmm"] = mvm
        return core_mapping_class(**kwargs)

    @property
    def rslice(self):
        """Returns the row slice of the core."""
        return self.parent.rslice

    @property
    def cslice(self):
        """Returns the col slice of the core."""
        return self.parent.cslice

    @property
    def fast_matmul(self):
        """Returns whether fast matmul will be used."""
        return self.parent.fast_matmul

    def transpose(self) -> AnalogCore:
        """Returns a transposed view of the core."""
        return self.parent

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
        return self.parent.read_matrix(apply_errors=apply_errors).T

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
        self.parent.set_matrix(
            matrix=matrix.T,
            apply_errors=apply_errors,
            error_mask=error_mask,
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
        return self.parent.vecmat(other)

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

        order = (*range(other.ndim - 2), other.ndim - 1, other.ndim - 2)
        return xp.transpose(self.parent.rmatmat(xp.transpose(other, order)), order)

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

        return self.parent.matvec(other)

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

        order = (*range(other.ndim - 2), other.ndim - 1, other.ndim - 2)
        return xp.transpose(self.parent.matmat(xp.transpose(other, order)), order)

    def __repr__(self) -> str:
        """Return repr(self)."""
        prefix = "TransposedCore("
        mid = xp.array2string(
            self.read_matrix(apply_errors=False),
            separator=", ",
            prefix=prefix,
        )
        suffix = ")"
        return prefix + mid + suffix
