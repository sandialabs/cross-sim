#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from .iarray import IArray
from simulator.backend import ComputeBackend
import numpy.typing as npt
import typing

xp = ComputeBackend()


class UncoupledNonlinearArray(IArray):
    """This class models a current-mode MVM array with devices that have
    nonlinear I-V curves.

    It is a simple solver to be used with the assumption of no parasitics.
    For current-mode MVM, that means it does not have to be an iterative solver.
    Can handle both interleaved and non-interleaved array.
    """

    def __init__(
        self,
        params,
        device=None,
    ) -> None:
        """Initialized a current mode array with nonlinear I-V curves."""
        super().__init__(params, device=device)

    def solve_mvm(
        self,
        vector: npt.NDArray,
        matrix: npt.NDArray,
        matrix_neg: typing.Optional[npt.NDArray] = None,
        row_in: bool = True,
    ) -> npt.NDArray:
        """Calculates the MVM result including parasitic resistance."""
        if self.enable_fast_nonlinear_IV:
            # Compute fast nonlinear current sums inside device I-V nonlinearity
            # model
            Icols = self._nonlinear_current_sum(matrix, vector, row_in=row_in)
            if matrix_neg is not None:
                Icols -= self._nonlinear_current_sum(matrix_neg, vector, row_in=row_in)

        else:
            # Compute voltage drop across every element of the array
            dV = self._init_dV(vector, matrix, row_in=row_in)

            # Compute cell currents, including the effect of I-V nonlinearity
            Ires = self._cell_currents(matrix, dV)

            # Handle interleaved
            if matrix_neg is not None:
                Ires -= self._cell_currents(matrix_neg, dV)

            # Sum the cell currents
            Icols = xp.sum(Ires, axis=1)

            Icols = self._post_process(
                Icols, input_dim=len(vector.shape), row_in=row_in
            )

        return Icols
