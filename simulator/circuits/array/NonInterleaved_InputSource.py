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

xp = ComputeBackend()


class NoninterleavedInputSourceArray(IArray):
    """This class models an array where:
    1) The current through the memory cells is sourced from the same
        interconnect that carries the input signal.
    2) There is no local interleaving of memory cells for positive and negative
        weights. If differential cells are used, the subtraction occurs outside
        of the array model (e.g. in the peripheral circuits or in digital).
    """

    def __init__(
        self,
        params,
        device=None,
    ) -> None:
        """Initialized a non-interleaved array with a single source."""
        super().__init__(params, device=device)
        self.interleaved = False

    def solve_mvm(  # noqa:C901
        self,
        vector: npt.NDArray,
        matrix: npt.NDArray,
        row_in: bool = True,
    ) -> npt.NDArray:
        """Calculates the MVM result including parasitic resistance."""
        # Initialize error and number of iterations
        Verr = 1e9
        Niters = 0

        # Compute element-wise voltage drops and currents
        dV0 = self._init_dV(vector, matrix, row_in=row_in)
        Ires = self._cell_currents(matrix, dV0)
        dV = dV0.copy()

        # Create parasitic resistance matrices if they don't exist yet
        if self.Rp_in_mat is None or self.Rp_out_mat is None:
            if row_in:
                Rp_in = self.Rp_row_norm
                Rp_out = self.Rp_col_norm
                Rp_in_terminal = self.Rp_row_terminal_norm
                Rp_out_terminal = self.Rp_col_terminal_norm
            else:
                Rp_in = self.Rp_col_norm
                Rp_out = self.Rp_row_norm
                Rp_in_terminal = self.Rp_col_terminal_norm
                Rp_out_terminal = self.Rp_row_terminal_norm

            # Account for terminal resistance; if zero, parasitic resistance is
            # a scalar
            self.Rp_in_mat = Rp_in
            self.Rp_out_mat = Rp_out
            if Rp_in_terminal > 0:
                self.Rp_in_mat = Rp_in * xp.ones((matrix.shape[0], matrix.shape[1]))
                self.Rp_in_mat[:, 0] += Rp_in_terminal
            if Rp_out_terminal > 0:
                self.Rp_out_mat = Rp_out * xp.ones((matrix.shape[0], matrix.shape[1]))
                self.Rp_out_mat[:, -1] += Rp_out_terminal

        # Iteratively calculate parasitics and update device currents
        while Verr > self.Verr_th and Niters < self.Niters_max:
            # Calculate parasitic voltage drops
            Isum_col = xp.cumsum(Ires, 1)
            Isum_row = xp.cumsum(Ires[::-1], 0)[::-1]
            if self.useMask:
                Isum_col *= self.mask
                Isum_row *= self.mask

            if len(Isum_col.shape) == 4 and isinstance(self.Rp_out_mat, xp.ndarray):
                Vdrops_col = xp.cumsum(
                    (self.Rp_out_mat[:, :, None, None] * Isum_col)[:, ::-1], 1
                )[:, ::-1]
            else:
                Vdrops_col = xp.cumsum((self.Rp_out_mat * Isum_col)[:, ::-1], 1)[
                    :, ::-1
                ]

            if len(Isum_row.shape) == 4 and isinstance(self.Rp_in_mat, xp.ndarray):
                Vdrops_row = xp.cumsum(self.Rp_in_mat[:, :, None, None] * Isum_row, 0)
            else:
                Vdrops_row = xp.cumsum(self.Rp_in_mat * Isum_row, 0)

            Vpar = Vdrops_col + Vdrops_row

            # Calculate the error for the current estimate of memristor currents
            VerrMat = dV0 - Vpar - dV

            # Evaluate overall error
            Verr = self._error_metric(VerrMat)
            if Verr < self.Verr_th:
                break

            # Update memristor currents for the next iteration
            dV += self.gamma * VerrMat
            Ires = self._cell_currents(matrix, dV)
            Niters += 1

        # Calculate the summed currents on the columns
        Icols = xp.sum(Ires, axis=1)
        Icols = self._post_process(Icols, input_dim=len(vector.shape), row_in=row_in)

        # Should add some more checks here on whether the results of this
        # calculation are erroneous even if it converged
        if Verr > self.Verr_th:
            raise RuntimeError("Parasitic resistance too high: could not converge!")
        return Icols
