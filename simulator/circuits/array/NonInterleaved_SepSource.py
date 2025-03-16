#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from .iarray import IArray
from simulator.backend import ComputeBackend

xp = ComputeBackend()

class NonInterleaved_SeparateSource_Array(IArray):
    """
    This class models an array where:
    1) The current through the memory cells is sourced from a separate interconnect
        than the input signal, and this interconnect is parallel to the output
        summation column. The input signal is used to connect/disconnect the summation
        column to the current source column. 
    2) There is no local interleaving of memory cells for positive and negative weights.
        If differential cells are used, the subtraction occurs outside of the
        array model (e.g. in the peripheral circuits or in digital).
    """

    def __init__(
        self,
        params,
    ) -> None:
        super().__init__(params)
        self.interleaved = False
        self.selected_rows = params.xbar.array.parasitics.selected_rows
        self.rows_max = params.core.rows_max
        self.cols_max = params.core.cols_max


    def solve_mvm_parasitics(self, vector, matrix, row_in=True):

        # Initialize error and number of iterations
        Verr = 1e9
        Niters = 0

        # Compute element-wise voltage drops and currents
        dV0 = self._init_dV(vector, matrix, row_in=row_in)

        # If input is zero, device is gated off and has effectively zero conductance
        active_inputs = xp.abs(dV0) > 1e-9
        if len(dV0.shape) == 4 and len(matrix.shape) != 4:
            matrix = matrix[:,:,None,None] * active_inputs
        else:
            matrix *= active_inputs

        Ires = matrix * dV0
        dV = dV0.copy()

        # Create parasitic resistance matrices if they don't exist yet
        if self.Rp_in_mat is None or self.Rp_out_mat is None:

            #   Rp_out is the parasitic resistance of the interconnect that carries
            #       the output partial sums
            #   Rp_in is the parasitic resistance of the interconnect on other side,
            #       supplying or sinking current
            if row_in:
                Rp_in = self.Rp_row_norm
                Rp_out = self.Rp_col_norm
                Rp_in_terminal = self.Rp_row_terminal_norm
                Rp_out_terminal = self.Rp_col_terminal_norm
                Nmax_in = self.rows_max
            else:
                Rp_in = self.Rp_col_norm
                Rp_out = self.Rp_row_norm
                Rp_in_terminal = self.Rp_col_terminal_norm
                Rp_out_terminal = self.Rp_row_terminal_norm
                Nmax_in = self.cols_max

            self.Rp_in_mat = Rp_in * xp.ones((matrix.shape[0], matrix.shape[1]))
            self.Rp_out_mat = Rp_out * xp.ones((matrix.shape[0], matrix.shape[1]))

            # Account for a weight matrix smaller than the array size
            if matrix.shape[1] < Nmax_in:
                N_unselected = Nmax_in - matrix.shape[1]
                if self.selected_rows == "top":
                    # Matrix uses top rows
                    self.Rp_out_mat[:,-1] = Rp_out * (N_unselected + 1)
                elif self.selected_rows == "bottom":
                    # Matrix uses bottom rows
                    self.Rp_in_mat[:,0] = Rp_in * (N_unselected + 1)

            # Account for terminal resistance
            if Rp_in_terminal > 0 or Rp_out_terminal > 0:
                self.Rp_in_mat[:,0] += Rp_in_terminal
                self.Rp_out_mat[:,-1] += Rp_out_terminal

        # Iteratively calculate parasitics and update device currents
        while Verr > self.Verr_th and Niters < self.Niters_max:
            # Calculate parasitic voltage drops
            Isum_col = xp.cumsum(Ires, 1)
            Isum_supply = xp.cumsum(Ires[:, ::-1], 1)[:, ::-1]
            if self.useMask:
                Isum_col *= self.mask
                Isum_supply *= self.mask

            if len(Isum_col.shape) == 4:
                Vdrops_col = xp.cumsum((self.Rp_out_mat[:,:,None,None]*Isum_col)[:, ::-1], 1)[:, ::-1]
                Vdrops_supply = xp.cumsum(self.Rp_in_mat[:,:,None,None]*Isum_supply, 1)
            else:
                Vdrops_col = xp.cumsum((self.Rp_out_mat*Isum_col)[:, ::-1], 1)[:, ::-1]
                Vdrops_supply = xp.cumsum(self.Rp_in_mat*Isum_supply, 1)
            Vpar = Vdrops_col + Vdrops_supply
            VerrMat = dV0 - Vpar - dV

            Verr = self._error_metric(VerrMat)
            if Verr < self.Verr_th:
                break

            # Update cell currents for the next iteration
            dV += self.gamma * VerrMat
            Ires = matrix * dV
            Niters += 1

        # The current sum has already been calculated
        Icols = Isum_col[:,-1]
        Icols = self._post_process(Icols, input_dim=len(vector.shape), row_in=row_in)
        if Verr > self.Verr_th:
            raise RuntimeError("Parasitic resistance too high: could not converge!")
        return Icols