#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from .iarray import IArray
from simulator.backend import ComputeBackend

xp = ComputeBackend()

class Interleaved_SeparateSource_Array(IArray):
    """
    This class models an array where:
    1) The current through the memory cells is sourced from a separate interconnect
        than the input signal, and this interconnect is parallel to the output
        summation column. The input signal is used to connect/disconnect the summation
        column to the current source column. 
    2) Memory cells for positive and negative weights in a differential pair are 
        interleaved, connecting to the same summation column. On the other side, the
        positive cell is connected to a positive voltage and the negative cell is
        connected to a negative voltage w.r.t the summation column. This allows current
        to cancel locally along the summation column.

    This array is only used for a BALANCED style core.
    """

    def __init__(
        self,
        params,
    ) -> None:
        super().__init__(params)
        self.interleaved = True
        self.selected_rows = params.xbar.array.parasitics.selected_rows
        self.rows_max = params.core.rows_max
        self.cols_max = params.core.cols_max


    def solve_mvm_parasitics(self, vector, matrix_pos, matrix_neg, row_in=True):
        """
        matrix_pos  : normalized conductance matrix for positive conductances in diff. pair
        matrix_neg  : normalized conductance matrix for negative conductances in diff. pair
        """
        # Initialize error and number of iterations
        Verr = 1e9
        Niters = 0

        # Compute element-wise voltage drops and currents
        dV0_pos = self._init_dV(vector, matrix_pos, row_in=row_in)

        # If input is zero, device is gated off and has effectively zero conductance
        active_inputs = xp.abs(dV0_pos) > 1e-9
        if len(dV0_pos.shape) == 4 and len(matrix_pos.shape) != 4:
            matrix_pos = matrix_pos[:,:,None,None] * active_inputs
            matrix_neg = matrix_neg[:,:,None,None] * active_inputs
        else:
            matrix_pos *= active_inputs
            matrix_neg *= active_inputs

        Ires_pos = matrix_pos * dV0_pos
        Ires_neg = -matrix_neg * dV0_pos

        # Compute interleaved currents
        Ires = Ires_pos + Ires_neg

        # Initial estimate of device currents
        dV_pos = dV0_pos.copy()
        dV_neg = -dV0_pos.copy()

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

            # Parasitic resistance matrices
            self.Rp_in_mat = Rp_in * xp.ones((matrix_pos.shape[0], matrix_pos.shape[1]))
            self.Rp_out_mat = Rp_out * xp.ones((matrix_pos.shape[0], matrix_pos.shape[1]))

            # Account for a weight matrix smaller than the array size
            if matrix_pos.shape[1] < Nmax_in:
                N_unselected = Nmax_in - matrix_pos.shape[1]
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
            Isum_pos = xp.cumsum(Ires_pos[:, ::-1], 1)[:, ::-1]
            Isum_neg = xp.cumsum(Ires_neg[:, ::-1], 1)[:, ::-1]
            if self.useMask:
                Isum_col *= self.mask
                Isum_pos *= self.mask
                Isum_neg *= self.mask

            if len(Isum_col.shape) == 4:
                Vdrops_col = xp.cumsum((self.Rp_out_mat[:,:,None,None]*Isum_col)[:, ::-1], 1)[:, ::-1]
                Vpar_pos = Vdrops_col + xp.cumsum(self.Rp_in_mat[:,:,None,None]*Isum_pos, 1)
                Vpar_neg = Vdrops_col + xp.cumsum(self.Rp_in_mat[:,:,None,None]*Isum_neg, 1)
            else:
                Vdrops_col = xp.cumsum((self.Rp_out_mat*Isum_col)[:, ::-1], 1)[:, ::-1]
                Vpar_pos = Vdrops_col + xp.cumsum(self.Rp_in_mat*Isum_pos, 1)
                Vpar_neg = Vdrops_col + xp.cumsum(self.Rp_in_mat*Isum_neg, 1)
            VerrMat_pos = dV0_pos - Vpar_pos - dV_pos
            VerrMat_neg = -dV0_pos - Vpar_neg - dV_neg

            Verr = self._error_metric(VerrMat_pos, VerrMat_neg=VerrMat_neg)

            if Verr < self.Verr_th:
                break

            # Update cell currents for the next iteration
            dV_pos += self.gamma * VerrMat_pos
            dV_neg += self.gamma * VerrMat_neg
            Ires_pos = matrix_pos * dV_pos
            Ires_neg = matrix_neg * dV_neg
            Ires = Ires_pos + Ires_neg
            Niters += 1

        # The current sum has already been calculated
        Icols = Isum_col[:,-1]
        Icols = self._post_process(Icols, input_dim=len(vector.shape), row_in=row_in)
        if Verr > self.Verr_th:
            raise RuntimeError("Parasitic resistance too high: could not converge!")
        return Icols
