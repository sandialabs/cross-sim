#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from abc import ABC, abstractmethod
from simulator.backend import ComputeBackend
from simulator.parameters.core_parameters import CoreStyle
import numpy.typing as npt
import typing

xp = ComputeBackend()


class IArray(ABC):
    """Class that contains compact circuit simulators of different memory
    array electrical topologies, accounting for parasitic metal resistance and
    series resistances.
    """

    def __init__(self, params, device=None) -> None:
        """Creates an uninitialized array object."""
        super().__init__()

        # Normalized parasitic resistances
        self.Rp_row_norm = params.xbar.array.parasitics.Rp_row_norm
        self.Rp_col_norm = params.xbar.array.parasitics.Rp_col_norm
        self.Rp_row_terminal_norm = params.xbar.array.parasitics.Rp_row_terminal_norm
        self.Rp_col_terminal_norm = params.xbar.array.parasitics.Rp_col_terminal_norm

        # Simulation settings
        self.useMask = False
        self.fast_matmul = params.simulation.fast_matmul
        self.Niters_max = params.simulation.Niters_max_parasitics
        self.Verr_th = params.simulation.Verr_th_mvm
        self.hide_convergence_msg = params.simulation.hide_convergence_msg
        self.Verr_matmul_criterion = params.simulation.Verr_matmul_criterion
        self.interleaved = (
            params.core.style != CoreStyle.OFFSET
        ) and params.core.balanced.interleaved_posneg

        # Starting value of gamma. This can change dynamically
        self.gamma = params.simulation.relaxation_gamma

        # Parasitic resistance matrix
        # This will be set on the first call to solve_mvm if needed,
        # with the assumption that the matrix dimensions do not change (if they
        # do, these will have to be reset manually)
        self.Rp_in_mat = None
        self.Rp_out_mat = None

        # Device model (used to apply nonlinear I-V effects)
        self.nonlinear_IV = params.xbar.device.nonlinear_IV.enable
        if self.nonlinear_IV:
            self.device = device
            self.enable_fast_nonlinear_IV = params.simulation.fast_nonlinear_IV

    @abstractmethod
    def solve_mvm(
        self,
        vector: npt.NDArray,
        matrix: npt.NDArray,
        row_in: bool = True,
    ) -> npt.NDArray:
        """Calculates the MVM result including parasitic resistance.

        The solver implementation depends on the specific array topology.

            vector : input vector
            matrix : normalized conductance matrix
            row_in : inputs fed through rows if True, through columns if False.
        """
        raise NotImplementedError

    def iterative_solve(
        self,
        matrix: npt.NDArray,
        vector: npt.NDArray,
        row_in: bool = True,
        matrix_neg: typing.Optional[npt.NDArray] = None,
    ) -> npt.NDArray:
        """Wrapper that is used to implement a convergence loop around the
        circuit solver. Each solver uses successive under-relaxation.

        If the circuit solver fails to find a solution, the relaxation parameter
        will be reduced until the solver converges, or a lower limit on the
        relaxation parameter is reached (returns a ValueError).

        matrix : normalized conductance matrix (for interleaved, this is the
            positive matrix)
        matrix_neg  : conductance matrix for the negative weights, if
            interleaved
        """
        solved, retry = False, False

        while not solved:
            solved = True
            try:
                if not self.interleaved:
                    result = self.solve_mvm(
                        vector,
                        matrix,
                        row_in=row_in,
                    )
                else:
                    result = self.solve_mvm(
                        vector,
                        matrix.copy(),
                        matrix_neg.copy(),
                        row_in=row_in,
                    )

            except RuntimeError as err:
                solved, retry = False, True
                self.gamma *= 0.98
                if self.gamma <= 1e-2:
                    raise ValueError("Parasitic MVM solver failed to converge") from err
        if retry and not self.hide_convergence_msg:
            print(
                "Reduced MVM convergence parameter to: {:.5f}".format(
                    self.gamma,
                ),
            )

        return result

    def _init_dV(
        self,
        input_mat: npt.NDArray,
        matrix: npt.NDArray,
        row_in: bool = True,
    ) -> npt.NDArray:
        """Creates a 4D matrix from the input for use by cumsum function.

        input_mat has three possible shapes:
            matvec 1D - (# inputs)
            matmul 2D - (# inputs, batch size) - dense layer
            matmul 3D - (batch size, # inputs, # MVMs) - convolutional layer
        outputs have the following dimensionality:
            matvec: (# outputs, # inputs)
            matmul: (# outputs, # inputs, # MVMs, batch size).
        """
        input_dim = len(input_mat.shape)
        if input_dim > 1:
            if input_dim == 2:
                if row_in:
                    input_mat = xp.transpose(input_mat, (1, 0))[:, :, xp.newaxis]
                else:
                    input_mat = input_mat[:, :, xp.newaxis]
            if input_dim == 3 and not row_in:
                input_mat = xp.transpose(input_mat, (0, 2, 1))
            input_mat = xp.transpose(input_mat, (1, 2, 0))
            dV0 = xp.tile(input_mat, (matrix.shape[0], 1, 1, 1))
        else:
            # Initial estimate of device voltage and current seen at every
            # element
            dV0 = xp.tile(input_mat, (matrix.shape[0], 1))

        return dV0

    def _cell_currents(
        self,
        Gmat: npt.NDArray,
        Vmat: npt.NDArray,
    ) -> npt.NDArray:
        """Compute the every cell current given the voltage across each cell."""
        if not self.nonlinear_IV:
            if Gmat.ndim == Vmat.ndim:
                return Gmat * Vmat
            else:
                return Gmat[(..., *(None,) * (Vmat.ndim - Gmat.ndim))] * Vmat
        else:
            return self.device.nonlinear_current(Gmat, Vmat)

    def _nonlinear_current_sum(
        self,
        Gmat: npt.NDArray,
        Vterm: npt.NDArray,
        row_in: bool = True,
    ) -> npt.NDArray:
        """Compute the summed currents in an array given an array of terminal
        voltages Vmat.
        This will use the fast nonlinear IV method inside the device model.
        Not valid if parasitic resistance is present.
        """
        if row_in:
            return self.device.nonlinear_current_sum(Gmat, Vterm)
        else:
            return self.device.nonlinear_current_sum(Gmat.T, Vterm)

    def _error_metric(  # noqa:C901
        self,
        VerrMat_pos: npt.NDArray,
        VerrMat_neg: typing.Optional[npt.NDArray] = None,
    ) -> float:
        """Given a matrix of voltage errors (i.e. inconsistency between terminal
        voltages and parasitic voltage drops), compute a single metric that will
        be used to determine circuit solver convergence.
        """
        if VerrMat_neg is None:
            if self.fast_matmul:
                if self.Verr_matmul_criterion == "max_max":
                    # Take max along all four dimensions: rows, columns,
                    # sliding windows, and batch
                    Verr = xp.max(xp.abs(VerrMat_pos))
                elif self.Verr_matmul_criterion == "max_mean":
                    # Take max along rows and columns, then mean across
                    # sliding windows and batch
                    Verr = xp.mean(xp.max(xp.abs(VerrMat_pos), axis=(0, 1)))
                elif self.Verr_matmul_criterion == "max_min":
                    Verr = xp.min(xp.max(xp.abs(VerrMat_pos), axis=(0, 1)))
            else:
                if self.useMask:
                    Verr = xp.max(xp.abs(VerrMat_pos * self.mask))
                else:
                    Verr = xp.max(xp.abs(VerrMat_pos))
        else:
            if self.fast_matmul:
                if self.Verr_matmul_criterion == "max_max":
                    Verr = 0.5 * (
                        xp.max(xp.abs(VerrMat_pos)) + xp.max(xp.abs(VerrMat_neg))
                    )
                elif self.Verr_matmul_criterion == "max_mean":
                    Verr = xp.mean(
                        0.5
                        * (
                            xp.max(xp.abs(VerrMat_pos), axis=(0, 1))
                            + xp.max(xp.abs(VerrMat_neg), axis=(0, 1))
                        )
                    )
                elif self.Verr_matmul_criterion == "max_min":
                    Verr = xp.min(
                        0.5
                        * (
                            xp.max(xp.abs(VerrMat_pos), axis=(0, 1))
                            + xp.max(xp.abs(VerrMat_neg), axis=(0, 1))
                        )
                    )
            else:
                if self.useMask:
                    Verr = 0.5 * (
                        xp.max(xp.abs(VerrMat_pos * self.mask))
                        + xp.max(xp.abs(VerrMat_neg * self.mask))
                    )
                else:
                    Verr = 0.5 * (
                        xp.max(xp.abs(VerrMat_pos)) + xp.max(xp.abs(VerrMat_neg))
                    )

        return Verr

    def _post_process(
        self,
        Icols: npt.NDArray,
        input_dim: int = 3,
        row_in: bool = True,
    ) -> npt.NDArray:
        """Reshapes parasitic simulation matrix and check for NaNs."""
        # Undo permute and axis switching for FC layer
        if input_dim > 1:
            Icols = xp.transpose(Icols, (2, 0, 1))
            if input_dim == 2:
                Icols = Icols[:, :, 0]
                if row_in:
                    Icols = xp.transpose(Icols, (1, 0))
            elif input_dim == 3 and not row_in:
                Icols = xp.transpose(Icols, (0, 2, 1))
        if xp.isnan(Icols).any():
            raise RuntimeError("Nans due to parasitic resistance simulation")
        return Icols

    def _create_parasitics_mask(
        self,
        matrix: npt.NDArray,
        Ncopy: int,
    ) -> npt.NDArray:
        """Create a mask for parasitic simulations with sliding window packing.

        matrix: the array whose shape and dtype will be used to construct the
            parasitics mask.
        """
        if Ncopy > 1:
            Nx, Ny = matrix.shape
            self.mask = xp.zeros((Ncopy * Nx, Ncopy * Ny), dtype=matrix.dtype)
            for m in range(Ncopy):
                x_start, x_end = m * Nx, (m + 1) * Nx
                y_start, y_end = m * Ny, (m + 1) * Ny
                self.mask[x_start:x_end, y_start:y_end] = 1
            self.mask = self.mask > 1e-9
            self.useMask = True
