#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from simulator.backend import ComputeBackend

xp = ComputeBackend()  # Represents either cupy or numpy

##############################

# This file implements Python compact circuit solvers to simulate the effects
# of undesired resistances in the array
# To optimize performance, solvers for different circuit topologies are implemented
# in different functions

##############################


def solve_mvm_circuit(
    circuit_solver,
    vector,
    matrix,
    params,
    interleaved=False,
    matrix_neg=None,
    useMask=False,
    mask=None,
    row_in=True,
):
    """Wrapper that is used to implement a convergence loop around the circuit solver.

    Each solver uses successive under-relaxation.

    If the circuit solver fails to find a solution, the relaxation parameter will
    be reduced until the solver converges, or a lower limit on the relaxation parameter
    is reached (returns a ValueError)

    circuit_solver : function handle to the desired circuit solver
    vector : input vector
    matrix : normalized conductance matrix (for interleaved, this is the positive matrix)
        params : simulation parameter object
        interleaved : whether positive and negative resistors are interleaved in the array
        matrix_neg  : conductance matrix for the negative weights, if interleaved
        useMask     : whether to use a mask to improve performance with SW packing
        mask        : mask to use with SW packing, if useMask = True
    row_in      : inputs passed in through rows, False if inputs passed in through columns
                    In numeric_core, row_in = True if MVM, row_in = False if VMM
    """
    solved, retry = False, False

    while not solved:
        solved = True
        try:
            if not interleaved:
                result = circuit_solver(
                    vector,
                    matrix,
                    params,
                    useMask=useMask,
                    mask=mask,
                    row_in=row_in,
                )
            else:
                result = circuit_solver(
                    vector,
                    matrix.copy(),
                    matrix_neg.copy(),
                    params,
                    useMask=useMask,
                    mask=mask,
                    row_in=row_in,
                )

        except RuntimeError:
            solved, retry = False, True
            params.simulation.relaxation_gamma *= 0.98
            if params.simulation.relaxation_gamma <= 1e-2:
                raise ValueError("Parasitic MVM solver failed to converge")
    if retry and not params.simulation.hide_convergence_msg:
        print(
            "Reduced MVM convergence parameter to: {:.5f}".format(
                params.simulation.relaxation_gamma,
            ),
        )

    return result


def mvm_parasitics(vector, matrix, params, useMask=False, mask=None, row_in=True):
    """Calculates the MVM result including parasitic resistance, for a non-interleaved array.

    vector : input vector
    matrix : normalized conductance matrix
        params : simulation parameter object
        useMask: whether to use a mask to improve performance with SW packing
        mask   : mask to use with SW packing, if useMask = True
    row_in : inputs fed through rows if True, through columns if False
    """
    if params.xbar.array.parasitics.gate_input and params.xbar.dac.mvm.input_bitslicing:
        return mvm_parasitics_gateInput(vector, matrix, params, useMask, mask=mask)

    # Parasitic resistance
    if row_in:
        Rp_in = params.xbar.array.parasitics.Rp_row
        Rp_out = params.xbar.array.parasitics.Rp_col
    else:
        Rp_in = params.xbar.array.parasitics.Rp_row
        Rp_out = params.xbar.array.parasitics.Rp_col

    Niters_max = params.simulation.Niters_max_parasitics
    Verr_th = params.simulation.Verr_th_mvm
    gamma = params.simulation.relaxation_gamma

    # Initialize error and number of iterations
    Verr = 1e9
    Niters = 0

    # Initial estimate of device currents
    # Input seen at every element
    dV0 = xp.tile(vector, (matrix.shape[0], 1))
    Ires = matrix * dV0
    dV = dV0.copy()

    # Iteratively calculate parasitics and update device currents
    while Verr > Verr_th and Niters < Niters_max:
        # Calculate parasitic voltage drops
        if useMask:
            Isum_col = mask * xp.cumsum(Ires, 1)
            Isum_row = mask * xp.cumsum(Ires[::-1], 0)[::-1]
        else:
            Isum_col = xp.cumsum(Ires, 1)
            Isum_row = xp.cumsum(Ires[::-1], 0)[::-1]

        Vdrops_col = Rp_out * xp.cumsum(Isum_col[:, ::-1], 1)[:, ::-1]
        Vdrops_row = Rp_in * xp.cumsum(Isum_row, 0)
        Vpar = Vdrops_col + Vdrops_row

        # Calculate the error for the current estimate of memristor currents
        VerrMat = dV0 - Vpar - dV

        # Evaluate overall error; if using SIMD, make sure only to count the cells that matter
        if useMask:
            Verr = xp.max(xp.abs(VerrMat[mask]))
        else:
            Verr = xp.max(xp.abs(VerrMat))
        if Verr < Verr_th:
            break

        # Update memristor currents for the next iteration
        dV += gamma * VerrMat
        Ires = matrix * dV
        Niters += 1

    # Calculate the summed currents on the columns
    Icols = xp.sum(Ires, axis=1)

    # Should add some more checks here on whether the results of this calculation are erroneous even if it converged
    if Verr > Verr_th:
        raise RuntimeError("Parasitic resistance too high: could not converge!")
    if xp.isnan(Icols).any():
        raise RuntimeError("Nans due to parasitic resistance simulation")
    return Icols


def mvm_parasitics_gateInput(
    vector,
    matrix,
    params,
    useMask=False,
    mask=None,
    row_in=True,
):
    """Calculates the MVM result including parasitic resistance, for a non-interleaved array.

    Assumes an array topology where input is applied bitwise on the gate, and there is no parasitic
    resistance along the input dimension (gate_input = True, input_bitslicing = True)

    Input arguments are the same as mvm_parasitics
    """
    # Parasitic resistance
    if row_in:
        Rp_out = params.xbar.array.parasitics.Rp_col
    else:
        Rp_out = params.xbar.array.parasitics.Rp_row
    Niters_max = params.simulation.Niters_max_parasitics
    Verr_th = params.simulation.Verr_th_mvm
    gamma = params.simulation.relaxation_gamma

    # Initialize error and number of iterations
    Verr = 1e9
    Niters = 0

    # Initial estimate of device currents
    dV0 = xp.tile(vector, (matrix.shape[0], 1))

    # FOR A THREE-TERMINAL DEVICE:
    # If input is zero, the device is gated off and cannot conduct current regardless of the drain voltage
    # Ensure this by zeroing out the conductances where the row is not activated
    active_inputs = xp.abs(dV0) > 1e-9
    matrix *= active_inputs
    Ires = matrix * dV0
    dV = dV0.copy()

    # Iteratively calculate parasitics and update device currents
    while Verr > Verr_th and Niters < Niters_max:
        # # Calculate parasitic voltage drops
        Isum_col = xp.cumsum(Ires, 1)
        if useMask:
            Isum_col *= mask

        Vpar = xp.flip(Isum_col, 1)
        xp.cumsum(Vpar, 1, out=Vpar)
        Vpar = xp.flip(Vpar, 1)
        Vpar *= Rp_out

        # Calculate the error for the current estimate of memristor currents
        # If using SIMD, make sure only to count the cells that matter
        VerrMat = dV0 - Vpar - dV
        if useMask:
            Verr = xp.max(xp.abs(VerrMat[mask]))
        else:
            Verr = xp.max(xp.abs(VerrMat))

        if Verr < Verr_th:
            break

        # Update memristor currents for the next iteration
        # Under-relaxation
        dV += gamma * VerrMat

        # Over-relaxation
        # dV = (1-gamma)*dV + gamma*(dV0 - Vpar)

        Ires = matrix * dV
        Niters += 1

    # Calculate the summed currents on the columns
    Icols = xp.sum(Ires, axis=1)

    if Verr > Verr_th:
        raise RuntimeError("Parasitic resistance too high: could not converge!")
    if xp.isnan(Icols).any():
        raise RuntimeError("Nans due to parasitic resistance simulation")
    return Icols


def mvm_parasitics_interleaved(
    vector,
    matrix_pos,
    matrix_neg,
    params,
    useMask=False,
    mask=None,
    row_in=True,
):
    """Calculates the MVM result including parasitic resistance, for an array where the positive
    and negative resistors in a differential pair are connected to the same column. On the other
    side, the positive resistor is connected to +VDD, the negative resistor is connected to -VDD.

    Parasitic resistance is inserted between every pair, and not between the two devices in the pair

    Currently this is compatible only with a topology where the input is applied bitwise on the gate,
    and there is no parasitic resistance along a row (gate_input = True, input_bitslicing = True)

        Input arguments are the same as mvm_parasitics, except:
    matrix_pos 	: normalized conductance matrix for positive conductances in diff. pair
    matrix_neg 	: normalized conductance matrix for negative conductances in diff. pair
    """
    if not params.xbar.array.parasitics.gate_input:
        raise ValueError("Currently, interleaved parasitics option requires gate input")

    # Parasitic resistance
    if row_in:
        Rp_out = params.xbar.array.parasitics.Rp_col
    else:
        Rp_out = params.xbar.array.parasitics.Rp_row
    Niters_max = params.simulation.Niters_max_parasitics
    Verr_th = params.simulation.Verr_th_mvm
    gamma = params.simulation.relaxation_gamma

    # Initialize error and number of iterations
    Verr = 1e9
    Niters = 0

    # Initial voltage drops across the positive and negative devices
    dV0 = xp.tile(vector, (matrix_pos.shape[0], 1))

    # If input is zero, the device is gated off and cannot conduct current regardless of the drain voltage
    # Ensure this by zeroing out the conductances where the row is not activated
    active_inputs = xp.abs(dV0) > 1e-9
    matrix_pos *= active_inputs
    matrix_neg *= active_inputs
    Ires = (matrix_pos - matrix_neg) * dV0

    # Pre-compute some intermediate values
    matrix_sum = matrix_pos + matrix_neg
    I_int = matrix_neg * dV0 * 2

    # Initial estimate of device currents
    # Initial dV is the same for positive and negative other than the sign
    dV_pos = dV0.copy()

    # Iteratively calculate parasitics and update device currents
    while Verr > Verr_th and Niters < Niters_max:
        # Calculate parasitic voltage drops
        if useMask:
            Isum_col = mask * xp.cumsum(Ires, 1)
        else:
            Isum_col = xp.cumsum(Ires, 1)

        Vpar = Rp_out * xp.cumsum(Isum_col[:, ::-1], 1)[:, ::-1]

        # Calculate the error for the current estimate of memristor currents
        VerrMat_pos = dV0 - Vpar - dV_pos

        # Evaluate overall error; if using SIMD, make sure only to count the cells that matter
        if useMask:
            Verr = xp.max(xp.abs(VerrMat_pos[mask]))
        else:
            Verr = xp.max(xp.abs(VerrMat_pos))

        if Verr < Verr_th:
            break

        # Update cell currents for the next iteration
        dV_pos += gamma * VerrMat_pos
        Ires = matrix_sum * dV_pos - I_int

        # Underlying math for the Ires line, expanded:
        # dV_neg = 2*dV0 - dV_pos
        # Ires = matrix_pos*dV_pos - matrix_neg*dV_neg

        Niters += 1

    # Calculate the summed currents on the columns
    Icols = xp.sum(Ires, axis=1)

    # Should add some more checks here on whether the results of this calculation are erroneous even if it converged
    if Verr > Verr_th:
        raise RuntimeError("Parasitic resistance too high: could not converge!")
    if xp.isnan(Icols).any():
        raise RuntimeError("Nans due to parasitic resistance simulation")
    return Icols
