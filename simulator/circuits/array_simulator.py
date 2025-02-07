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
    simulation_params,
    xbar_params,
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
    vector      : input vector
    matrix      : normalized conductance matrix (for interleaved, this is the positive matrix)
    simulation_params : simulation parameter object
    xbar_params : xbar parameter object for the core that calls this function
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
                    simulation_params,
                    xbar_params,
                    useMask=useMask,
                    mask=mask,
                    row_in=row_in,
                )
            else:
                result = circuit_solver(
                    vector,
                    matrix.copy(),
                    matrix_neg.copy(),
                    simulation_params,
                    xbar_params,
                    useMask=useMask,
                    mask=mask,
                    row_in=row_in,
                )

        except RuntimeError:
            solved, retry = False, True
            simulation_params.relaxation_gamma *= 0.98
            if simulation_params.relaxation_gamma <= 1e-2:
                raise ValueError("Parasitic MVM solver failed to converge")
    if retry and not simulation_params.hide_convergence_msg:
        print(
            "Reduced MVM convergence parameter to: {:.5f}".format(
                simulation_params.relaxation_gamma,
            ),
        )

    return result


def _init_dV(input_mat, matrix, fast_matmul = False):
    """
    Takes an input matrix and creates a 4D matrix that can be passed to the cumsum
    function.
    input_mat has two possible shapes:
        matvec mode:
            1D - (# inputs)
        matmul mode:
            2D - (# inputs, batch size) - from dense layer
            3D - (batch size, # inputs, # MVMs) - from convolutional layer
    outputs have the following dimensionality:
        matvec: (# outputs, # inputs)
        matmul: (# outputs, # inputs, # MVMs, batch size)
    """
    if fast_matmul:
        if len(input_mat.shape) == 2:
            input_mat = xp.transpose(input_mat,(1,0))[:,:,xp.newaxis]
        input_mat = xp.transpose(input_mat, (1,2,0))
        dV0 = xp.tile(input_mat, (matrix.shape[0], 1, 1, 1))
    else:
        # Initial estimate of device voltage and current seen at every element
        dV0 = xp.tile(input_mat, (matrix.shape[0], 1))

    return dV0


def _error_metric(
    VerrMat_pos,
    VerrMat_neg=None,
    mask=None,
    useMask=False,
    fast_matmul=False,
    Verr_matmul_criterion="max_max"):
    """
    Given a matrix of voltage errors (i.e. inconsistency between terminal voltages and parasitic
    voltage drops), compute a single metric that will be used to determine circuit solver
    convergence.
    """
    if VerrMat_neg is None:
        if fast_matmul:
            if Verr_matmul_criterion == "max_max":
                # Take max along all four dimensions: rows, columns, sliding windows, and batch
                Verr = xp.max(xp.abs(VerrMat_pos))
            elif Verr_matmul_criterion == "max_mean":
                # Take max along rows and columns, then mean across sliding windows and batch
                Verr = xp.mean(xp.max(xp.abs(VerrMat_pos),axis=(0,1)))
            elif Verr_matmul_criterion == "max_min":
                Verr = xp.min(xp.max(xp.abs(VerrMat_pos),axis=(0,1)))
        else:
            if useMask:
                Verr = xp.max(xp.abs(VerrMat_pos*mask))
            else:
                Verr = xp.max(xp.abs(VerrMat_pos))
    else:
        if fast_matmul:
            if Verr_matmul_criterion == "max_max":
                Verr = 0.5 * (xp.max(xp.abs(VerrMat_pos)) + \
                    xp.max(xp.abs(VerrMat_neg)))
            elif Verr_matmul_criterion == "max_mean":
                Verr = xp.mean(0.5 * (xp.max(xp.abs(VerrMat_pos),axis=(0,1)) + \
                    xp.max(xp.abs(VerrMat_neg),axis=(0,1))))
            elif Verr_matmul_criterion == "max_min":
                Verr = xp.min(0.5 * (xp.max(xp.abs(VerrMat_pos),axis=(0,1)) + \
                    xp.max(xp.abs(VerrMat_neg),axis=(0,1))))
        else:
            if useMask:
                Verr = 0.5 * (xp.max(xp.abs(VerrMat_pos*mask)) + xp.max(xp.abs(VerrMat_neg*mask)))
            else:
                Verr = 0.5 * (xp.max(xp.abs(VerrMat_pos)) + xp.max(xp.abs(VerrMat_neg)))

    return Verr


def _post_process(Icols, fast_matmul=False, input_dim=3):
    """Takes the output of parasitic simulation and reshapes it if needed, and checks for Nans.
    """
    # Undo permute and axis switching for FC layer
    if fast_matmul:
        Icols = xp.transpose(Icols,(2,0,1))
        if input_dim == 2:
            Icols = xp.transpose(Icols[:,:,0],(1,0))
    if xp.isnan(Icols).any():
        raise RuntimeError("Nans due to parasitic resistance simulation")
    return Icols


def mvm_parasitics(
    vector,
    matrix,
    simulation_params,
    xbar_params,
    useMask=False,
    mask=None,
    row_in=True):
    """Calculates the MVM result including parasitic resistance, for a non-interleaved array.
    """
    if xbar_params.array.parasitics.gate_input:
        return mvm_parasitics_gateInput(vector, matrix, simulation_params, xbar_params, 
            useMask=useMask, mask=mask, row_in=row_in)

    if row_in:
        Rp_in = xbar_params.array.parasitics.Rp_row_norm
        Rp_out = xbar_params.array.parasitics.Rp_col_norm
    else:
        Rp_in = xbar_params.array.parasitics.Rp_col_norm
        Rp_out = xbar_params.array.parasitics.Rp_row_norm

    Niters_max = simulation_params.Niters_max_parasitics
    Verr_th = simulation_params.Verr_th_mvm
    gamma = simulation_params.relaxation_gamma
    fast_matmul = simulation_params.fast_matmul

    # Initialize error and number of iterations
    Verr = 1e9
    Niters = 0

    # Compute element-wise voltage drops and currents
    dV0 = _init_dV(vector, matrix, fast_matmul=fast_matmul)
    if fast_matmul:
        Ires = matrix[:,:,None,None] * dV0
    else:
        Ires = matrix * dV0
    dV = dV0.copy()

    # Iteratively calculate parasitics and update device currents
    while Verr > Verr_th and Niters < Niters_max:
        # Calculate parasitic voltage drops
        Isum_col = xp.cumsum(Ires, 1)
        Isum_row = xp.cumsum(Ires[::-1], 0)[::-1]
        if useMask:
            Isum_col *= mask
            Isum_row *= mask

        Vdrops_col = Rp_out * xp.cumsum(Isum_col[:, ::-1], 1)[:, ::-1]
        Vdrops_row = Rp_in * xp.cumsum(Isum_row, 0)
        Vpar = Vdrops_col + Vdrops_row

        # Calculate the error for the current estimate of memristor currents
        VerrMat = dV0 - Vpar - dV

        # Evaluate overall error
        Verr = _error_metric(
            VerrMat,
            mask=mask,
            useMask=useMask,
            fast_matmul=fast_matmul,
            Verr_matmul_criterion=simulation_params.Verr_matmul_criterion)

        if Verr < Verr_th:
            break

        # Update memristor currents for the next iteration
        dV += gamma * VerrMat
        if fast_matmul:
            Ires = matrix[:,:,None,None] * dV
        else:
            Ires = matrix * dV
        Niters += 1

    # Calculate the summed currents on the columns
    Icols = xp.sum(Ires, axis=1)
    Icols = _post_process(Icols, fast_matmul=fast_matmul, input_dim=len(vector.shape))

    # Should add some more checks here on whether the results of this calculation are erroneous even if it converged
    if Verr > Verr_th:
        raise RuntimeError("Parasitic resistance too high: could not converge!")
    return Icols


def mvm_parasitics_gateInput(
    vector,
    matrix,
    simulation_params,
    xbar_params,
    useMask=False,
    mask=None,
    row_in=True,
):
    """Calculates the MVM result including parasitic resistance, for a non-interleaved array.

    Assumes an array topology where input is applied bitwise on the gate, and there is no parasitic
    resistance along the input dimension (gate_input = True, input_bitslicing = True)

    Input arguments are the same as mvm_parasitics
    """
    if row_in:
        Rp_out = xbar_params.array.parasitics.Rp_col_norm
    else:
        Rp_out = xbar_params.array.parasitics.Rp_row_norm
    Niters_max = simulation_params.Niters_max_parasitics
    Verr_th = simulation_params.Verr_th_mvm
    gamma = simulation_params.relaxation_gamma
    fast_matmul = simulation_params.fast_matmul

    # Initialize error and number of iterations
    Verr = 1e9
    Niters = 0

    dV0 = _init_dV(vector, matrix, fast_matmul=fast_matmul)

    # If input is zero, device is gated off and has effectively zero conductance
    active_inputs = xp.abs(dV0) > 1e-9
    if fast_matmul:
        matrix = matrix[:,:,None,None] * active_inputs
    else:
        matrix *= active_inputs
    Ires = matrix * dV0
    dV = dV0.copy()

    # Iteratively calculate parasitics and update device currents
    while Verr > Verr_th and Niters < Niters_max:
        # # Calculate parasitic voltage drops
        Isum_col = xp.cumsum(Ires, 1)
        if useMask:
            Isum_col *= mask
        Vpar = Rp_out * xp.cumsum(Isum_col[:, ::-1], 1)[:, ::-1]

        # Calculate the error for the current estimate of memristor currents
        # If using SIMD, make sure only to count the cells that matter
        VerrMat = dV0 - Vpar - dV
        Verr = _error_metric(
            VerrMat,
            mask=mask,
            useMask=useMask,
            fast_matmul=fast_matmul,
            Verr_matmul_criterion=simulation_params.Verr_matmul_criterion)

        if Verr < Verr_th:
            break

        # Update memristor currents for the next iteration
        dV += gamma * VerrMat
        Ires = matrix * dV
        Niters += 1

    # Calculate the summed currents on the columns
    Icols = xp.sum(Ires, axis=1)
    Icols = _post_process(Icols, fast_matmul=fast_matmul, input_dim=len(vector.shape))
    if Verr > Verr_th:
        raise RuntimeError("Parasitic resistance too high: could not converge!")
    return Icols


def mvm_parasitics_interleaved(
    vector,
    matrix_pos,
    matrix_neg,
    simulation_params,
    xbar_params,
    useMask=False,
    mask=None,
    row_in=True,
):
    """Calculates the MVM result including parasitic resistance, for an array where the positive
    and negative resistors in a differential pair are connected to the same column. On the other
    side, the positive resistor is connected to +VDD, the negative resistor is connected to -VDD.

    Parasitic resistance is inserted between every pair, and not between the two devices in the pair

    Input arguments are the same as mvm_parasitics, except:
        matrix_pos  : normalized conductance matrix for positive conductances in diff. pair
        matrix_neg  : normalized conductance matrix for negative conductances in diff. pair
    """
    if xbar_params.array.parasitics.gate_input:
        return mvm_parasitics_interleaved_gateInput(vector, 
            matrix_pos, matrix_neg, simulation_params, xbar_params, 
            useMask=useMask, mask=mask, row_in=row_in)

    if row_in:
        Rp_in = xbar_params.array.parasitics.Rp_row_norm
        Rp_out = xbar_params.array.parasitics.Rp_col_norm
    else:
        Rp_in = xbar_params.array.parasitics.Rp_col_norm
        Rp_out = xbar_params.array.parasitics.Rp_row_norm
    Niters_max = simulation_params.Niters_max_parasitics
    Verr_th = simulation_params.Verr_th_mvm
    gamma = simulation_params.relaxation_gamma
    fast_matmul = simulation_params.fast_matmul

    # Initialize error and number of iterations
    Verr = 1e9
    Niters = 0

    # Compute element-wise voltage drops and currents
    dV0_pos = _init_dV(vector, matrix_pos, fast_matmul=fast_matmul)

    if fast_matmul:
        Ires_pos = matrix_pos[:,:,None,None] * dV0_pos
        Ires_neg = -matrix_neg[:,:,None,None] * dV0_pos
    else:
        Ires_pos = matrix_pos * dV0_pos
        Ires_neg = -matrix_neg * dV0_pos

    # Compute interleaved currents
    Ires = Ires_pos + Ires_neg

    # Initial estimate of device currents
    dV_pos = dV0_pos.copy()
    dV_neg = -dV0_pos.copy()

    # Iteratively calculate parasitics and update device currents
    while Verr > Verr_th and Niters < Niters_max:
        # Calculate parasitic voltage drops
        # Use the same variable names to reduce memory usage
        Ires = xp.cumsum(Ires, 1)
        Ires_pos = xp.cumsum(Ires_pos[::-1], 0)[::-1]
        Ires_neg = xp.cumsum(Ires_neg[::-1], 0)[::-1]
        if useMask:
            Ires *= mask
            Ires_pos *= mask
            Ires_neg *= mask

        Vdrops_col = Rp_out * xp.cumsum(Ires[:, ::-1], 1)[:, ::-1]
        Vpar_pos = Vdrops_col + Rp_in * xp.cumsum(Ires_pos, 0)
        Vpar_neg = Vdrops_col + Rp_in * xp.cumsum(Ires_neg, 0)
        VerrMat_pos = dV0_pos - Vpar_pos - dV_pos
        VerrMat_neg = -dV0_pos - Vpar_neg - dV_neg

        Verr = _error_metric(
            VerrMat_pos, 
            VerrMat_neg=VerrMat_neg,
            useMask=useMask,
            mask=mask,
            fast_matmul=fast_matmul,
            Verr_matmul_criterion=simulation_params.Verr_matmul_criterion)

        if Verr < Verr_th:
            break

        # Update cell currents for the next iteration
        dV_pos += gamma * VerrMat_pos
        dV_neg += gamma * VerrMat_neg
        if fast_matmul:
            Ires_pos = matrix_pos[:,:,None,None] * dV_pos
            Ires_neg = matrix_neg[:,:,None,None] * dV_neg
        else:
            Ires_pos = matrix_pos * dV_pos
            Ires_neg = matrix_neg * dV_neg
        Ires = Ires_pos + Ires_neg
        Niters += 1

    # The current sum has already been calculated
    Icols = Ires[:,-1,:,:]
    Icols = _post_process(Icols, fast_matmul=fast_matmul, input_dim=len(vector.shape))
    if Verr > Verr_th:
        raise RuntimeError("Parasitic resistance too high: could not converge!")
    return Icols


def mvm_parasitics_interleaved_gateInput(
    vector,
    matrix_pos,
    matrix_neg,
    simulation_params,
    xbar_params,
    useMask=False,
    mask=None,
    row_in=True,
):
    """Calculates the interleaved MVM where input is applied biwise on the gate and there is
    no parasitic resistance along a row (gate_input = True, input_bitslicing = True)

        Input arguments are the same as mvm_parasitics_interleaved
    """
    if row_in:
        Rp_out = xbar_params.array.parasitics.Rp_col_norm
    else:
        Rp_out = xbar_params.array.parasitics.Rp_row_norm
    Niters_max = simulation_params.Niters_max_parasitics
    Verr_th = simulation_params.Verr_th_mvm
    gamma = simulation_params.relaxation_gamma
    fast_matmul = simulation_params.fast_matmul

    # Initialize error and number of iterations
    Verr = 1e9
    Niters = 0

    # Initial voltage drops
    dV0 = _init_dV(vector, matrix_pos, fast_matmul=fast_matmul)

    # If input is zero, device is gated off and has effectively zero conductance
    active_inputs = xp.abs(dV0) > 1e-9
    if fast_matmul:
        matrix_pos = matrix_pos[:,:,None,None] * active_inputs
        matrix_neg = matrix_neg[:,:,None,None] * active_inputs
    else:
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
        Isum_col = xp.cumsum(Ires, 1)
        if useMask:
            Isum_col *= xp.cumsum(Ires, 1)

        Vpar = Rp_out * xp.cumsum(Isum_col[:, ::-1], 1)[:, ::-1]

        # Calculate the error for the current estimate of memristor currents
        VerrMat_pos = dV0 - Vpar - dV_pos
        Verr = _error_metric(
            VerrMat_pos,
            mask=mask,
            useMask=useMask,
            fast_matmul=fast_matmul,
            Verr_matmul_criterion=simulation_params.Verr_matmul_criterion)

        if Verr < Verr_th:
            break

        # Update cell currents for the next iteration
        dV_pos += gamma * VerrMat_pos
        Ires = matrix_sum * dV_pos - I_int
        Niters += 1

    # Calculate the summed currents on the columns
    Icols = xp.sum(Ires, axis=1)
    Icols = _post_process(Icols, fast_matmul=fast_matmul, input_dim=len(vector.shape))
    if Verr > Verr_th:
        raise RuntimeError("Parasitic resistance too high: could not converge!")
    return Icols
