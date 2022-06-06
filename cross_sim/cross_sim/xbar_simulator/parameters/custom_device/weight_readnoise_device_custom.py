#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
import numpy as np

# Apply custom device programming errors
#   input_      : input conductance matrix in normalized units (min: 1/On_off_ratio, max:1)
#   noise_model : keyword used to select which read noise model is used
#   vcp         : ClipQuantizeConstraints object containing normalized value ranges
#   param_root  : object containing simulation parameters
#   mask        : binary mask used when applying errors to matrices with blockwise sparsity, e.g. depthwise convolutions

def setDeviceReadNoise(input_, noise_model, vcp, param_root, mask):

    if param_root.xbar_params.weights.minimum > 0:
        on_off_ratio = 1/param_root.xbar_params.weights.minimum
    else:
        on_off_ratio = 1e20

    if param_root.numeric_params.useGPU:
        import cupy as cp
        cp.cuda.Device(param_root.numeric_params.gpu_id).use()
        ncp = cp
    else:
        ncp = np


    ##########################################################
    ### ----- START DEFINITION OF CUSTOM DEVICE MODELS
    ##########################################################


    if noise_model == "parabolic":

        # This is a fully hypothetical nonlinear device read noise model
        # First convert normalized conductances to actual conductances in the range 0 to 1000 nS
        # Then, the noise stdev is computed as a parabolic function of the conductance
        # The noise is not applied on the weights in the call to this function, but is applied subsequently on
        # every MVM during the simulation using the stdev values found here
        
        Gmax = 1000 # nS        
        if on_off_ratio == 0:
            Gmin = 0
        else:
            Gmin = Gmax/on_off_ratio
        G = Gmin + (Gmax-Gmin) * (input_ - vcp.minimum) / vcp.range            

        # Polynomial parameters
        A, B, C = np.array([2e-4,2e-2,10])
        G_std = ncp.maximum(A*pow(G,2) + B*G + C,0)

        # Convert real conductance noise back into CrossSim's normalized weight units (no need to modify)
        std_matrix = G_std * vcp.range / (Gmax - Gmin)


    ##########################################################
    ### ----- END DEFINITION OF CUSTOM DEVICE MODELS
    ##########################################################


    else:
        raise ValueError("Custom read noise model not recognized")

    if mask is not None:
        std_matrix *= mask

    return std_matrix
