#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np

# Apply custom device programming errors
#   input_      : input conductance matrix in normalized units (min: 1/On_off_ratio, max:1)
#   error_model : keyword used to select which programming error model is used
#   vcp         : ClipQuantizeConstraints object containing normalized value ranges
#   param_root  : object containing simulation parameters
#   clip_output : whether to clip conductances values to (1/On_off_ratio, 1) after errors
#   mask        : binary mask used when applying errors to matrices with blockwise sparsity, e.g. depthwise convolutions
#
#  When adding a new custom device error model, it is recommended that one of the below cases be copied,
#  then adapted for the new custom device.
#

def applyCustomProgrammingError(input_, error_model, vcp, param_root, clip_output, mask):

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


    if error_model == "SONOS":
        #
        # This programming error model corresponds to the data published for the
        # SONOS charge trapping memory device in
        # T. P. Xiao, et al. "An Accurate, Error-Tolerant, and Energy-Efficient Neural
        # Network Inference Engine Based on SONOS Analog Memory", 
        # IEEE Transactions on Circuits and Systems-I 69 (4), 2022.
        # https://ieeexplore.ieee.org/abstract/document/9669117
        #

        # First, convert xbar normalized conductances to real PCM conductances
        # Max SONOS current
        Imax = 1600 # nA        
        if on_off_ratio == 0:
            Imin = 0
        else:
            Imin = Imax/on_off_ratio
        I = Imin + (Imax-Imin) * (input_ - vcp.minimum) / vcp.range            

        # Next apply a function that converts the programmed current to a corresponding error
        # in the programmed current.
        # Fit coefficients, saturating exponential fit to data in Fig. 4(d)
        A, B = 0.00188704, 39.5656
        sigma_I = ncp.maximum(B - B*ncp.exp(-A*I), 0)

        # Convert back to CrossSim's normalized weight units
        sigma_W = sigma_I / (Imax - Imin)


    elif error_model == "PCM_Joshi":
        #
        # This programming error model corresponds to the data published for
        # phase change memory in:
        # V. Joshi, et al. "Accurate deep neural network inference using computational phase-change memory", 
        # Nature Communications 11, 2473, 2020.
        # https://www.nature.com/articles/s41467-020-16108-9
        #
        # SUGGESTED ON/OFF RATIO : 100
        #

        # First, convert xbar normalized conductances to real PCM conductances
        # Max conductance in PCM model
        Gmax = 25 # uS
        if on_off_ratio == 0:
            Gmin = 0
        else:
            Gmin = Gmax/on_off_ratio
        G = Gmin + (Gmax - Gmin) * (input_ - vcp.minimum) / vcp.range

        # Next determine the conductance programming error at these conductance values
        # The initial time (programming time) is 27.36s after programming
        # Fit coefficients, quadaratic fit to the data in Fig. 3(b)
        A, B, C = -0.00178767, 0.07585724, 0.28638599
        sigma_G = ncp.maximum(A*(G**2) + B*G + C, 0)

        # Convert back to CrossSim normalized weight units
        sigma_W = sigma_G / (Gmax - Gmin)


    elif error_model == "RRAM_Milo":
        #
        # This programming error model corresponds to the data published for
        # resistive random access memory (RRAM) devices in
        # V. Milo, et al. "Optimized programming algorithms for multilevel RRAM in hardware neural networks", 
        # IEEE International Reliability Physics Symposium (IRPS), 2021.
        # https://ieeexplore.ieee.org/document/9405119
        #
        # SUGGESTED ON/OFF RATIO : 5
        #

        # First, convert xbar normalized conductances to real ReRA< conductances
        # Max conductance in ReRAM model
        Gmax = 225 # uS
        if on_off_ratio == 0:
            Gmin = 0
        else:
            Gmin = Gmax/on_off_ratio
        G = Gmin + (Gmax - Gmin) * (input_ - vcp.minimum) / vcp.range

        # Now determine the conductance programming error at these conductance values
        # Fit coefficients: linear fit to the data in Fig. 4 (IGVVA-100 algorithm)
        A, B = -0.009107, 4.782321
        sigma_G = ncp.maximum(A*G + B, 0)
        
        # Convert back to CrossSim normalized weight units
        sigma_W = sigma_G / (Gmax - Gmin)

    else:
        raise ValueError("Custom programming error type not recognized")


    ##########################################################
    ### ----- END DEFINITION OF CUSTOM DEVICE MODELS
    ##########################################################


    #### Apply the programming errors to the weight matrix

    if sigma_W.any():
        if param_root.numeric_params.useGPU:
            randMat = ncp.random.normal(scale=vcp.range, size=input_.shape, dtype=input_.dtype)
        else:
            randMat = ncp.random.normal(scale=vcp.range, size=input_.shape).astype(input_.dtype)
        randMat *= sigma_W
        input_ += randMat
        if clip_output:
            input_ = vcp.clip(input_)
        if mask is not None:
            input_ *= mask
        return input_
    else:
        return input_