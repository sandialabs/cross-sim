#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np
from scipy.interpolate import interp1d, interp2d

# Apply custom conductance drift model
#   input_      : input conductance matrix in normalized units (min: 1/On_off_ratio, max:1)
#   T           : time after programming (days)
#   drift_model : keyword used to select which state drift model is used
#   vcp         : ClipQuantizeConstraints object containing normalized value ranges
#   param_root  : object containing simulation parameters
#   clip_output : whether to clip conductances values to (1/On_off_ratio, 1) after errors
#   mask        : binary mask used when applying errors to matrices with blockwise sparsity, e.g. depthwise convolutions
#
#  When adding a new custom device error model, it is recommended that one of the below cases be copied, as applicable,
#  then adapted for the new custom device.
#
#  Note that the two drift models apply errors in different ways. For an accurate simulation, write a custom method that
#  is consistent with your experimental characterization and modeling method. If you need help implementing your device
#  drift model, contact the developers or raise an issue on GitHub
#

def applyDriftModel(input_, T, drift_model, vcp, param_root, clip_output, mask):

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


    if drift_model == "SONOS_interpolate":
        #
        # This simple drift model interpolates the state retention data for the SONOS
        # charge trapping memory device in
        # T. P. Xiao, et al. "An Accurate, Error-Tolerant, and Energy-Efficient Neural
        # Network Inference Engine Based on SONOS Analog Memory", 
        # IEEE Transactions on Circuits and Systems-I 69 (4), 2022.
        # https://ieeexplore.ieee.org/abstract/document/9669117
        #
        # For this device, the drift and the device-to-device variability were separately
        # extracted from experimental data collected at t = 0, 1, 2, and 5 days after
        # programming. The model applies the drift and the random variability separately.
        # For time points different from the measured times, the behavior is interpolated or
        # extrapolated. Extrapolation to drift times significantly longer than 5 days may be
        # inaccurate.
        #
        # Note that this function applies both drift and programming error, so it is
        # not necessary to separately apply programming errors
        #
        Imax = 1600 # nA

        # Get initial currents
        if on_off_ratio == 0:
            Imin = 0
        else:
            Imin = Imax/on_off_ratio
        I = Imin + (Imax-Imin) * (input_ - vcp.minimum) / vcp.range 
        if param_root.numeric_params.useGPU:
            I = cp.asnumpy(I)

        # The following are based on Fig. 4(e) of the above reference.

        # List of currents measured at time zero
        Id_initial = np.array([52.36,101.54,151.1, 202.17,254.18,305.76,361.97,408.88,\
            512.92,607.13,721.61,825.66,930.48,1034.71,1303.16,1585.32])
        A0, B0 = 0.00188704, 39.5656
        # List of currents measured at a later time (days)
        Id_final1 = np.array([54.94,103.82,155.11,206.69,259.1,308.48,363.51,411.18,\
            512.22,601.67,711.76,817.82,917.37,1019.99,1281.87,1553.01])
        A1, B1 = 0.00203079, 46.1065
        Id_final2 = np.array([56.51,105.8,157.73,210.48,262.89,312.24,369.76,417.15,\
            519.34,604.93,715.29,814.42,918.3,1019.02,1278.3,1550.17])
        A2, B2 = 0.00205587, 50.3982
        Id_final5 = np.array([58.44,110.93,162.27,213.43,273.47,319.09,377.16,421.33,\
            524.99,608.45,717.14,819.56,918.27,1023.06,1278.56,1543.26])
        A5, B5 = 0.00217328, 53.4358

        # If T is exactly at the measured time points, no need to interpolate
        if T in (1,2,5):
            if T == 1:
                Id_final = Id_final1
                A, B = A1, B1
            elif T == 2:
                Id_final = Id_final2
                A, B = A2, B2
            elif T == 5:
                Id_final = Id_final5
                A, B = A5, B5

        # Interpolate between the measured time points
        else:
            t_vec = np.array([0,1,2,5])
            Id_final_mat = np.zeros((4,16))
            AB_mat = np.zeros((2,4))
            Id_final_mat[0,:] = Id_initial
            Id_final_mat[1,:] = Id_final1
            Id_final_mat[2,:] = Id_final2
            Id_final_mat[3,:] = Id_final5
            
            Id_final = np.zeros(len(Id_initial))
            for i in range(len(Id_initial)):
                interp_func0 = interp1d(t_vec,Id_final_mat[:,i],kind='linear',copy=True,fill_value='extrapolate')
                Id_final[i] = interp_func0(T)

            As = np.array([A0,A1,A2,A5])
            Bs = np.array([B0,B1,B2,B5])
            A = interp1d(t_vec,As,kind='linear',copy=True,fill_value='extrapolate')(T)
            B = interp1d(t_vec,Bs,kind='linear',copy=True,fill_value='extrapolate')(T)

        # Interpolate drift characteristic vs conductance
        interp_func = interp1d(Id_initial,Id_final,kind='linear',copy=True,fill_value='extrapolate')
        I = interp_func(I)

        if param_root.numeric_params.useGPU:
            I = cp.array(I)
            if T not in (1,2,5):
                A = cp.array(A)
                B = cp.array(B)

        # Apply errors
        sigma_I = ncp.maximum(B - B*ncp.exp(-A*I), 0)
        sigma_W = sigma_I / (Imax - Imin)
        input_ = vcp.minimum + vcp.range*(I - Imin)/(Imax - Imin)

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


    elif drift_model == "PCM_Joshi":
        #
        # This drift + programming error model is based on the data for the phase change memory device #in:
        # V. Joshi, et al. "Accurate deep neural network inference using computational phase-change memory", 
        # Nature Communications 11, 2473, 2020.
        # https://www.nature.com/articles/s41467-020-16108-9
        # Supplementary Information, Note 2
        #
        # SUGGESTED ON/OFF RATIO : 100
        #
        # Note: The drift "model" in Fig 4c of the main paper does not seem to use the analytical
        # model presented in the SI but instead seems to use an interpolation of the measure data
        # vs time. Also, the accuracy vs time results in Fig. 4d includes an algorithmic drift compensation
        # before inference that is not included here
        #

        # First, convert xbar normalized conductances to target PCM conductances
        # Following SI, the target conductances are defined as the measured conductances at t=20s
        # Max conductance in PCM model
        Gmax = 25 # uS
        if on_off_ratio == 0:
            Gmin = 0
        else:
            Gmin = Gmax/on_off_ratio
        G = Gmin + (Gmax - Gmin) * (input_ - vcp.minimum) / vcp.range

        # The drift exponent depends on the target conductance value and is a random variable
        # to account for variability in drift
        # See Eq. 8-9, Supplementary Note 2 for the analytical drift model and the two parameters below
        mu_v = ncp.maximum(-0.0045*G + 0.08,ncp.minimum(0.06,ncp.maximum(-0.00169*G+0.0825, 0.055)))
        sigma_v = 1/(8*G + 30.0320)

        # Apply random errors to the exponent v
        if param_root.numeric_params.useGPU:
            randMat_v = ncp.random.normal(scale=sigma_v, size=input_.shape, dtype=input_.dtype)
        else:
            randMat_v = ncp.random.normal(scale=sigma_v, size=input_.shape).astype(input_.dtype)
        v = mu_v + randMat_v

        # Apply drift and program error according to the procedure in Eqs 12-16 of Supplementary Information
        T0 = 27.36
        T1 = 55.39
        T *= 86400 # days to seconds

        if T < T1:
            raise ValueError("For PCM drift model, time must be at minimum 55.39s. For shorter times, use programming error model.")

        # Programming noise at T0: fit to blue curve in Fig. 3b of main paper
        A, B, C = -0.00178767, 0.07585724, 0.28638599
        sigma_G0 = ncp.maximum(A*(G**2) + B*G + C, 0)

        # Apply noise and drift at T0
        if param_root.numeric_params.useGPU:
            randMat0 = ncp.random.normal(scale=sigma_G0, size=input_.shape, dtype=input_.dtype)
        else:
            randMat0 = ncp.random.normal(scale=sigma_G0, size=input_.shape).astype(input_.dtype)
        G_T0 = G * pow(T0/20,-v) + randMat0

        # Programming noise at T1: fit to black curve in Fig. 16 of SI
        A, B, C, D = 1.904e-4, -7.290e-3, 1.004e-1, 1.923e-1
        sigma_G1 = ncp.maximum(A*(G**3) + B*(G**2) + C*G + D, 0)

        # Apply drift at T1
        if param_root.numeric_params.useGPU:
            randMat1 = ncp.random.normal(scale=sigma_G1, size=input_.shape, dtype=input_.dtype)
        else:
            randMat1 = ncp.random.normal(scale=sigma_G1, size=input_.shape).astype(input_.dtype)
        G_T1 = G_T0 * pow(T1/T0,-v) + randMat1

        # Apply drift at T
        G_T = G_T1 * pow(T/T1,-v)

        # Convert back to CrossSim weight units
        # The part below does not need to be modified, unless desired
        input_ = vcp.minimum + vcp.range * (G_T - Gmin) / (Gmax - Gmin)

        if clip_output:
            input_ = vcp.clip(input_)
        if mask is not None:
            input_ *= mask
        return input_


    ##########################################################
    ### ----- END DEFINITION OF CUSTOM DEVICE MODELS
    ##########################################################

    else:
        raise ValueError("Custom interpolation drift error type not recognized")