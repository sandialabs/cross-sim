#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

__authors__ = 'txiao'

import os
import numpy as np
from warnings import warn
from scipy.interpolate import interp1d

from .base import ParametersBase, Parameter
from . import parameter_defaults as params
from .valueconstraints import NormalError
from .parameter_defaults import CrossbarTypeEnum


class WeightErrorParameters(ParametersBase):
    """
    parameters for weight drift
    """
    if False:
        sigma_error = float
        error_model = str
        noise_model = str
        drift_model = str
        W0_vec = float
        sigma0_vec = float
        T = float
        keep_within_range = bool
        TID = float
        rad_type = str

    def __init__(self, param_root, **kwargs):
        ParametersBase.__init__(self, param_root, **params.WeightErrorParamsDefaults)


    def manual_post_set(self):
    
        # For multiAlpha model (deprecated), sort the weight pin points        
        if self.W0_vec is not None:  # only process if error is known for at least one weight value
            W0_vec = np.array(self.W0_vec)
            W0_inds = W0_vec.argsort()
            self.W0_vec = W0_vec[W0_inds]
            if self.sigma0_vec is not None:
                self.sigma0_vec = self.sigma0_vec[W0_inds]


    def applyWeightQuantization(self, input_, vcp):

        Wbits = self.param_root.algorithm_params.weight_bits

        if self.param_root.numeric_params.useGPU:
            import cupy as cp
            cp.cuda.Device(self.param_root.numeric_params.gpu_id).use()
            ncp = cp
        else:
            ncp = np

        if Wbits > 0:
            Wmax = vcp.maximum
            Wmin = vcp.minimum

            # set qmult:  multiply by this factor to convert every level to an absolute range of 1
            # this allows quantization to be done by rounding
            qmult = (2 ** Wbits) / (Wmax - Wmin)  # The first level maps to a nonzero current
            input_ -= Wmin  # shift min to zero
            input_ *= qmult  # multiply by a quantization factor to allow for rounding

            input_ = ncp.rint(input_, out=input_)

            input_ /= qmult
            input_ += Wmin  # shift zero back

        return input_


    # For depthwise convolution, some elements are actually fictitious
    # These need to have hard zero values after applying error; create a mask to ensure this
    def depthwise_mask(self, input_shape, ncp):

        Noc = self.param_root.convolution_parameters.Noc
        Kx = self.param_root.convolution_parameters.Kx
        Ky = self.param_root.convolution_parameters.Ky
        
        # If this is a split subarray, we must find out which one it is
        # If it is not split, this will default to zero
        subarray_id = self.param_root.convolution_parameters.subarray_id
        
        if subarray_id > 0:
            if input_shape[1] % (Kx*Ky) != 0:
                print(input_shape)
                print(Kx*Ky)
                raise ValueError("For split depthwise convolution, haven't implemented the case where max # rows is not a multiple of kernel size")
            i_offset = subarray_id * input_shape[1]//(Kx*Ky)
        else:
            i_offset = 0

        mask = ncp.zeros(input_shape)
        for i in range(Noc):
            i_start, i_end = i*Kx*Ky, (i+1)*Kx*Ky
            if i + i_offset < Noc:
                mask[i+i_offset,i_start:i_end] = 1

        if self.param_root.convolution_parameters.bias:
            if self.param_root.convolution_parameters.last_subarray:
                mask[:,-1] = 1

        return mask


    # Apply programming error
    # Mode:
    #   'none' : don't apply any error
    #   'alpha' : single sigma error value for all weights
    #   'multiAlpha' : provide a vector of sigma error values and interpolate betwen them
    #   'SONOS' : use an analytical expression for the sigma vs weight for SONOS
    #   'DWMTJ' : use an analytical expression for the sigma vs weight for DWMTJ (straight)
    def applyProgrammingError(self, input_, vcp):

        if self.error_model == "none":
            return input_

        if self.T > 0 and self.drift_model != "none" and self.error_model != "alpha":
            print("Warning: custom device programming error model ignored since drift model is active")
            return input_

        if self.param_root.numeric_params.useGPU:
            import cupy as cp
            cp.cuda.Device(self.param_root.numeric_params.gpu_id).use()
            ncp = cp
        else:
            ncp = np

        clip_output = (self.keep_within_range and not self.param_root.algorithm_params.disable_clipping)

        # if depthwise convolution, some elements are fictitious and should be set to a hard zero
        mask = None
        if self.param_root.convolution_parameters.is_conv_core and self.param_root.convolution_parameters.depthwise:
            mask = self.depthwise_mask(input_.shape, ncp)

        # Check error type and apply error

        if self.error_model == "alpha":
            sigma = self.sigma_error
            if not self.proportional:
                sigma *= vcp.range
            if self.sigma_error > 0:
                if self.param_root.numeric_params.useGPU:
                    randMat = ncp.random.normal(scale=sigma, size=input_.shape, dtype=input_.dtype)
                else:
                    randMat = ncp.random.normal(scale=sigma, size=input_.shape).astype(input_.dtype)
                if self.proportional:
                    randMat += 1
                    input_ *= randMat
                else:
                    input_ += randMat
                # clip the final value to the clipping range
                if clip_output:
                    input_ = vcp.clip(input_)
                if mask is not None:
                    input_ *= mask
                return input_

            else:
                return input_

        elif self.error_model == "multiAlpha":
            # --- DEPRECATED ---
            # Rather than this, define a custom error model in weight_error_device_custom.py
            # Linearly interpolate the programming error sigma based on
            #   A vector of initial weights W0_vec
            #   A vector of corresponding programming errors sigma0_vec
            if self.sigma0_vec.any():
                output = input_.copy()
                numBins = len(self.W0_vec) + 1
                if self.param_root.numeric_params.useGPU:
                    randMat = ncp.random.normal(size=input_.shape, dtype=input_.dtype)
                else:
                    randMat = ncp.random.normal(size=input_.shape).astype(input_.dtype)
                    
                for k in range(numBins):
                    if k == 0:
                        bin_k = (input_ <= self.W0_vec[0])
                        m, mprev = 1, 0
                    elif k == len(self.W0_vec):
                        bin_k = (input_ > self.W0_vec[-1])
                        m, mprev = -1, -2
                    else:
                        bin_k = ncp.logical_and(input_ > self.W0_vec[k - 1], input_ <= self.W0_vec[k])
                        m, mprev = k, k - 1

                    # Linear interpolation of sigma
                    Wk, Wkprev = self.W0_vec[m], self.W0_vec[mprev]
                    sigma0k, sigma0kprev = self.sigma0_vec[m], self.sigma0_vec[mprev]
                    sigma0s = ((sigma0k - sigma0kprev) / (Wk - Wkprev)) * input_ + (
                                sigma0kprev * Wk - sigma0k * Wkprev) / (Wk - Wkprev)
                    
                    # Apply programming error
                    if self.proportional:
                        varMat = sigma0s * randMat
                        output *= bin_k * (1 + varMat)
                    else:
                        sigma0s *= vcp.range
                        varMat = sigma0s * randMat
                        output += bin_k * varMat

                # Clip values and mask (if depthwise)
                if clip_output:
                    output = vcp.clip(output)
                if mask is not None:
                    output *= mask
                return output

            else:
                return input_

        else:
            from .custom_device.weight_error_device_custom import applyCustomProgrammingError
            return applyCustomProgrammingError(input_, self.error_model, vcp, self.param_root, clip_output, mask)


    # Set the standard deviation of the read noise for each element of the array
    # Do not apply the errors to the elements at this point (this is done at runtime during each MVM)
    def setCustomReadNoise(self, input_, vcp):

        if self.noise_model == "none":
            return input_

        if self.param_root.numeric_params.useGPU:
            import cupy as cp
            cp.cuda.Device(self.param_root.numeric_params.gpu_id).use()
            ncp = cp
        else:
            ncp = np

        # if depthwise convolution, some elements are fictitious and should be set to a hard zero
        mask = None
        if self.param_root.convolution_parameters.is_conv_core and self.param_root.convolution_parameters.depthwise:
            mask = self.depthwise_mask(input_.shape, ncp)

        from .custom_device.weight_readnoise_device_custom import setDeviceReadNoise
        return setDeviceReadNoise(input_, self.noise_model, vcp, self.param_root, mask)


    def applyDrift(self, input_, vcp):
        """
        :param input_: the set of weights to apply drift to
        :param vcp:  value constraint parameters:  the value constraint parameter object that has the overall clipping rrange
        :type vcp: ClipQuantizeAndNoiseConstraints
        :return: input with the appropriate drift and time-dependent errors added
        """

        # If there is no drift, return the input
        if self.T == 0 or self.drift_model == "none":
            return input_

        # if depthwise convolution, some elements are fictitious and should be set to a hard zero
        mask = None
        if self.param_root.convolution_parameters.is_conv_core and self.param_root.convolution_parameters.depthwise:
            mask = self.depthwise_mask(input_.shape)
        clip_output = (self.keep_within_range and not self.param_root.algorithm_params.disable_clipping)

        from .custom_device.weight_drift_device_custom import applyDriftModel
        return applyDriftModel(input_, self.T, self.drift_model, vcp, self.param_root, clip_output, mask)


    # Apply TID radiation effects
    def applyRadEffects(self, input_, vcp):

        try:
            from .weight_error_radiation import applyRadEffects_TID
        except ModuleNotFoundError:
            raise NotImplementedError("Missing implementation of radiation effects model")

        clip_output = (self.keep_within_range and not self.param_root.algorithm_params.disable_clipping)    
        if self.param_root.convolution_parameters.is_conv_core and self.param_root.convolution_parameters.depthwise:
            mask = self.depthwise_mask(input_.shape)
        else:
            mask = None

        return applyRadEffects_TID(input_, self.rad_type, self.TID, vcp, self.param_root, clip_output, mask)