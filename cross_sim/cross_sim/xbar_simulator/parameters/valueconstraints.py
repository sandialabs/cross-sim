#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import copy

from .base import ParametersBase, Parameter
from . import parameter_defaults as params
from .parameter_defaults import CrossbarTypeEnum, SimTypeEnum

import numpy as np
import time

class ClipConstraints(ParametersBase):

    def post_set(self):
        if (self.maximum is not None) and (self.minimum is not None):
            self.override_readonly = True

            self.range = self.maximum - self.minimum
            self.middle = (self.range / 2.0) + self.minimum
            self.absmaxmin = max(abs(self.maximum), abs(self.minimum))

            self.override_readonly = False

    # set post set functions (copy to valueconstraints)
    minimum = Parameter(name="minimum", post_set=post_set)
    maximum = Parameter(name="maximum", post_set=post_set)

    # set readonly vars
    range = Parameter(name="range", readonly=True)
    middle = Parameter(name="middle", readonly=True)
    absmaxmin = Parameter(name="absmaxmin", readonly=True)

    def __init__(self, param_root, **kwargs):

        attributes = params.ClipConstraintsDefaults.copy()

        #override defaults
        attributes.update(kwargs)
        #load the defaults
        ParametersBase.__init__(self,param_root, **attributes)


    def clip(self, input_):
        if self.param_root.algorithm_params.disable_clipping:
            return input_
        else:
            return input_.clip(self.minimum, self.maximum)  # use numpy clip function


    def scale_from(self, other, input_=1.0):
        """
        :param other: a clip constraint that has the range to scale from
        :param input_: value(s) to be scaled
        :return: take a value from other's range and scale it to self's range
        """
        input_ *= self.range / other.range
        return input_

    def scale_to(self, other, input_=1.0):
        """
        :param other: a clip constraint that has the range to scale to
        :param input_: value(s) to be scaled
        :return:  take a value in self's range and scale it to other's range
        """

        input_ *= other.range / self.range
        return input_

    def __mul__(self, other):
        newself = copy.deepcopy(self)
        newself *= other
        return newself

    def __imul__(self, other):
        self.minimum *= other
        self.maximum *= other
        return self

    def __truediv__(self, other):
        newself = copy.deepcopy(self)
        newself /= other
        return newself

    def __itruediv__(self, other):
        self *= (1.0 / other)
        return self


class NormalError(ParametersBase):
    '''
    Defines requirements and implementation of normal (gaussian) errors
    '''
    if False: # parameters for IDE code completion
        sigma = float
        proportional = float
        keep_within_range = bool

    def __init__(self, param_root, **kwargs):

        attributes = params.NormalErrorDefaults.copy()

        #override defaults
        attributes.update(kwargs)

        #load the defaults
        ParametersBase.__init__(self,param_root, **attributes)


    def apply(self, input_, vcp, noise_model="none", std_matrix=None):
        """

        :param input_: the input values to apply the noise to (modify in place)
        :param vcp:  value constraint parameters:  the value constraint parameter object that has the overall clipping rrange
        :type vcp: ClipQuantizeAndNoiseConstraints
        :param scale: scale the noise (and noise clipping) by this amount
        :return: input with noise added
        """
        if noise_model == "none":
            return input_

        if noise_model == "alpha" and self.sigma == 0:
            return input_

        scale = vcp.range
        sigma = self.sigma

        if not self.proportional:
            sigma *= scale

        global ncp
        if self.param_root.numeric_params.useGPU:
            import cupy as cp
            cp.cuda.Device(self.param_root.numeric_params.gpu_id).use()
            ncp = cp
        else:
            rng = np.random.default_rng()
            ncp = np

        # Clip noised result?
        clip = (self.keep_within_range and not self.param_root.algorithm_params.disable_clipping)

        # Batched? (sliding windows or training examples)
        x_par = self.param_root.numeric_params.x_par
        y_par = self.param_root.numeric_params.y_par
        Nex_par = self.param_root.numeric_params.Nex_par
        batched = input_.ndim >= 2 and (x_par > 1 or y_par > 1 or Nex_par > 1)

        # For depthwise convolution, there is a block diagonal matrix within another block diagonal matrix
        is_depthwise = (self.param_root.convolution_parameters.is_conv_core and self.param_root.convolution_parameters.depthwise)
        if is_depthwise:
            Nch = self.param_root.convolution_parameters.Noc
            Kx = self.param_root.convolution_parameters.Kx
            Ky = self.param_root.convolution_parameters.Ky
            if self.param_root.convolution_parameters.bias:
                Nrows = Kx*Ky + 1
            else:
                Nrows = Kx*Ky

        if x_par > 1 and y_par > 1 and Nex_par > 1 and is_depthwise:
            raise ValueError("Not yet supported: depthwise convolution + read noise + SW packing + parasitic resistance. For this neural network,"+\
                " please set x_par, y_par = (1, 1) in get_xy_parallel_parasitics().")

        # For all cases that are not a depthwise convolution, the process is simple regardless of whether
        # sliding windows are batched
        if not is_depthwise:
            if self.param_root.numeric_params.useGPU:
                Rall = ncp.random.normal(scale=1, size=input_.shape, dtype=input_.dtype)
            else:
                Rall = rng.standard_normal(size=input_.shape, dtype=input_.dtype)

            if noise_model == "alpha":
                Rall *= sigma
            else:
                Rall *= std_matrix 

            if self.proportional and noise_model == "alpha":
                Rall += 1
                input_ *= Rall
            else:
                input_ += Rall

        else:
            if self.param_root.numeric_params.useGPU:
                Rmat = ncp.random.normal(scale=1, size=(input_.shape[0],Nrows), dtype=input_.dtype)
            else:
                Rmat = rng.standard_normal(size=(input_.shape[0],Nrows), dtype=input_.dtype)

            if noise_model == "alpha":
                Rmat *= sigma
            else:
                Rmat *= std_matrix 

            if not batched:
                for ch in range(Nch):
                    y_start, y_end = ch*Ky*Kx, (ch+1)*Ky*Kx
                    if self.proportional and noise_model == "alpha":
                        input_[ch,y_start:y_end] *= 1 + Rmat[ch,0:(Kx*Ky)]
                    else:
                        input_[ch,y_start:y_end] += Rmat[ch,0:(Kx*Ky)]
                if self.param_root.convolution_parameters.bias:
                    if self.proportional and noise_model == "alpha":
                        input_[:,-1] *= 1 + Rmat[:,-1]
                    else:
                        input_[:,-1] += Rmat[:,-1]

                for ch in range(Nch):
                    y_start, y_end = ch*Ky*Kx, (ch+1)*Ky*Kx
                    input_[:ch,y_start:y_end] = 0
                    input_[(ch+1):,y_start:y_end] = 0

            else:
                if x_par > 1 or y_par > 1:
                    Ncopy = x_par * y_par
                else:
                    Ncopy = Nex_par
                Nx0,Ny0 = input_.shape[0]//Ncopy, input_.shape[1]
                if input_.shape[0]%Ncopy != 0:
                    raise ValueError("Could not evenly divide weight matrix into parallel VMMs -- change x_par, y_par")

                for m in range(Ncopy):
                    x_start0, x_end0 = m*Nx0, (m+1)*Nx0
                    for ch in range(Nch):
                        y_start, y_end = ch*Ky*Kx, (ch+1)*Ky*Kx
                        if self.proportional and noise_model == "alpha":
                            input_[x_start0+ch,y_start:y_end] *= 1 + Rall[x_start0+ch,0:(Kx*Ky)]
                        else:
                            input_[x_start0+ch,y_start:y_end] += Rall[x_start0+ch,0:(Kx*Ky)]
                    if self.param_root.convolution_parameters.bias:
                        if self.proportional and noise_model == "alpha":
                            input_[x_start0:x_end0,-1] *= 1 + Rall[x_start0:x_end0,-1]
                        else:
                            input_[x_start0:x_end0,-1] += Rall[x_start0:x_end0,-1]

        if clip:
            input_ = vcp.clip(input_)
            
        return input_


class UniformError(ParametersBase):
    '''
    Defines requirements and implementation of uniform errors
    '''
    if False: # parameters for IDE code completion
        range = float
        keep_within_range = bool

    def __init__(self, param_root, **kwargs):

        attributes = params.UniformErrorDefaults .copy()

        #override defaults
        attributes.update(kwargs)
        #load the defaults
        ParametersBase.__init__(self,param_root, **attributes)


    def apply(self, input_, vcp):
        if bool(self.range):
            range = self.range
            scale = vcp.range

            range *= scale
            half_error = range / 2.0

            x_par = self.param_root.numeric_params.x_par
            y_par = self.param_root.numeric_params.y_par

            if input_.ndim!=2 or (x_par == 1 and y_par == 1):
                rand = np.random.uniform(low=-half_error, high=half_error, size=input_.shape).astype(input_.dtype)
                input_ += rand
                if self.keep_within_range and not self.param_root.algorithm_params.disable_clipping:
                    input_ = vcp.clip(input_)

            else:
                Ncopy = x_par * y_par
                Nx0,Ny0 = input_.shape[0]//Ncopy, input_.shape[1]//Ncopy
                randshape = (Nx0,Ny0*Ncopy)
                rand = np.random.uniform(low=-half_error, high=half_error, size=randshape, dtype=input_.dtype).astype(input_.dtype)
                for m in range(Ncopy):
                    x_start, x_end = m*Nx0, (m+1)*Nx0
                    y_start, y_end = m*Ny0, (m+1)*Ny0
                    input_[x_start:x_end,y_start:y_end] += rand[:,y_start:y_end]
                    if self.keep_within_range and not self.param_root.algorithm_params.disable_clipping:
                        input_[x_start:x_end,y_start:y_end] = input_[x_start:x_end,y_start:y_end].clip(vcp.minimum,vcp.maximum)

        return input_



class ClipQuantizeAndNoiseConstraints(ClipConstraints, ParametersBase):
    '''
    Union of :py:class:`ClipConstraints` and :py:class:`QuantizationAndNoiseConstraints`
    
    Handles quantization of values
    '''


    #define parameters for IDE / code completion
    if False:
        normal_error_pre = NormalError
        normal_error_post = NormalError
        uniform_error_post = UniformError
        bits = int
        sign_bit = bool
        stochastic_rounding = bool
        extra_half_bit = bool


    def post_set(self):  # calculate and save self.levels
        self.override_readonly = True
        if self.bits is not None:
            if self.bits==0:
                self.levels = None
            else:
                if self.sign_bit is not None:  # check if sign bit is set yet
                    if self.sign_bit:
                        # self.levels = 2 ** (self.bits + 1) - 1
                        self.levels = 2 ** self.bits - 1
                    else:
                        self.levels = 2 ** self.bits

        self.override_readonly = False



    #set post_set params  (copy to valueconstraints)
    bits = Parameter(name="bits", post_set=post_set)
    sign_bit = Parameter(name="sign_bit", post_set=post_set)

    # set readonly values
    # number of levels in number (i.e. 2 bits + sign  = 7 levels, 3 bits, no sign  = 8 levels)
    levels = Parameter(name="levels", readonly=True)


    def __init__(self, param_root, **kwargs):

        #add quantization constraint defaults
        attributes = params.QuantizationConstaintsDefaults.copy()
        #add clip constraint defaults
        attributes.update(params.ClipConstraintsDefaults)
        #override defaults
        attributes.update(kwargs)

        #add noise param objects for quantization constraints
        attributes['normal_error_pre']=NormalError(param_root)
        attributes['normal_error_post']=NormalError(param_root)
        attributes['uniform_error_post']=UniformError(param_root)

        #load the defaults
        ParametersBase.__init__(self,param_root, **attributes)



    def quantize_nobits(self, input_):
        input_ = self.normal_error_pre.apply(input_, self)
        input_ = self.normal_error_post.apply(input_, self)
        input_ = self.uniform_error_post.apply(input_, self)
        return input_

    def quantize(self, input_):
        if not self.bits:
            return self.quantize_nobits(input_)

        # apply normal pre-error
        # input_ = self.normal_error_pre.apply(input_, self)

        # set qmult (quantization multiplier):  multiply by this factor to convert every level to an absolute range of 1
        qmult = (self.levels-1) / self.range  #The -1 is because the first level is 0, i.e. 2 bits define 3 segments between 0 and 3

        # do quantization using rounding
        input_ -= self.minimum  #shift min to zero
        input_ *= qmult   #multiply by a quantization factor to allow for rounding -> sigma becomes defined relative to 1 level

        if self.stochastic_rounding ==True:
            input_floor = np.floor(input_)
            input_ = input_floor+(np.random.random_sample(np.shape(input_))<(input_-input_floor) )
        else:
            input_ = np.rint(input_, out=input_)

        input_ /= qmult
        input_ += self.minimum #shift zero back

        # apply post errors
        # input_ = self.normal_error_post.apply(input_, self)
        # input_ = self.uniform_error_post.apply(input_, self)

        return input_

    def clip_and_quantize(self, input_):
        return self.quantize(self.clip(input_))





class XbarParams(ParametersBase):
    if False:        # define parameters for easy code completion (these can be commented out, it's only to help the IDE)
        weights = ClipQuantizeAndNoiseConstraints
        weight_clipping = ClipConstraints
        row_input = ClipQuantizeAndNoiseConstraints
        col_input = ClipQuantizeAndNoiseConstraints
        row_update = ClipQuantizeAndNoiseConstraints
        col_update = ClipQuantizeAndNoiseConstraints
        row_output = ClipQuantizeAndNoiseConstraints
        col_output = ClipQuantizeAndNoiseConstraints

    def __init__(self, param_root):
        attributes = params.XbarParamsDefaults['attributes'].copy()

        attributes['weights']=ClipQuantizeAndNoiseConstraints(param_root,**params.XbarParamsDefaults['weights'])
        attributes['weight_clipping']=ClipConstraints(param_root,**params.XbarParamsDefaults['weight_clipping'])
        attributes['row_input']=ClipQuantizeAndNoiseConstraints(param_root,**params.XbarParamsDefaults['row_input'])
        attributes['col_input']=ClipQuantizeAndNoiseConstraints(param_root,**params.XbarParamsDefaults['col_input'])
        attributes['row_update']=ClipQuantizeAndNoiseConstraints(param_root,**params.XbarParamsDefaults['row_update'])
        attributes['col_update']=ClipQuantizeAndNoiseConstraints(param_root,**params.XbarParamsDefaults['col_update'])
        attributes['row_output']=ClipQuantizeAndNoiseConstraints(param_root,**params.XbarParamsDefaults['row_output'])
        attributes['col_output']=ClipQuantizeAndNoiseConstraints(param_root,**params.XbarParamsDefaults['col_output'])

        ParametersBase.__init__(self,param_root, **attributes)

    def copy_clip_constaints_to(self,new_contraints):
        """
        Copy only the clip constraints to another XbarParams, AlgorithmParams or WrapperParams
        modify input in place

        :param new_contraints: the clip constraints in self will be copied to new_constraints
        :type new_contraints: WrapperParams
        :return: modifed constraints

        """
        param_root = new_contraints.param_root
        new_contraints.weights= ClipConstraints(param_root, minimum=self.weights.minimum, maximum=self.weights.maximum)
        new_contraints.row_input= ClipConstraints(param_root, minimum=self.row_input.minimum, maximum=self.row_input.maximum)
        new_contraints.col_input= ClipConstraints(param_root, minimum=self.col_input.minimum, maximum=self.col_input.maximum)
        new_contraints.row_update= ClipConstraints(param_root, minimum=self.row_update.minimum, maximum=self.row_update.maximum)
        new_contraints.col_update= ClipConstraints(param_root, minimum=self.col_update.minimum, maximum=self.col_update.maximum)
        new_contraints.row_output= ClipConstraints(param_root, minimum=self.row_output.minimum, maximum=self.row_output.maximum)
        new_contraints.col_output= ClipConstraints(param_root, minimum=self.col_output.minimum, maximum=self.col_output.maximum)

        return new_contraints


class AlgorithmParams(ParametersBase):
    if False:        # define parameters for easy code completion (these can be commented out, it's only to help the IDE)
        weights = ClipConstraints
        row_input = ClipConstraints
        col_input = ClipConstraints
        row_update = ClipConstraints
        col_update = ClipConstraints
        row_output = ClipConstraints
        col_output = ClipConstraints
        row_update_portion = float
        subtract_current_in_xbar = bool
        calculate_inner_from_outer = bool
        disable_clipping = bool
        serial_read_scaling = float
        crossbar_type = CrossbarTypeEnum
        sim_type = SimTypeEnum

    crossbar_type = Parameter(name="crossbar_type",
                              post_set=ParametersBase.generate_enum_post_set("crossbar_type", CrossbarTypeEnum))
    sim_type = Parameter(name="sim_type", post_set=ParametersBase.generate_enum_post_set("sim_type", SimTypeEnum))

    def __init__(self, param_root):
        attributes = params.AlgorithmParamsDefaults['attributes'].copy()

        attributes['weights']=ClipConstraints(param_root,**params.AlgorithmParamsDefaults['weights'])
        attributes['row_input']=ClipConstraints(param_root,**params.AlgorithmParamsDefaults['row_input'])
        attributes['col_input']=ClipConstraints(param_root,**params.AlgorithmParamsDefaults['col_input'])
        attributes['row_update']=ClipConstraints(param_root,**params.AlgorithmParamsDefaults['row_update'])
        attributes['col_update']=ClipConstraints(param_root,**params.AlgorithmParamsDefaults['col_update'])
        attributes['row_output']=ClipConstraints(param_root,**params.AlgorithmParamsDefaults['row_output'])
        attributes['col_output']=ClipConstraints(param_root,**params.AlgorithmParamsDefaults['col_output'])

        ParametersBase.__init__(self,param_root, **attributes)


class WrapperParams(ParametersBase):
    if False:        # define parameters for easy code completion (these can be commented out, it's only to help the IDE)
        weights = ClipConstraints
        row_input = ClipConstraints
        col_input = ClipConstraints
        row_update = ClipConstraints
        col_update = ClipConstraints
        row_output = ClipConstraints
        col_output = ClipConstraints



    def __init__(self, param_root):
        attributes =params.WrapperParamsDefaults['attributes'].copy()
        attributes['weights']=ClipConstraints(param_root,minimum=None,maximum=None)
        attributes['row_input']=ClipConstraints(param_root,minimum=None,maximum=None)
        attributes['col_input']=ClipConstraints(param_root,minimum=None,maximum=None)
        attributes['row_update']=ClipConstraints(param_root,minimum=None,maximum=None)
        attributes['col_update']=ClipConstraints(param_root,minimum=None,maximum=None)
        attributes['row_output']=ClipConstraints(param_root,minimum=None,maximum=None)
        attributes['col_output']=ClipConstraints(param_root,minimum=None,maximum=None)

        ParametersBase.__init__(self,param_root, **attributes)