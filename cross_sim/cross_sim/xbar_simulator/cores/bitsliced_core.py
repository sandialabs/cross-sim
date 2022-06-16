#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

__authors__ = 'txiao'

import numpy as np
from .debug import print_debug_calls
from .wrapper_core import WrapperCore
from ..parameters.parameter_defaults import UpdateModelEnum


class BitslicedCore(WrapperCore):
    '''
    A core consisting of two or more arrays that implements bit-sliced weights
    At least two slices are required (otherwise, use balanced core).
    '''
    
    def __init__(self, clipper_core_factory, params):
        """

        :param clipper_core_factory:
        :param params: all parameters
        :type params: Parameters
        :return:
        """

        WrapperCore.__init__(self, clipper_core_factory, params)

        Nslices = params.xbar_params.Nslices
        Wbits = params.algorithm_params.weight_bits
        self.balanced = params.xbar_params.balanced_bitsliced
        self.ADC_per_ibit = params.xbar_params.ADC_per_ibit

        if Nslices == 1:
            raise ValueError("Should not initiate BitslicedCore if Nslices = 1. Please use balanced core.")

        if np.abs(self.params.xbar_params.weights.maximum - 1.0) > 1e-6:
            raise ValueError("For bit sliced core, please set Xbar max weight to 1.0")

        if self.ADC_per_ibit and not self.params.xbar_params.input_bitslicing:
            print("ADC per input bit enabled but input bit slicing disabled. Disabling ADC per input bit.")
            self.ADC_per_ibit = False

        # i = 0 is least significant bit
        self.Nslices = Nslices
        if self.balanced:
            self.core_slices = [ [clipper_core_factory(),clipper_core_factory()] for i in range(Nslices)]

            self.fast_balanced = params.xbar_params.fast_balanced
            if self.fast_balanced and self.params.numeric_params.read_noise.sigma > 0 and self.params.weight_error_params.noise_model != "none":
                print("Fast balanced core option cannot be used with read noise: reverting.")
                self.fast_balanced = False
            if self.fast_balanced and self.params.numeric_params.Rp > 0:
                print("Fast balanced core option cannot be used with parasitic resistance: reverting.")
                self.fast_balanced = False
            if self.fast_balanced and self.params.xbar_params.clip_Icol:
                print("Fast balanced core option cannot be used with current limiter: reverting.")
                self.fast_balanced = False

            if self.fast_balanced:
                # Initialize list of matrices
                self.W_balanced = [None for i in range(Nslices)]

        else:
            self.core_slices = [clipper_core_factory() for i in range(Nslices)]        
            if self.params.numeric_params.weight_reorder:
                raise ValueError("Topelitz reordering is not compatible with offset core")

        self.adc_range_option = self.params.xbar_params.adc_range_option
        if self.adc_range_option not in ("max","calibrated","granular"):
            raise ValueError("Invalid ADC range option for bitsliced core")
        if self.adc_range_option == "granular" and not self.ADC_per_ibit:
            raise ValueError("Granular ADC range option can currently only be used with digital input S&A.")
        if self.adc_range_option == "granular" and self.balanced:
            print("Warning: you are using granular with differential cells.")

        Nbits_reduction = params.xbar_params.Nbits_reduction
        if Nbits_reduction is not None:
            self.Nbits_reduction = Nbits_reduction
        if params.xbar_params.row_output.bits > 0 and self.adc_range_option == "calibrated" and Nbits_reduction is None:
            raise ValueError("ADC range calibration option selected for bit sliced core but calibrated ranges not specified")

        global ncp
        if params.numeric_params.useGPU:
            global cp
            import cupy as cp
            cp.cuda.Device(params.numeric_params.gpu_id).use()
            ncp = cp
        else:
            ncp = np

        self.i_mvm = 0

    def _given_inner_update_wrapper_limits(self):
        inner_constraints = self.params.xbar_params
        wrapper_constraints = self.params.wrapper_params

        #copy over inner constraints
        inner_constraints.copy_clip_constaints_to(wrapper_constraints)

        if inner_constraints.weights.minimum != 0:
            raise ValueError('BitSlicedCore cannot handle inner core with minimum value != 0.')
        icc_c_m = inner_constraints.weights.middle
        wrapper_constraints.weights.minimum -= icc_c_m
        wrapper_constraints.weights.maximum -= icc_c_m
        wrapper_constraints.row_output /= 2.0
        wrapper_constraints.col_output /= 2.0
        return wrapper_constraints

    def _given_wrapper_update_inner_limits(self):
        """
        Uses the wrapper constraints and updates inner constraints based on the current core type
        :return:
        """
        # save local copies of wrapper and outer clip constraints
        wrapper_cc = self.params.wrapper_params
        # outer_cc = self.params.algorithm_params
        inner_cc = self.params.xbar_params

        # for balanced core double wrapper output limits
        inner_cc.row_output.maximum = wrapper_cc.row_output.maximum*2
        inner_cc.row_output.minimum = wrapper_cc.row_output.minimum*2

        inner_cc.col_output.maximum = wrapper_cc.col_output.maximum*2
        inner_cc.col_output.minimum = wrapper_cc.col_output.minimum*2

        # copy the row/col update limits
        inner_cc.row_update.maximum = wrapper_cc.row_update.maximum
        inner_cc.row_update.minimum = wrapper_cc.row_update.minimum

        inner_cc.col_update.maximum = wrapper_cc.col_update.maximum
        inner_cc.col_update.minimum = wrapper_cc.col_update.minimum

        return inner_cc


    def _wrapper_set_matrix(self, matrix, applyErrors=True):
        
        # Quantize weights here, and not in numeric_core
        # Wbits as set in inference_net is the number of weight bits per device, assuming a balanced core
        # In this case we add one to get the true number of weight bits (pos and neg)
        Wbits = self.params.algorithm_params.weight_bits
        Nslices = self.params.xbar_params.Nslices
        signed_input = (self.params.xbar_params.col_input.minimum < 0)
        Wrange_xbar = self.params.xbar_params.weights.maximum - self.params.xbar_params.weights.minimum

        if Wbits % Nslices == 0:
            Wbits_slice = int(Wbits / Nslices)
        elif self.balanced:
            Wbits_slice = np.ceil((Wbits-1)/Nslices).astype(int)
        else:
            Wbits_slice = np.ceil(Wbits/Nslices).astype(int)

        # Shape of weight matrix before duplication
        self.W_shape = matrix.shape
        self.Wbits_slice = Wbits_slice
        self.Wmax = pow(2,Wbits)
        self.Woffset = pow(2,Wbits-1)

        # Place matrix in the range (-0.5, 0.5)
        W = matrix.copy()

        if self.params.xbar_params.weights.minimum > 0:
            W /= Wrange_xbar

        # First convert the matrix to the range (-2^(Wbits-1)-1, +2^(Wbits-1)-1)
        W *= self.Wmax

        # Then convert the matrix to the range (0, 2^Wbits) by adding a bias
        if not self.balanced:
            W += self.Woffset

        # Quantize the weights to Wbits
        W = np.rint(W, out=W)

        # Decompose W into bit slices
        W_slices = [None for i in range(Nslices)]
        if not self.balanced:
            # Positive weights case
            W_slices[0] = W % pow(2,Wbits_slice)
            for i in range(1,Nslices):
                if i < Nslices-1:
                    W = (W - W_slices[i-1]) / pow(2, Wbits_slice)
                    W_slices[i] = W % pow(2, Wbits_slice)
                else:
                    W_slices[i] = (W - W_slices[i-1]) / pow(2, Wbits_slice)
        else:
            # Negative weights case
            W_sign = np.sign(W)
            W_slices[0] = np.abs(W) % pow(2,Wbits_slice)
            for i in range(1,Nslices):
                if i < Nslices-1:
                    W = (np.abs(W) - W_slices[i-1]) / pow(2, Wbits_slice)
                    W_slices[i] = np.abs(W) % pow(2, Wbits_slice)
                else:
                    W_slices[i] = (np.abs(W) - W_slices[i-1]) / pow(2, Wbits_slice)
            for i in range(Nslices):
                W_slices[i] = W_slices[i]*W_sign

        for i in range(Nslices):
            W_i = W_slices[i]

            # Now xbar weights in the range (0, 1)
            W_i /= pow(2,Wbits_slice)-1

            if not self.balanced:
                # Since weights have already been quantized, set inner core quantization to zero
                self.core_slices[i].params.algorithm_params.weight_bits = 0

                if self.params.numeric_params.Rp > 0 and self.params.xbar_params.noRpSlices[i]:
                    self.core_slices[i].params.numeric_params.Rp = 0

                # Set ADC range
                if self.core_slices[i].xbar_params.row_output.bits > 0:

                    if self.adc_range_option == "max" or self.adc_range_option == "calibrated":

                        Nbits_adc = self.core_slices[i].xbar_params.row_output.bits
                        Nbits_in = self.params.xbar_params.col_input.bits

                        # Bring # rows to nearest power of 2
                        ymax = pow(2,np.round(np.log2(self.W_shape[1])))
                        # Correct to make level separation a multiple of the min cell current
                        ymax *= pow(2,Wbits_slice)/(pow(2,Wbits_slice)-1)

                        if signed_input:
                            # Further correction to ensure level separation is a multiple of the min cell current
                            ymax *= (pow(2,Nbits_adc)-2)/pow(2,Nbits_adc)
                            # Correct for input bits
                            ymax *= pow(2,Nbits_in-1)/(pow(2,Nbits_in-1)-1)

                            self.core_slices[i].xbar_params.row_output.minimum = -ymax
                            self.core_slices[i].xbar_params.row_output.maximum = ymax
                        else:
                            # Further correction to ensure level separation is a multiple of the min cell current
                            ymax *= (pow(2,Nbits_adc)-1)/pow(2,Nbits_adc)
                            # Correct for input bits
                            ymax *= pow(2,Nbits_in)/(pow(2,Nbits_in)-1)
                            self.core_slices[i].xbar_params.row_output.minimum = 0
                            self.core_slices[i].xbar_params.row_output.maximum = ymax

                        if self.adc_range_option == "calibrated":
                            # Reduce the ADC range by a power of 2
                            if self.Nbits_reduction[i] > 0:
                                self.core_slices[i].xbar_params.row_output.minimum /= pow(2,self.Nbits_reduction[i])
                                self.core_slices[i].xbar_params.row_output.maximum /= pow(2,self.Nbits_reduction[i])

                    elif self.adc_range_option == "granular":
                        Nbits_adc = self.core_slices[i].xbar_params.row_output.bits
                        Nbits_in = self.params.xbar_params.col_input.bits
                        ymin = 1/(pow(2,Wbits_slice)-1)
                        if signed_input:
                            corr = pow(2,Nbits_in-1)/(pow(2,Nbits_in-1)-1)
                            self.core_slices[i].xbar_params.row_output.minimum = -ymin * (pow(2,Nbits_adc-1)-1) * corr
                            self.core_slices[i].xbar_params.row_output.maximum = ymin * (pow(2,Nbits_adc-1)-1) * corr
                        else:
                            corr = pow(2,Nbits_in)/(pow(2,Nbits_in)-1)
                            self.core_slices[i].xbar_params.row_output.minimum = 0
                            self.core_slices[i].xbar_params.row_output.maximum = ymin * (pow(2,Nbits_adc)-1) * corr

                    self.core_slices[i].xbar_params.row_output.post_set()


                if self.params.xbar_params.weights.minimum > 0:
                    W_i = self.params.xbar_params.weights.minimum + W_i*Wrange_xbar

                self.core_slices[i].set_matrix(W_i, applyErrors=applyErrors)


            else:
                # Since weights have already been quantized, set inner core quantization to zero
                self.core_slices[i][0].params.algorithm_params.weight_bits = 0
                self.core_slices[i][1].params.algorithm_params.weight_bits = 0

                if self.params.numeric_params.Rp > 0 and self.params.xbar_params.noRpSlices[i]:
                    self.core_slices[i][0].params.numeric_params.Rp = 0
                    self.core_slices[i][1].params.numeric_params.Rp = 0

                # Set ADC range: we will only use the ADC of one of the cores, assuming subtraction is done in analog
                if self.core_slices[i][0].xbar_params.row_output.bits > 0:

                    if self.adc_range_option == "max" or self.adc_range_option == "calibrated":
                        self.core_slices[i][0].xbar_params.row_output.minimum = -self.W_shape[1]
                        self.core_slices[i][0].xbar_params.row_output.maximum = self.W_shape[1]
                        # Reduce the ADC range of both min and max by the same power of 2
                        if self.adc_range_option == "calibrated":
                            if self.Nbits_reduction[i] > 0:
                                self.core_slices[i][0].xbar_params.row_output.minimum /= pow(2,self.Nbits_reduction[i])
                                self.core_slices[i][0].xbar_params.row_output.maximum /= pow(2,self.Nbits_reduction[i])

                    elif self.adc_range_option == "granular":
                        Nbits_adc = self.core_slices[i].xbar_params.row_output.bits
                        Nbits_in = self.params.xbar_params.col_input.bits
                        ymin = 1/(pow(2,Wbits_slice)-1)
                        if signed_input:
                            corr = pow(2,Nbits_in-1)/(pow(2,Nbits_in-1)-1)
                        else:
                            corr = pow(2,Nbits_in)/(pow(2,Nbits_in)-1)
                        self.core_slices[i].xbar_params.row_output.minimum = -ymin * (pow(2,Nbits_adc-1)-1) * corr
                        self.core_slices[i].xbar_params.row_output.maximum = ymin * (pow(2,Nbits_adc-1)-1) * corr

                    self.core_slices[i][0].xbar_params.row_output.post_set()

                # Separate positive and negative
                W_i_pos = np.abs(W_i) * (W_i >= 0)
                W_i_neg = np.abs(W_i) * (W_i < 0)

                if self.params.xbar_params.weights.minimum > 0:
                    W_i_pos = self.params.xbar_params.weights.minimum + W_i_pos*Wrange_xbar
                    W_i_neg = self.params.xbar_params.weights.minimum + W_i_neg*Wrange_xbar

                self.core_slices[i][0].set_matrix(W_i_pos, applyErrors=applyErrors)
                self.core_slices[i][1].set_matrix(W_i_neg, applyErrors=applyErrors)

                if self.fast_balanced:
                    self.W_balanced[i] = self.core_slices[i][0]._read_matrix() - self.core_slices[i][1]._read_matrix()
                    # Get rid of the negative core object
                    self.core_slices[i][1] = None
                    # Delete the positive core matrix but keep the core to use its ADC limits
                    self.core_slices[i][0].matrix = None

        if self.params.xbar_params.profile_ADC_inputs:
            if self.params.convolution_parameters.is_conv_core:
                Nmvms = self.params.convolution_parameters.Nwindows
            else:
                Nmvms = 1
            Nmvms *= self.params.xbar_params.Nimages_bitslicing

            # Account for input bits
            if self.params.xbar_params.input_bitslicing:
                Nbits_in = self.params.xbar_params.col_input.bits
                if (self.params.xbar_params.col_input.minimum < 0):
                    Nbits_in -= 1
                self.bitslice_outputs = ncp.zeros((Nbits_in*self.W_shape[0],Nslices,Nmvms),dtype=ncp.float32)
            else:
                self.bitslice_outputs = ncp.zeros((self.W_shape[0],Nslices,Nmvms),dtype=ncp.float32)


    def _wrapper_set_vmm_inputs(self, vector):
        raise ValueError("Can only do MVMs (forward) in bit sliced core")

    def _wrapper_run_xbar_vmm(self, output_integrator_scaling):
        raise ValueError("Can only do MVMs (forward) in bit sliced core")

    def _wrapper_set_mvm_inputs(self, vector):
        for i in range(self.Nslices):
            if not self.balanced:
                self.core_slices[i].set_mvm_inputs(vector)
            else:
                self.core_slices[i][0].set_mvm_inputs(vector)
                if not self.fast_balanced:
                    self.core_slices[i][1].set_mvm_inputs(vector)


    def _wrapper_run_xbar_mvm(self, output_integrator_scaling):

        Wbits = self.params.algorithm_params.weight_bits
        Nslices = self.params.xbar_params.Nslices
        Wbits_slice = self.Wbits_slice

        # For final scaling
        Wrange_xbar = self.params.xbar_params.weights.maximum - self.params.xbar_params.weights.minimum
        Wmax_wrapper = self.wrapper_params.weights.maximum

        output_slices = [None for i in range(Nslices)]

        ##
        ##  NO INPUT BIT SLICING
        ##
        if not self.params.xbar_params.input_bitslicing:

            for i in range(Nslices):

                if not self.balanced:
                    output_i = self.core_slices[i].run_xbar_mvm()
                    if self.params.xbar_params.clip_Icol:
                        output_slices[i] = output_i.clip(-self.params.xbar_params.Icol_max,self.params.xbar_params.Icol_max)
                    else:
                        output_slices[i] = output_i

                else:
                    if self.fast_balanced:
                        output_slices[i] = ncp.dot(self.W_balanced[i],self.core_slices[i][0].vector_mvm)
                    else:
                        if not self.params.xbar_params.interleaved_posneg:
                            output_pos = self.core_slices[i][0].run_xbar_mvm()
                            output_neg = self.core_slices[i][1].run_xbar_mvm()

                            if self.params.xbar_params.clip_Icol:
                                output_pos = output_pos.clip(-self.params.xbar_params.Icol_max,self.params.xbar_params.Icol_max)
                                output_neg = output_neg.clip(-self.params.xbar_params.Icol_max,self.params.xbar_params.Icol_max)

                            output_slices[i] = output_pos - output_neg

                        else:
                            output_i = self.core_slices[i][0].run_xbar_mvm_interleaved(self.core_slices[i][1])
                            if self.params.xbar_params.clip_Icol:
                                output_slices[i] = output_i.clip(-self.params.xbar_params.Icol_max,self.params.xbar_params.Icol_max)


                if self.params.xbar_params.profile_ADC_inputs:
                    self.bitslice_outputs[:,i,self.i_mvm] = ncp.array(output_slices[i].copy().flatten(),dtype=ncp.float32)
        
        ##
        ##  INPUT BIT SLICING
        ##
        else:
            # Input bit slicing (bit serial)
            signed = (self.params.xbar_params.col_input.minimum < 0)

            # First, convert the inputs to integers from 0 to 2^n-1
            if not self.balanced:
                x = self.core_slices[0].vector_mvm
            else:
                x = self.core_slices[0][0].vector_mvm

            Nbits = self.params.xbar_params.col_input.bits
            if signed:
                x_mag = ncp.abs(x)
                x_sign = ncp.sign(x)
                Nbits -= 1
                x_int = x_mag / self.params.xbar_params.col_input.maximum
                x_int = ncp.rint(x_int * (pow(2,Nbits)-1))
            else:
                x_int = x / self.params.xbar_params.col_input.maximum
                x_int = ncp.rint(x_int * (pow(2,Nbits)-1))
            corr = pow(2,Nbits)/(pow(2,Nbits)-1)

            for k in range(Nbits):
                x_mvm = x_int % 2
                if signed:
                    x_mvm *= x_sign
                x_int = x_int // 2

                # Correction to make this equal to the non-bitsliced case
                x_mvm *= corr

                for i in range(Nslices):
                    if not self.balanced:
                        output_i_k = self.core_slices[i].run_xbar_mvm(vector=x_mvm)

                        if self.params.xbar_params.clip_Icol:
                            output_i_k = output_i_k.clip(-self.params.xbar_params.Icol_max,self.params.xbar_params.Icol_max)

                    else:
                        if self.fast_balanced:
                            output_i_k = ncp.dot(self.W_balanced[i],x_mvm)

                        else:
                            if not self.params.xbar_params.interleaved_posneg:
                                output_i_k_pos = self.core_slices[i][0].run_xbar_mvm(vector=x_mvm) 
                                output_i_k_neg = self.core_slices[i][1].run_xbar_mvm(vector=x_mvm)
                                if self.params.xbar_params.clip_Icol:
                                    output_i_k_pos = output_i_k_pos.clip(-self.params.xbar_params.Icol_max,self.params.xbar_params.Icol_max)
                                    output_i_k_neg = output_i_k_neg.clip(-self.params.xbar_params.Icol_max,self.params.xbar_params.Icol_max)
                                output_i_k = output_i_k_pos - output_i_k_neg

                            else:
                                output_i_k = self.core_slices[i][0].run_xbar_mvm_interleaved(self.core_slices[i][1],vector=x_mvm)
                                if self.params.xbar_params.clip_Icol:
                                    output_i_k = output_i_k.clip(-self.params.xbar_params.Icol_max,self.params.xbar_params.Icol_max)

                    if self.params.xbar_params.profile_ADC_inputs:
                        n_start = k*self.W_shape[0]
                        n_end = n_start + self.W_shape[0]
                        self.bitslice_outputs[n_start:n_end,i,self.i_mvm] = ncp.array(output_i_k.copy().flatten(),dtype=ncp.float32)

                    # ADC
                    if self.ADC_per_ibit:
                        if not self.balanced:
                            output_i_k = self.core_slices[i].xbar_params.row_output.clip_and_quantize(output_i_k)
                        else:
                            output_i_k = self.core_slices[i][0].xbar_params.row_output.clip_and_quantize(output_i_k)

                    if k == 0:
                        output_slices[i] = output_i_k.copy()
                    else:
                        output_slices[i] += output_i_k
                        
                    # Charge division / shift-and-add
                    output_slices[i] /= 2.0
            
            # # Scaling correction
            # for i in range(Nslices):
            #     output_slices[i] *= self.params.xbar_params.col_input.maximum * corr

        self.i_mvm += 1

        # clip and quantize result
        # bypass clip and quantize if profiling weight bit slices
        if output_integrator_scaling != 1:
            for i in range(Nslices):
                output_slices[i] *= output_integrator_scaling
                if not self.ADC_per_ibit:
                    if not self.balanced:
                        output_slices[i] = self.core_slices[i].xbar_params.row_output.clip_and_quantize(output_slices[i])
                    else:
                        output_slices[i] = self.core_slices[i][0].xbar_params.row_output.clip_and_quantize(output_slices[i])
                output_slices[i] /= output_integrator_scaling
        else:
            if not self.ADC_per_ibit:
                for i in range(Nslices):
                    if not self.balanced:
                        output_slices[i] = self.core_slices[i].xbar_params.row_output.clip_and_quantize(output_slices[i])
                    else:
                        output_slices[i] = self.core_slices[i][0].xbar_params.row_output.clip_and_quantize(output_slices[i])


        # Account for finite on-off ratio
        if self.params.xbar_params.weights.minimum > 0:
            if self.balanced:
                for i in range(Nslices):
                    output_slices[i] /= Wrange_xbar
            else:
                # If using offset, a Gmin offset has to be subtracted, and the offset is input dependent
                # THE CODE BELOW HAS NOT BEEN TESTED
                x = self.core_slices[0].vector_mvm
                if self.params.numeric_params.x_par > 1 or self.params.numeric_params.y_par > 1:
                    x_par = self.params.numeric_params.x_par
                    y_par = self.params.numeric_params.y_par
                    Noutputs = self.W_shape[0]
                    x_reshape = x.reshape((x_par*y_par,len(x)//(x_par*y_par)))
                    Gmin_bias = self.params.xbar_params.weights.minimum * np.sum(x_reshape,axis=1)
                    Gmin_bias = ncp.repeat(Gmin_bias,Noutputs)
                else:
                    Gmin_bias = self.params.xbar_params.weights.minimum * ncp.sum(x)
                for i in range(Nslices):
                    output_slices[i] = (output_slices[i] - Gmin_bias)/Wrange_xbar

        # Aggregate bit slices
        output = output_slices[0]
        for i in range(1,Nslices):
            output += pow(2,i*Wbits_slice)*output_slices[i]

        output *= pow(2,Wbits_slice)-1

        # Subtract bias offset (needed if not using balanced core)
        if not self.balanced:
            x = self.core_slices[0].vector_mvm

            if self.params.numeric_params.x_par > 1 or self.params.numeric_params.y_par > 1:
                x_par = self.params.numeric_params.x_par
                y_par = self.params.numeric_params.y_par
                Noutputs = self.W_shape[0]

                x_reshape = x.reshape((x_par*y_par,len(x)//(x_par*y_par)))

                offset = self.Woffset * np.sum(x_reshape,axis=1)
                offset = ncp.repeat(offset,Noutputs)
                output -= offset

            else:
                output -= self.Woffset*np.sum(x)

        output /= self.Wmax

        if self.params.xbar_params.weights.minimum > 0:
            output *= Wrange_xbar

        return output


    def _wrapper_update_matrix(self, row_vector, col_vector, learning_rate):
        raise ValueError("Weight updates not implemented for bit sliced core (inference use only).")


    def _wrapper__read_matrix(self):

        W = self.core_slices[0]._read_matrix()
        for i in range(1,Nslices):
            if not self.balanced:
                W += pow(2,i*Wbits_slice)*self.core_slices[0]._read_matrix()
            else:
                W += pow(2,i*Wbits_slice)*(self.core_slices[0]._read_matrix()-self.core_slices[1]._read_matrix())

        W *= pow(2,Wbits_slice)-1
        W -= self.Woffset
        W /= self.Wmax

        return output


    def _wrapper__save_matrix(self):

        W_list = np.concatenate((self.core_slices[0]._save_matrix(), self.core_slices[1]._save_matrix()))
        if self.Nslices > 2:
            for i in range(2,self.Nslices):
                if not self.balanced:
                    W_list = np.concatenate((W_list, self.core_slices[i]._save_matrix()))
                else:
                    W_list = np.concatenate((W_list, self.core_slices[i][0]._save_matrix()))
                    W_list = np.concatenate((W_list, self.core_slices[i][1]._save_matrix()))

        return W_list


    def _wrapper__restore_matrix(self, matrix):
        raise ValueError("Not implemented")


    def expand_matrix(self,Ncopy,mode=0):
        # Calls expand_matrix in the inner cores
        # Makes multiple copies of matrix to compute multiple MVMs in parallel
        for i in range(self.Nslices):
            if not self.balanced:
                self.core_slices[i].expand_matrix(Ncopy,mode=mode)
            else:
                if not self.fast_balanced:
                    self.core_slices[i][0].expand_matrix(Ncopy,mode=mode)
                    self.core_slices[i][1].expand_matrix(Ncopy,mode=mode)
                else:
                    if not self.params.numeric_params.weight_reorder:
                        Nx, Ny = self.W_balanced[i].shape
                        W_i_temp = self.W_balanced[i].copy()
                        self.W_balanced[i] = ncp.zeros((Ncopy*Nx,Ncopy*Ny),dtype=self.W_balanced[i].dtype)
                        for m in range(Ncopy):
                            x_start, x_end = m*Nx, (m+1)*Nx
                            y_start, y_end = m*Ny, (m+1)*Ny
                            self.W_balanced[i][x_start:x_end,y_start:y_end] = W_i_temp.copy()
                    else:
                        Kx = self.params.convolution_parameters.Kx
                        Ky = self.params.convolution_parameters.Ky
                        Nic = self.params.convolution_parameters.Nic
                        Noc = self.params.convolution_parameters.Noc
                        x_par = self.params.numeric_params.x_par # parallel windows in x
                        y_par = self.params.numeric_params.y_par # parallel windows in y
                        stride = self.params.convolution_parameters.stride
                        x_par_in = (x_par-1)*stride + Kx
                        y_par_in = (y_par-1)*stride + Ky
                        Nx, Ny = self.W_balanced[i].shape
                        W_temp_i = self.W_balanced[i].copy()
                        self.W_balanced[i] = ncp.zeros((x_par*y_par*Noc,x_par_in*y_par_in*Nic),dtype=self.W_balanced[i].dtype)
                        m = 0
                        for ix in range(x_par):
                            for iy in range(y_par):
                                for ixx in range(Kx):
                                    for iyy in range(Ky):
                                        W_start = ixx*Ky + iyy
                                        W_end = W_start + Nic*Kx*Ky
                                        W_vec = np.arange(W_start,W_end,Kx*Ky)
                                        x_coord = stride*ix + ixx
                                        y_coord = stride*iy + iyy
                                        x_start = x_coord*y_par_in + y_coord
                                        x_end = x_start + Nic*x_par_in*y_par_in
                                        x_vec = np.arange(x_start,x_end,x_par_in*y_par_in)
                                        y_start, y_end = m*Noc, (m+1)*Noc
                                        self.W_balanced[i][y_start:y_end,x_vec] = W_temp_i[:,W_vec].copy()
                                m += 1


    def unexpand_matrix(self,mode=0):
        # Calls unexpand_matrix in the inner cores
        for i in range(self.Nslices):
            if not self.balanced:
                self.core_slices[i].unexpand_matrix(mode=mode)
            else:
                if not self.fast_balanced:
                    self.core_slices[i][0].unexpand_matrix(mode=mode)
                    self.core_slices[i][1].unexpand_matrix(mode=mode)
                else:
                    self.W_balanced[i] = self.W_balanced[i][:self.W_shape[0],:self.W_shape[1]]
