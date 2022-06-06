#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np

from .debug import print_debug_calls
from .wrapper_core import WrapperCore


class OffsetCore(WrapperCore):
    '''
    An offset core consisting of a single inner core.
    
    The middle value in the dynamic range of the inner core is used to store :code:`0.0`.
    
    An additional row and column are used to also store the middle value, which is needed to fully implement the offset.
    '''
    
    def __init__(self, clipper_core_factory, params):
        """

        :param clipper_core_factory:
        :param params:
        :type params: Parameters
        :return:
        """

        WrapperCore.__init__(self, clipper_core_factory, params)

        self.core = clipper_core_factory()
        """:type: ClipperCore"""

        #do current subtraction prior to clipping
        if params.algorithm_params.subtract_current_in_xbar is True:
            self.core.subtract_current_in_offset_xbar = True
        else:
            #tell the first neuron along a row/col should saturate at half the value of the others
            self.core.rescale_offset_neuron = True

        global ncp
        if params.numeric_params.useGPU:
            import cupy as cp
            cp.cuda.Device(params.numeric_params.gpu_id).use()
            ncp = cp
        else:
            ncp = np

        # Whether to subtract an offset value in digital or in analog
        self.inference_mode = self.params.xbar_params.offset_inference
        self.digital_offset = self.params.xbar_params.digital_offset
        self.ADC_per_ibit = self.params.xbar_params.ADC_per_ibit

        if self.inference_mode:
            self.core.subtract_current_in_offset_xbar = False

        if self.params.xbar_params.ADC_per_ibit and not self.params.xbar_params.input_bitslicing:
            print("ADC per input bit enabled but input bit slicing disabled. Disabling ADC per input bit.")
            self.ADC_per_ibit = False

        self.adc_range_option = self.params.xbar_params.adc_range_option
        if self.adc_range_option not in ("calibrated","max","granular"):
            raise ValueError("Invalid ADC range option for non-bitsliced balanced core")
        if self.adc_range_option == "granular" and not self.ADC_per_ibit:
            raise ValueError("Granular ADC range option can currently only be used with digital input S&A.")

        if self.params.numeric_params.weight_reorder:
            raise ValueError("Toeplitz reordering is not compatible with offset core")

        self.i_mvm = 0


    def _given_inner_update_wrapper_limits(self):
        """
        Uses the inner constraints and updates wrapper_constraints based on the current core type
        :return:
        """
        inner_constraints = self.params.xbar_params
        wrapper_constraints = self.params.wrapper_params
        outer_constraints = self.params.algorithm_params

        #copy over inner constraints
        inner_constraints.copy_clip_constaints_to(wrapper_constraints)


        if outer_constraints.weights.minimum==0:
            wrapper_constraints.weights.minimum = 0
            wrapper_constraints.weights.maximum = inner_constraints.weights.maximum - inner_constraints.weights.minimum

        else:
            # copy inner coefficient limits to wrapper and center around zero
            icc_c_m = inner_constraints.weights.middle
            wrapper_constraints.weights.minimum -= icc_c_m
            wrapper_constraints.weights.maximum -= icc_c_m

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

        # for offset core copy wrapper output limits
        inner_cc.row_output.maximum = wrapper_cc.row_output.maximum
        inner_cc.row_output.minimum = wrapper_cc.row_output.minimum

        inner_cc.col_output.maximum = wrapper_cc.col_output.maximum
        inner_cc.col_output.minimum = wrapper_cc.col_output.minimum

        # copy the row/col update limits
        inner_cc.row_update.maximum = wrapper_cc.row_update.maximum
        inner_cc.row_update.minimum = wrapper_cc.row_update.minimum

        inner_cc.col_update.maximum = wrapper_cc.col_update.maximum
        inner_cc.col_update.minimum = wrapper_cc.col_update.minimum

        return inner_cc




    def _wrapper_nonlinear_update_scaling(self, matrix, update):
        """
        Returns what the the update should be given a desired update to account for nonlinearity.
        It is used to pass the average nonlinearity all the way to the top level cores.

        :param matrix:  return the nonlinearity (update scaling) at each value in matrix for an update of 'update'
        :return:
        """
        # shift matrix (can skip bias row step)
        matrix -= (self.wrapper_params.weights.minimum - self.core.xbar_params.weights.minimum)
        # compute nonlinear update
        matrix = self.core.nonlinear_update_scaling(matrix, update)
        # don't unshift matrix (these are delta weights, not absolute weights)
        return matrix


    def _wrapper_set_matrix(self, matrix, applyErrors=True):
        print_debug_calls('OffsetCore.set_matrix',matrix)

        if not self.inference_mode:
            new_matrix = np.insert(np.insert(matrix, 0, 0, axis=0), 0, 0, axis=1)
            new_matrix -= (self.wrapper_params.weights.minimum - self.core.xbar_params.weights.minimum)
            self.core.set_matrix(new_matrix, applyErrors=applyErrors)

        else:
            if not self.digital_offset:
                # Figure out what wrapper value corresponds to a unit algorithm value
                new_matrix = np.insert(matrix, 0, 0, axis=0)
                new_matrix -= (self.wrapper_params.weights.minimum - self.core.xbar_params.weights.minimum)
                self.W_shape = new_matrix.shape
                self.core.set_matrix(new_matrix, applyErrors=applyErrors)

            else:
                new_matrix = matrix.copy()
                new_matrix -= (self.wrapper_params.weights.minimum - self.core.xbar_params.weights.minimum)
                self.W_shape = new_matrix.shape
                self.core.set_matrix(new_matrix, applyErrors=applyErrors)

            # ADC range options
            if self.core.xbar_params.row_output.bits > 0:

                # The default option is "calibrated" with ADC_per_ibit = False
                # In this case the xbar_params.row_output limits passed in from inference_net are used
                # The cases below override these limits
                signed_input = (self.params.xbar_params.col_input.minimum < 0)

                # Set ADC limits to maximum possible
                if self.adc_range_option == "max":
                    if signed_input:
                        self.core.xbar_params.row_output.minimum = -self.W_shape[1]
                    else:
                        self.core.xbar_params.row_output.minimum = 0
                    self.core.xbar_params.row_output.maximum = self.W_shape[1]

                elif self.adc_range_option == "granular":
                    # Set ADC limits according to resolution with a fixed level separation
                    # This case will seldom be used with differential cells
                    Nbits_adc = self.core.xbar_params.row_output.bits
                    Nbits_in = self.params.xbar_params.col_input.bits
                    Nbits_w = self.params.algorithm_params.weight_bits
                    ymin = 1/pow(2,Nbits_w)

                    if signed_input:
                        corr = pow(2,Nbits_in-1)/(pow(2,Nbits_in-1)-1)
                        self.core.xbar_params.row_output.minimum = -ymin * (pow(2,Nbits_adc-1)-1) * corr
                        self.core.xbar_params.row_output.maximum = ymin * (pow(2,Nbits_adc-1)-1) * corr
                    else:
                        corr = pow(2,Nbits_in)/(pow(2,Nbits_in)-1)
                        self.core.xbar_params.row_output.minimum = 0
                        self.core.xbar_params.row_output.maximum = ymin * (pow(2,Nbits_adc)-1) * corr

                elif self.adc_range_option == "calibrated" and self.ADC_per_ibit:
                    # This case will seldom be used: corresponds to differential cells with digital input bit accumulation
                    self.core.xbar_params.row_output.minimum = self.params.xbar_params.adc_range_internal[0]
                    self.core.xbar_params.row_output.maximum = self.params.xbar_params.adc_range_internal[1]

            # Profiling of ADC inputs
            # profile_ADC_inputs is ignored if not input bit slicing, since it is done inside backprop.py
            if self.params.xbar_params.profile_ADC_inputs and self.params.xbar_params.input_bitslicing:
                if self.params.convolution_parameters.is_conv_core:
                    Nmvms = self.params.convolution_parameters.Nwindows
                else:
                    Nmvms = 1
                if self.params.numeric_params.x_par > 1 or self.params.numeric_params.y_par > 1:
                    raise ValueError("If profiling bit slicing currents, must use x_par, y_par = 1")
                Nmvms *= self.params.xbar_params.Nimages_bitslicing
                Nbits_in = self.params.xbar_params.col_input.bits
                if (self.params.xbar_params.col_input.minimum < 0):
                    Nbits_in -= 1
                self.array_outputs = ncp.zeros((self.W_shape[0],Nbits_in,Nmvms),dtype=ncp.float32)


    def _wrapper_set_vmm_inputs(self, vector):
        print_debug_calls('OffsetCore.set_vmm_inputs',vector)
        return self.core.set_vmm_inputs(np.insert(vector, 0, 0))


    def _wrapper_set_mvm_inputs(self, vector):
        print_debug_calls('OffsetCore.set_mvm_inputs',vector)
        if not self.inference_mode:
            return self.core.set_mvm_inputs(np.insert(vector, 0, 0))
        else:
            return self.core.set_mvm_inputs(vector)
    
    
    def __process_output(self, output):
        print_debug_calls('OffsetCore.__process_output(',output,')')

        if self.core.subtract_current_in_offset_xbar is False:
            output = output[1:] - output[0]
        else:
            output = output[1:]
        print_debug_calls('  OffsetCore.__process_output returning', output)
        return output


    def _wrapper_run_xbar_vmm(self, output_integrator_scaling):
        print_debug_calls('OffsetCore.run_xbar_vmm')
        #run internal core vmm
        result = self.core.run_xbar_vmm()

        # if offset core is used double the first neuron output before clipping and then halve it
        if self.core.rescale_offset_neuron is True:
            result[0]*=2
        # clip and quantize result

        if output_integrator_scaling !=1:  #if the output of the integrator is scaled, scale result before clipping and quantizing and then unscale
            result*=output_integrator_scaling
            result = self.core.xbar_params.col_output.clip_and_quantize(result)
            result/=output_integrator_scaling
        else:
            result = self.core.xbar_params.col_output.clip_and_quantize(result)

        if self.core.rescale_offset_neuron is True:
            result[0]/=2


        #convert from internal core output to wrapper output
        return self.__process_output(result)


    def _wrapper_run_xbar_mvm(self, output_integrator_scaling):
        print_debug_calls('OffsetCore.run_xbar_mvm')

        if not self.inference_mode:

            result = self.core.run_xbar_mvm()

            # if offset core is used double the first neuron output before clipping and then halve it
            if self.core.rescale_offset_neuron is True:
                result[0] *= 2
            # clip and quantize result
            if output_integrator_scaling !=1:
                result*=output_integrator_scaling
                result = self.core.xbar_params.row_output.clip_and_quantize(result)
                result/=output_integrator_scaling
            else:
                result = self.core.xbar_params.row_output.clip_and_quantize(result)
            if self.core.rescale_offset_neuron is True:
                result[0]/=2

            return self.__process_output(result)


        else:

            ##
            ##  NO INPUT BIT SLICING
            ##
            if not self.params.xbar_params.input_bitslicing:
                output = self.core.run_xbar_mvm()
                if self.params.xbar_params.clip_Icol:
                    output = output.clip(-self.params.xbar_params.Icol_max,self.params.xbar_params.Icol_max)
            
            ##
            ##  INPUT BIT SLICING
            ##
            else:
                # Input bit slicing (bit serial)
                signed = (self.params.xbar_params.col_input.minimum < 0)
                # First, convert the inputs to integers from 0 to 2^n-1
                x = self.core.vector_mvm
                # Nbits = self.params.xbar_params.col_input.bits.astype(int)
                Nbits = self.params.xbar_params.col_input.bits
                if signed:
                    x_mag = ncp.abs(x)
                    x_sign = ncp.sign(x)
                    Nbits -= 1
                    x_int = x_mag / self.params.xbar_params.col_input.maximum
                    x_int = ncp.rint(x_int * (pow(2,Nbits)-1))
                    corr = pow(2,Nbits)/(pow(2,Nbits)-1)
                else:
                    x_int = x / self.params.xbar_params.col_input.maximum
                    x_int = ncp.rint(x_int * (pow(2,Nbits)-1))
                    corr = pow(2,Nbits)/(pow(2,Nbits)-1)

                for k in range(Nbits):
                    x_mvm = x_int % 2
                    if signed:
                        x_mvm *= x_sign
                    x_int = x_int // 2

                    output_k = self.core.run_xbar_mvm(vector=x_mvm)

                    # Clip the accumulated current on each column
                    # This clipping is done by the analog integrator rather than by the ADC
                    if self.params.xbar_params.clip_Icol:
                        output_k = output_k.clip(-self.params.xbar_params.Icol_max,self.params.xbar_params.Icol_max)

                    # Scaling correction
                    output_k *= self.params.xbar_params.col_input.maximum * corr

                    if self.params.xbar_params.profile_ADC_inputs:
                        self.array_outputs[:,k,self.i_mvm] = ncp.array(output_k.copy().flatten(),dtype=ncp.float32)

                    # ADC
                    if self.ADC_per_ibit:
                        output_k = self.core.xbar_params.row_output.clip_and_quantize(output_k)

                    if k == 0:
                        output = output_k.copy()
                    else:
                        output += output_k

                    # Charge division or shift right
                    output /= 2.0

            self.i_mvm += 1

            ##### Quantize and subtract offset

            # clip and quantize result
            if not self.ADC_per_ibit:
                if output_integrator_scaling !=1:
                    output*=output_integrator_scaling
                    output = self.core.xbar_params.row_output.clip_and_quantize(output)
                    output/=output_integrator_scaling
                else:
                    output = self.core.xbar_params.row_output.clip_and_quantize(output)

            if self.digital_offset:
                # Subtract bias offset
                x = self.core.vector_mvm
                if self.params.numeric_params.x_par > 1 or self.params.numeric_params.y_par > 1:
                    x_par = self.params.numeric_params.x_par
                    y_par = self.params.numeric_params.y_par
                    Noutputs = self.W_shape[0]
                    x_reshape = x.reshape((x_par*y_par,len(x)//(x_par*y_par)))
                    offset = 0.5*np.sum(x_reshape,axis=1)
                    offset = ncp.repeat(offset,Noutputs)
                    output -= offset
                else:
                    output -= 0.5*np.sum(x)

            else:
                if self.params.numeric_params.x_par > 1 or self.params.numeric_params.y_par > 1:
                    x_par = self.params.numeric_params.x_par
                    y_par = self.params.numeric_params.y_par
                    output = output.reshape((x_par*y_par,len(output)//(x_par*y_par)))
                    for m in range(x_par*y_par):
                        output[m,1:] = output[m,1:] - output[m,0]
                    output = output[:,1:].flatten()
                else:
                    output = output[1:]

            return output


    def _wrapper_update_matrix(self, row_vector, col_vector, learning_rate):
        print_debug_calls('OffsetCore.update_matrix')

        # TODO: Why are 0s added to the vectors?
        row_vector = np.insert(row_vector, 0, 0)
        col_vector = np.insert(col_vector, 0, 0)
        print_debug_calls('   row_vector:',row_vector)
        print_debug_calls('   col_vector:',col_vector)
        core_ind = 1
        return self.core.update_matrix(row_vector, col_vector, learning_rate,core_ind)


    def _wrapper_update_matrix_burst(self, update_matrix, learning_rate):
        print_debug_calls('OffsetCore.update_matrix')

        # TODO: Why are 0s added to the vectors?
        update_matrix = np.insert(update_matrix,0,0,axis=0)
        update_matrix = np.insert(update_matrix,0,0,axis=1)
        print_debug_calls('   update_matrix:',update_matrix)
        core_ind = 1
        return self.core.update_matrix_burst(update_matrix, learning_rate,core_ind)


    def _wrapper__read_matrix(self):
        print_debug_calls('OffsetCore._read_matrix',end=' ')
        output = self.core._read_matrix()[1:,1:]
        output = output.copy()
        output += (self.wrapper_params.weights.minimum - self.core.xbar_params.weights.minimum)
        print_debug_calls(output)
        return output


    def _wrapper__save_matrix(self):
        print_debug_calls('OffsetCore._save_matrix')
        output = self.core._save_matrix()
        return output.copy()


    def _wrapper__restore_matrix(self, matrix):
        print_debug_calls('OffsetCore._restore_matrix')
        return self.core._restore_matrix(matrix)


    def get_update_record(self):
        if self.core.record_updates:
            return self.core.target_updates, self.core.real_updates
        else:
            raise ModuleNotFoundError ("Attempting to retrieve update information that was never recorded")

    def expand_matrix(self,Ncopy,mode=0):
        # Calls expand_matrix in the inner cores
        # Makes multiple copies of matrix to compute multiple MVMs in parallel        
        self.core.expand_matrix(Ncopy,mode=mode)


    def unexpand_matrix(self,mode=0):
        # Calls unexpand_matrix in the inner cores
        self.core.unexpand_matrix(mode=mode)
