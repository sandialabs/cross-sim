#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import threading

import numpy as np

from .debug import print_debug_calls
from .wrapper_core import WrapperCore
from ..parameters.parameter_defaults import UpdateModelEnum


class BalancedCore(WrapperCore):
    '''
    A balanced core consisting of two inner cores.
    
    One core is designated as "positive"; the other one as "negative". The actual value is the sum of the values of the inner cores.
    
    Both cores are started at the middle of their dynamic range. Requested updates are divided by two, and then performed on both cores,
        but the update performed on the negative core first has its sign flipped.
    '''
    
    def __init__(self, clipper_core_factory, params):
        """

        :param clipper_core_factory:
        :param params: all parameters
        :type params: Parameters
        :return:
        """

        WrapperCore.__init__(self, clipper_core_factory, params)

        self.core_pos = clipper_core_factory()
        self.core_neg = clipper_core_factory()

        global ncp
        if params.numeric_params.useGPU:
            global cp
            import cupy as cp
            cp.cuda.Device(params.numeric_params.gpu_id).use()
            ncp = cp
        else:
            ncp = np

        if params.xbar_params.input_bitslicing:
            if not params.algorithm_params.subtract_current_in_xbar:
                raise ValueError("If input bit slicing, must subtract current in the balanced crossbar")
            if self.params.xbar_params.col_input.bits == 0:
                raise ValueError("Input bit slicing must be disabled if input activations are not quantized")

        if self.params.xbar_params.interleaved_posneg and not self.algorithm_params.subtract_current_in_xbar:
            raise ValueError("Analog subtraction must be enabled for posneg interleaved array")

        if self.params.xbar_params.interleaved_posneg and self.params.numeric_params.Rp > 0:
            if self.params.numeric_params.circuit.Vselect > 0:
                raise ValueError("Not implemented: interleaved balanced core with select device")
            if not self.params.numeric_params.circuit.noRowParasitics:
                raise ValueError("Not implemented: interleaved balanced core with row parasitics")

        self.fast_balanced = params.xbar_params.fast_balanced        
        if self.fast_balanced and self.params.numeric_params.read_noise.sigma > 0 and self.params.weight_error_params.noise_model != "none":
            print("Fast balanced core option cannot be used with read noise: reverting.")
            self.fast_balanced = False
        if self.fast_balanced and self.params.numeric_params.Rp > 0:
            print("Fast balanced core option cannot be used with parasitic resistance: reverting.")
            self.fast_balanced = False
        if self.fast_balanced and not self.algorithm_params.subtract_current_in_xbar:
            print("Fast balanced core option must be used with analog subtraction: reverting.")
            self.fast_balanced = False
        if self.fast_balanced and self.params.xbar_params.interleaved_posneg:
            print("Fast balanced core option cannot be used with interleaving: reverting.")
            self.fast_balanced = False
        if self.fast_balanced and self.params.xbar_params.clip_Icol:
            print("Fast balanced core option cannot be used with current limiter: reverting.")
            self.fast_balanced = False

        self.ADC_per_ibit = self.params.xbar_params.ADC_per_ibit
        if self.params.xbar_params.ADC_per_ibit and not self.params.xbar_params.input_bitslicing:
            print("ADC per input bit enabled but input bit slicing disabled. Disabling ADC per input bit.")
            self.ADC_per_ibit = False

        self.adc_range_option = self.params.xbar_params.adc_range_option
        if self.adc_range_option not in ("calibrated","max","granular"):
            raise ValueError("Invalid ADC range option for non-bitsliced balanced core")
        if self.adc_range_option == "granular" and not self.ADC_per_ibit:
            raise ValueError("Granular ADC range option can currently only be used with digital input S&A.")

        self.i_mvm = 0


    def _given_inner_update_wrapper_limits(self):
        inner_constraints = self.params.xbar_params
        wrapper_constraints = self.params.wrapper_params

        #copy over inner constraints
        inner_constraints.copy_clip_constaints_to(wrapper_constraints)

        if inner_constraints.weights.minimum != 0:
            raise ValueError('BalancedCore cannot handle inner core with minimum value != 0.')
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



    def _wrapper_nonlinear_update_scaling(self, matrix, update):
        """
        Returns what the the update should be given a desired update to account for nonlinearity.
        It is used to pass the average nonlinearity all the way to the top level cores.

        :param matrix:  return the nonlinearity (update scaling) at each value in matrix
        :return:
        """
        # shift matrix (can skip bias row step)
        matrix -= (self.wrapper_params.weights.minimum - self.core_pos.xbar_params.weights.minimum)
        matrix_neg = (self.core_pos.xbar_params.weights.minimum + self.core_pos.xbar_params.weights.maximum)- matrix

        # compute nonlinear update
        matrix = self.core_pos.nonlinear_update_scaling(matrix, update=update)
        matrix_neg = self.core_neg.nonlinear_update_scaling(matrix_neg, update=-update)

        # combine results

        matrix -= matrix_neg
        matrix /= 2.0
        return matrix



    def _wrapper_set_matrix(self, matrix, applyErrors=True):
        print_debug_calls('BalancedCore.set_matrix',matrix)

        # Save the matrix shape
        self.W_shape = matrix.shape

        if self.core_pos.xbar_params.balanced_style == "one_sided":
            Wrange_xbar = self.core_pos.xbar_params.weights.maximum - self.core_pos.xbar_params.weights.minimum
            Wmax_wrapper = self.wrapper_params.weights.maximum
            if self.params.xbar_params.weights.bits > 0:
                Wmin_res = 2**(-(self.params.xbar_params.weights.bits+1))
            else:
                Wmin_res = 0

            if (np.abs(self.wrapper_params.weights.minimum) - Wmax_wrapper) > 1e-3:
                raise ValueError("Wrapper weight range not symmetric")

            mat_pos = self.core_pos.xbar_params.weights.minimum * (matrix < -Wmin_res) + \
                (self.core_pos.xbar_params.weights.minimum + Wrange_xbar * matrix/Wmax_wrapper) * (matrix >= Wmin_res)
            mat_neg = self.core_pos.xbar_params.weights.minimum * (matrix >= Wmin_res) + \
                (self.core_pos.xbar_params.weights.minimum - Wrange_xbar * matrix/Wmax_wrapper) * (matrix < -Wmin_res)

            mat_pos = mat_pos.astype(ncp.float32)
            mat_neg = mat_neg.astype(ncp.float32)

            self.core_pos.set_matrix(mat_pos, applyErrors=applyErrors)
            self.core_neg.set_matrix(mat_neg, applyErrors=applyErrors)

        else:
            matrix -= (self.wrapper_params.weights.minimum - self.core_pos.xbar_params.weights.minimum)
            self.core_pos.set_matrix(matrix, applyErrors=applyErrors)
            self.core_neg.set_matrix( (self.core_pos.xbar_params.weights.minimum + self.core_pos.xbar_params.weights.maximum)- matrix, applyErrors=applyErrors)

        if self.fast_balanced:
            self.W_balanced = self.core_pos._read_matrix() - self.core_neg._read_matrix()
            self.core_neg = None
            # Delete the positive core matrix but keep the core to use its ADC limits
            self.core_pos.matrix = None

        # ADC range options
        if self.core_pos.xbar_params.row_output.bits > 0:

            # The default option is "calibrated" with ADC_per_ibit = False
            # In this case the xbar_params.row_output limits passed in from inference_net are used
            # The cases below override these limits
            signed_input = (self.params.xbar_params.col_input.minimum < 0)

            # Set ADC limits to maximum possible
            if self.adc_range_option == "max":
                if self.algorithm_params.subtract_current_in_xbar:
                    self.core_pos.xbar_params.row_output.minimum = -self.W_shape[1]
                    self.core_pos.xbar_params.row_output.maximum = self.W_shape[1]
                else:
                    if signed_input:
                        self.core_pos.xbar_params.row_output.minimum = -self.W_shape[1]
                        self.core_neg.xbar_params.row_output.minimum = -self.W_shape[1]
                    else:
                        self.core_pos.xbar_params.row_output.minimum = 0
                        self.core_neg.xbar_params.row_output.minimum = 0
                    self.core_pos.xbar_params.row_output.maximum = self.W_shape[1]
                    self.core_neg.xbar_params.row_output.maximum = self.W_shape[1]

            elif self.adc_range_option == "granular":
                # Set ADC limits according to resolution with a fixed level separation
                # This case will seldom be used with differential cells
                Nbits_adc = self.params.xbar_params.row_output.bits
                Nbits_in = self.params.xbar_params.col_input.bits
                Nbits_w = self.params.algorithm_params.weight_bits
                ymin = 1/pow(2,Nbits_w)
                if signed_input:
                    corr = pow(2,Nbits_in-1)/(pow(2,Nbits_in-1)-1)
                else:
                    corr = pow(2,Nbits_in)/(pow(2,Nbits_in)-1)
                    
                if self.algorithm_params.subtract_current_in_xbar:
                    self.core_pos.xbar_params.row_output.minimum = -ymin * (pow(2,Nbits_adc-1)-1) * corr
                    self.core_pos.xbar_params.row_output.maximum = ymin * (pow(2,Nbits_adc-1)-1) * corr
                else:
                    if signed_input:
                        self.core_pos.xbar_params.row_output.minimum = -ymin * (pow(2,Nbits_adc-1)-1) * corr
                        self.core_pos.xbar_params.row_output.maximum = ymin * (pow(2,Nbits_adc-1)-1) * corr
                        self.core_neg.xbar_params.row_output.minimum = -ymin * (pow(2,Nbits_adc-1)-1) * corr
                        self.core_neg.xbar_params.row_output.maximum = ymin * (pow(2,Nbits_adc-1)-1) * corr
                    else:
                        self.core_pos.xbar_params.row_output.minimum = 0
                        self.core_pos.xbar_params.row_output.maximum = ymin * (pow(2,Nbits_adc)-1) * corr
                        self.core_neg.xbar_params.row_output.minimum = 0
                        self.core_neg.xbar_params.row_output.maximum = ymin * (pow(2,Nbits_adc)-1) * corr

            elif self.adc_range_option == "calibrated" and self.ADC_per_ibit:
                # This case will seldom be used: corresponds to differential cells with digital input bit accumulation
                self.core_pos.xbar_params.row_output.minimum = self.params.xbar_params.adc_range_internal[0]
                self.core_pos.xbar_params.row_output.maximum = self.params.xbar_params.adc_range_internal[1]
                if not self.algorithm_params.subtract_current_in_xbar:
                    self.core_neg.xbar_params.row_output.minimum = self.params.xbar_params.adc_range_internal[0]
                    self.core_neg.xbar_params.row_output.maximum = self.params.xbar_params.adc_range_internal[1]

        # If profiling bit slicing, initialize data structure here now that matrix dimensions are known
        # profile_ADC_inputs is ignored if not input bit slicing, since it is done inside backprop.py
        if self.params.xbar_params.input_bitslicing and self.params.xbar_params.profile_ADC_inputs:
            Nbits = self.params.xbar_params.col_input.bits
            if self.params.xbar_params.col_input.minimum < 0:
                Nbits -= 1
            # This is to ensure accurate binning of column currents to specific MVMs
            if self.params.numeric_params.x_par > 1 or self.params.numeric_params.y_par > 1:
                raise ValueError("If profiling bit slicing currents, must use x_par, y_par = 1")
            # Assume that if profiling bit slicing, total # images = 1
            Ncol = matrix.shape[0]
            if self.params.convolution_parameters.is_conv_core:
                Nmvms = self.params.convolution_parameters.Nwindows
            else:
                Nmvms = 1

            # Currently hard coding the number of images to profile
            Nmvms *= self.params.xbar_params.Nimages_bitslicing

            # ADC inputs are profiled after subtraction
            # If profiling currents before subtraction, use the commented out lines
            self.array_outputs = ncp.zeros((Ncol,Nbits,Nmvms),dtype=ncp.float32)


    def _wrapper_set_vmm_inputs(self, vector):
        print_debug_calls('BalancedCore.set_vmm_inputs',vector)
        self.core_pos.set_vmm_inputs(vector)
        self.core_neg.set_vmm_inputs(vector)


    def _wrapper_set_mvm_inputs(self, vector):
        print_debug_calls('BalancedCore.set_mvm_inputs',vector)
        self.core_pos.set_mvm_inputs(vector)
        if not self.fast_balanced:
            self.core_neg.set_mvm_inputs(vector)
    
    
    def _wrapper_run_xbar_vmm(self, output_integrator_scaling):
        print_debug_calls('BalancedCore.run_xbar_vmm')
        # run positive and negative cores in parallel
        if self.core_pos.__class__.__name__ == 'XyceCore':
            results = [None,None]
            #define functions for each thread
            def pos():
                results[0] = self.core_pos.run_xbar_vmm()
            def neg():
                results[1] = self.core_neg.run_xbar_vmm()
            #define threads
            threads = (threading.Thread(target=pos,name='pos'),
                       threading.Thread(target=neg,name='neg'))
            #start threads
            for thread in threads:
                thread.start()
            # add locks for threads to finish
            for thread in threads:
                thread.join()
            output_pos = results[0]
            output_neg = results[1]
        else:
            output_pos = self.core_pos.run_xbar_vmm()
            output_neg = self.core_neg.run_xbar_vmm()


        #check to see if clipping should be done before or after subtraction
        #TODO:  Implement this in Xyce
        if self.algorithm_params.subtract_current_in_xbar is True:
            output = output_pos - output_neg
            # clip and quantize result
            if output_integrator_scaling!=1:
                output*=output_integrator_scaling
                output = self.core_pos.xbar_params.col_output.clip_and_quantize(output)
                output/=output_integrator_scaling
            else:
                output = self.core_pos.xbar_params.col_output.clip_and_quantize(output)

        else:
            # clip and quantize result
            if output_integrator_scaling!=1:
                output_pos*=output_integrator_scaling
                output_neg*=output_integrator_scaling
                output_pos = self.core_pos.xbar_params.col_output.clip_and_quantize(output_pos)
                output_neg = self.core_neg.xbar_params.col_output.clip_and_quantize(output_neg)
                output_pos/=output_integrator_scaling
                output_neg/=output_integrator_scaling
            else:
                output_pos = self.core_pos.xbar_params.col_output.clip_and_quantize(output_pos)
                output_neg = self.core_neg.xbar_params.col_output.clip_and_quantize(output_neg)
            output = output_pos - output_neg

        output /= 2.0
        return output


    def _wrapper_run_xbar_mvm(self, output_integrator_scaling):
        print_debug_calls('BalancedCore.run_xbar_mvm')

        if self.core_pos.__class__.__name__ == 'XyceCore':
            results = [None,None]
            def pos():
                results[0] = self.core_pos.run_xbar_mvm()
            def neg():
                results[1] = self.core_neg.run_xbar_mvm()
            threads = (threading.Thread(target=pos,name='pos'),
                       threading.Thread(target=neg,name='neg'))
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            output_pos = results[0]
            output_neg = results[1]
        else:
            ##
            ##  NO INPUT BIT SLICING
            ##
            if not self.params.xbar_params.input_bitslicing:

                if not self.params.xbar_params.interleaved_posneg:
                    if self.fast_balanced:
                        output = ncp.dot(self.W_balanced,self.core_pos.vector_mvm)
                    else:
                        output_pos = self.core_pos.run_xbar_mvm()
                        output_neg = self.core_neg.run_xbar_mvm()

                        if self.params.xbar_params.clip_Icol:
                            output_pos = output_pos.clip(-self.params.xbar_params.Icol_max,self.params.xbar_params.Icol_max)
                            output_neg = output_neg.clip(-self.params.xbar_params.Icol_max,self.params.xbar_params.Icol_max)

                        if self.algorithm_params.subtract_current_in_xbar:
                            output = output_pos - output_neg

                else:
                    output = self.core_pos.run_xbar_mvm_interleaved(self.core_neg)
                    if self.params.xbar_params.clip_Icol:
                        output = output.clip(-self.params.xbar_params.Icol_max,self.params.xbar_params.Icol_max)

            
            ##
            ##  INPUT BIT SLICING
            ##
            else:
                # Input bit slicing (bit serial)
                signed = (self.params.xbar_params.col_input.minimum < 0)

                # First, convert the inputs to integers from 0 to 2^n-1
                x = self.core_pos.vector_mvm
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

                    if not self.params.xbar_params.interleaved_posneg:
                        if self.fast_balanced:
                            output_bal = ncp.dot(self.W_balanced,x_mvm)
                        else:
                            output_pos = self.core_pos.run_xbar_mvm(vector=x_mvm)
                            output_neg = self.core_neg.run_xbar_mvm(vector=x_mvm)

                            # Clip the accumulated current on each column
                            if self.params.xbar_params.clip_Icol:
                                output_pos = output_pos.clip(-self.params.xbar_params.Icol_max,self.params.xbar_params.Icol_max)
                                output_neg = output_neg.clip(-self.params.xbar_params.Icol_max,self.params.xbar_params.Icol_max)

                            output_bal = output_pos - output_neg

                    else:
                        output_bal = self.core_pos.run_xbar_mvm_interleaved(self.core_neg,vector=x_mvm)

                        if self.params.xbar_params.clip_Icol:
                            output_bal = output_bal.clip(-self.params.xbar_params.Icol_max,self.params.xbar_params.Icol_max)

                    # Scaling correction
                    output_bal *= self.params.xbar_params.col_input.maximum * corr

                    # Profiling of bit sliced array outputs
                    if self.params.xbar_params.profile_ADC_inputs:
                        output_bal_f = ncp.array(output_bal.flatten(),dtype=ncp.float32)
                        self.array_outputs[:,k,self.i_mvm] = output_bal_f

                    # ADC
                    if self.ADC_per_ibit:
                        output_bal = self.core_pos.xbar_params.row_output.clip_and_quantize(output_bal)

                    if k == 0:
                        output = output_bal.copy()
                    else:
                        output += output_bal

                    # Charge division or shift right
                    # Charge leakage from the integrating cap can be modeled here
                    output /= 2.0

        self.i_mvm += 1

        if self.core_pos.xbar_params.balanced_style == "one_sided":
            Wrange_xbar = self.core_pos.xbar_params.weights.maximum - self.core_pos.xbar_params.weights.minimum
            Wmax_wrapper = self.wrapper_params.weights.maximum


        if self.algorithm_params.subtract_current_in_xbar:

            # ADC
            if not self.ADC_per_ibit:
                if output_integrator_scaling!=1:
                    output*=output_integrator_scaling
                    output = self.core_pos.xbar_params.row_output.clip_and_quantize(output)
                    output/=output_integrator_scaling
                else:
                    output = self.core_pos.xbar_params.row_output.clip_and_quantize(output)

            if self.core_pos.xbar_params.balanced_style == "one_sided":
                output /= (Wrange_xbar/Wmax_wrapper)
            else:
                output /= 2.0
        else:
            # clip and quantize result
            if output_integrator_scaling!=1:
                output_pos*=output_integrator_scaling
                output_neg*=output_integrator_scaling
                if not self.ADC_per_ibit:
                    output_pos = self.core_pos.xbar_params.row_output.clip_and_quantize(output_pos)
                    output_neg = self.core_neg.xbar_params.row_output.clip_and_quantize(output_neg)
                output_pos/=output_integrator_scaling
                output_neg/=output_integrator_scaling
            else:
                if not self.ADC_per_ibit:
                    output_pos = self.core_pos.xbar_params.row_output.clip_and_quantize(output_pos)
                    output_neg = self.core_neg.xbar_params.row_output.clip_and_quantize(output_neg)

            output = output_pos - output_neg
            if self.core_pos.xbar_params.balanced_style == "one_sided":
                output /= (Wrange_xbar/Wmax_wrapper)
            else:
                output /= 2.0

        return output


    def _wrapper_update_matrix(self, row_vector, col_vector, learning_rate):
        print_debug_calls('BalancedCore.update_matrix')
        if self.core_pos.__class__.__name__ == 'XyceCore':
            def pos():
                self.core_pos.update_matrix(row_vector, col_vector, learning_rate)
            def neg():
                self.core_neg.update_matrix(row_vector, -col_vector, learning_rate)
            threads = (threading.Thread(target=pos,name='pos'),
                       threading.Thread(target=neg,name='neg'))
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            if self.core_pos.record_updates and self.core_neg.record_updates:
                # Pass in the same random value to both cores so they record updates simultaneously
                randRecord = np.random.rand(1)
                self.core_pos.update_matrix(row_vector, col_vector, learning_rate,1,randRecord=randRecord)
                self.core_neg.update_matrix(row_vector, -col_vector, learning_rate,0,randRecord=randRecord)

            else:
                self.core_pos.update_matrix(row_vector, col_vector, learning_rate,1)
                self.core_neg.update_matrix(row_vector, -col_vector, learning_rate,0)


    def _wrapper_update_matrix_burst(self, update_matrix, learning_rate):
        print_debug_calls('BalancedCore.update_matrix')
        if self.core_pos.__class__.__name__ == 'XyceCore':
            def pos():
                self.core_pos.update_matrix_burst(update_matrix, learning_rate)
            def neg():
                self.core_neg.update_matrix_burst(-update_matrix, learning_rate)
            threads = (threading.Thread(target=pos,name='pos'),
                       threading.Thread(target=neg,name='neg'))
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            if self.core_pos.record_updates and self.core_neg.record_updates:
                randRecord = np.random.rand(1)
            else:
                randRecord = 1e20
                
            update_pos = update_matrix
            update_neg = -update_matrix

            # Pass in the same random value to both cores so they record updates simultaneously
            self.core_pos.update_matrix_burst(update_pos, learning_rate,1,randRecord=randRecord)
            self.core_neg.update_matrix_burst(update_neg, learning_rate,0,randRecord=randRecord)



    def _wrapper__read_matrix(self):
        print_debug_calls('BalancedCore._read_matrix',end=' ')
        output = self.core_pos._read_matrix() - self.core_neg._read_matrix()

        if self.core_pos.xbar_params.balanced_style == "one_sided":
            Wrange_xbar = self.core_pos.xbar_params.weights.maximum - self.core_pos.xbar_params.weights.minimum
            Wmax_wrapper = self.wrapper_params.weights.maximum
            output /= (Wrange_xbar/Wmax_wrapper)
        else:
            output /= 2.0
        print_debug_calls(output)
        return output


    def _wrapper__save_matrix(self):
        print_debug_calls('BalancedCore._save_matrix')
        return np.concatenate(self.core_pos._save_matrix(), self.core_neg._save_matrix())


    def _wrapper__restore_matrix(self, matrix):
        print_debug_calls('BalancedCore._restore_matrix')
        matrix = np.split(matrix, 2)
        self.core_pos._restore_matrix(matrix[0])
        self.core_neg._restore_matrix(matrix[1])


    def get_update_record(self):
        if self.core_pos.record_updates and self.core_neg.record_updates:
            # inner core updates
            #target_updates = np.concatenate((self.core_pos.target_updates,self.core_neg.target_updates))
            #real_updates = np.concatenate((self.core_pos.real_updates,self.core_neg.real_updates))

            # outer core updates
            target_updates = (self.core_pos.target_updates-self.core_neg.target_updates)/2
            real_updates = (self.core_pos.real_updates-self.core_neg.real_updates)/2
            return target_updates, real_updates
        else:
            error("Attempting to retrieve update information that was never recorded")


    def expand_matrix(self,Ncopy,mode=0):
        # Calls expand_matrix in the inner cores
        # Makes multiple copies of matrix to compute multiple MVMs in parallel
        if not self.fast_balanced:
            self.core_pos.expand_matrix(Ncopy,mode=mode)
            self.core_neg.expand_matrix(Ncopy,mode=mode)
        else:
            if not self.params.numeric_params.weight_reorder:
                Nx, Ny = self.W_balanced.shape
                W_temp = self.W_balanced.copy()
                self.W_shape = self.W_balanced.shape
                self.W_balanced = ncp.zeros((Ncopy*Nx,Ncopy*Ny),dtype=self.W_balanced.dtype)
                for m in range(Ncopy):
                    x_start, x_end = m*Nx, (m+1)*Nx
                    y_start, y_end = m*Ny, (m+1)*Ny
                    self.W_balanced[x_start:x_end,y_start:y_end] = W_temp.copy()

            else:
                Kx = self.params.convolution_parameters.Kx
                Ky = self.params.convolution_parameters.Ky
                Nic = self.params.convolution_parameters.Nic
                Noc = self.params.convolution_parameters.Noc
                stride = self.params.convolution_parameters.stride
                x_par = self.params.numeric_params.x_par # parallel windows in x
                y_par = self.params.numeric_params.y_par # parallel windows in y
                x_par_in = (x_par-1)*stride + Kx
                y_par_in = (y_par-1)*stride + Ky
                Nx, Ny = self.W_balanced.shape
                W_temp = self.W_balanced.copy()
                self.W_balanced = ncp.zeros((x_par*y_par*Noc,x_par_in*y_par_in*Nic),dtype=self.W_balanced.dtype)
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
                                self.W_balanced[y_start:y_end,x_vec] = W_temp[:,W_vec].copy()
                        m += 1


    def unexpand_matrix(self,mode=0):
        if not self.fast_balanced:
            # Calls unexpand_matrix in the inner cores
            self.core_pos.unexpand_matrix(mode=mode)
            self.core_neg.unexpand_matrix(mode=mode)
        else:
            self.W_balanced = self.W_balanced[:self.W_shape[0],:self.W_shape[1]]