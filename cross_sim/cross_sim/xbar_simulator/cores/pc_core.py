#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#


from .icore import ICore
from ..parameters import Parameters
import numpy as np
from ..cores import  WrapperCore
from warnings import warn
from . import MakeCore2
from ..parameters.parameter_defaults import ZeroResetPCEnum, SimTypeEnum

class PeriodicCarryCore(ICore):
    '''
    Implements a periodic carry using multiple crossbars
    '''


    def __init__(self, params):
        """
        Carry updates are applied entirely on the row drivers (one col at a time).


        :param params: All parameters
        :type params: Parameters
        :return:
        """

        self.params = params
        self.pc_params = params.periodic_carry_params
        self.n_cores = params.periodic_carry_params.cores_per_weight


        self.cores = []  # stores all the wrapper cores for the periodic carry.  Core 0 is the highest order bit
        ''':type: list of WrapperCore'''

        if params.periodic_carry_params.cores_per_weight<2:
            raise ValueError("Must have more than 1 periodic carry core")

        # adjust all scaling limits for each core
        for core_num in range(self.n_cores):
            params = self.params.copy()
            if core_num !=0:  # adjust weight ranges and output ranges for all but highest order bit
                scaling =1 / (self.pc_params.number_base**core_num)
                params.algorithm_params.weights.maximum = self.params.algorithm_params.weights.maximum*scaling
                params.algorithm_params.weights.minimum = self.params.algorithm_params.weights.minimum*scaling



                params.algorithm_params.col_output.maximum = self.params.algorithm_params.row_input.maximum*\
                                                             self.params.algorithm_params.weights.maximum*self.pc_params.normalized_output_scale
                params.algorithm_params.col_output.minimum = -params.algorithm_params.col_output.maximum

                params.algorithm_params.row_output.maximum = self.params.algorithm_params.col_input.maximum*\
                                                             self.params.algorithm_params.weights.maximum*self.pc_params.normalized_output_scale
                params.algorithm_params.row_output.minimum = -params.algorithm_params.row_output.maximum


            # change update range for all cores except lowest order bit (scale to weight range of next lower order bit):
            if core_num!=params.periodic_carry_params.cores_per_weight-1:

                # max out col driver
                params.algorithm_params.col_update.maximum = 1
                params.algorithm_params.col_update.minimum = -1

                params.algorithm_params.row_update_portion=1 # apply updates on rows

                if params.periodic_carry_params.min_carry_update is None:
                    weight_max = self.params.algorithm_params.weights.maximum / (self.pc_params.number_base**(core_num+1)) # weight max of 1 bit lower order core

                    if params.algorithm_params.sim_type==SimTypeEnum.NUMERIC:
                        middle = params.xbar_params.weights.middle
                        update_overshoot= max((params.xbar_params.weight_clipping.maximum-middle) / (params.xbar_params.weights.maximum-middle),
                                              (middle-params.xbar_params.weight_clipping.minimum) / (middle-params.xbar_params.weights.minimum))
                        update_overshoot=max(update_overshoot,2.0)
                    else:
                        update_overshoot=2
                        warn("For periodic carry it is assumed that the weights will not exceed more than twice the intended range")

                    # print("update_overshoot=",update_overshoot)
                    params.algorithm_params.row_update.maximum = weight_max*update_overshoot
                    params.algorithm_params.row_update.minimum = -weight_max*update_overshoot
                else:
                    # compute row update to give a fixed minimum update and set number of bits accordingly
                    bits = np.ceil(np.log2(1/params.periodic_carry_params.min_carry_update) )
                    # print("bits=",bits)
                    params.xbar_params.col_update.bits=1
                    params.xbar_params.col_update.sign_bit=True
                    params.xbar_params.row_update.bits=bits-1
                    params.xbar_params.row_update.sign_bit=True

                    dX = params.algorithm_params.row_update.range
                    W_range = 2*params.algorithm_params.weights.absmaxmin
                    row_levels = params.xbar_params.row_update.levels - 1

                    dX_new = W_range*row_levels*params.periodic_carry_params.min_carry_update

                    # update dX and dY
                    params.algorithm_params.row_update.maximum = dX_new/2
                    params.algorithm_params.row_update.minimum = -dX_new/2

                    # print("row_update_max",params.algorithm_params.row_update.maximum,"weight_max",params.algorithm_params.weights.maximum)
                    # print("row_update_min",params.algorithm_params.row_update.minimum,"weight_min",params.algorithm_params.weights.minimum)


            else:
                # check that max update is not larger than the weights
                if params.algorithm_params.row_update.absmaxmin*params.algorithm_params.col_update.absmaxmin>params.algorithm_params.weights.range:
                    warn("The maximum update is larger than the weight range of lowest order bit, updates will be clipped")


            print("core {0} has a weight range of {1} to {2}".format(core_num,params.algorithm_params.weights.maximum,params.algorithm_params.weights.minimum))
            self.cores.append(MakeCore2(params))

        self.update_ctrs = np.zeros(len(self.cores))
        self.update_count = [0] # count of current update for plotting weight updates, use mutable list that pointers in a core can be updated to point to
        for core in self.cores:  # update pointer in all cores to the update_count list
            core.external_update_counter=self.update_count

        global ncp
        if self.params.numeric_params.useGPU:
            import cupy as cp
            ncp = cp
        else:
            ncp = np



    def nonlinear_update_scaling(self, matrix, update, core =0):
        """
        Returns what the the update should be given a desired update to account for nonlinearity.
        It is used to pass the average nonlinearity all the way to the top level cores.

        return results on highest order bit (unless otherwise specified) that has original weight ranges

        :param matrix:  return the nonlinearity (update scaling) at each value in matrix for an update of 'update'
        :param core: which core # to return the nonlinear scaling for
        :return:
        """
        return self.cores[core].nonlinear_update_scaling(matrix,update)


    def set_matrix(self, matrix, applyErrors=False):
        '''
        Set the matrix on only the highest order bit
        '''

        for ind in range(len(self.cores)):
            if ind ==0:
                self.cores[ind].set_matrix(matrix,applyErrors=applyErrors)
            else:
                self.cores[ind].set_matrix(np.zeros_like(matrix),applyErrors=applyErrors)  # TODO: add option to set random intitial values


    def set_vmm_inputs(self, vector):
        '''
        Set the vector to use on either only highest order core or all cores depending on read_low_order_bits
        '''
        if self.pc_params.read_low_order_bits == True:
            for core in self.cores:
                result = core.set_vmm_inputs(vector)
            return result
        else:
            return self.cores[0].set_vmm_inputs(vector)

    def set_mvm_inputs(self, vector):
        '''
        Set the vector to use on either only highest order core or all cores depending on read_low_order_bits
        '''
        if self.pc_params.read_low_order_bits == True:
            for core in self.cores:
                result=core.set_mvm_inputs(vector)
            return result
        else:
            return self.cores[0].set_mvm_inputs(vector)
    
    def run_xbar_vmm(self,vector=None):
        '''
        run VMM on all cores or highest order bit only
        '''

        if vector is not None:
            self.set_vmm_inputs(vector)

        result = self.cores[0].run_xbar_vmm()

        # add result from lower order bits if needed (all scaling is already done)
        if self.pc_params.read_low_order_bits == True:
            for core in self.cores[1:]:
                result+= core.run_xbar_vmm()

        return result



    def run_xbar_mvm(self, vector=None):
        '''
        run MVM on all cores or highest order bit only
        '''

        if vector is not None:
            self.set_mvm_inputs(vector)


        result = self.cores[0].run_xbar_mvm()

        # add result from lower order bits if needed (all scaling is already done)
        if self.pc_params.read_low_order_bits == True:
            for core in self.cores[1:]:
                result+= core.run_xbar_mvm()

        return result


    def update_matrix(self, row_vector, col_vector, learning_rate=1):
        r'''
        Updates the matrix given input row and column vectors.
        assumes serial updates are done by col (one col at a time)

        '''

        self.update_ctrs[self.n_cores-1]+=1
        self.update_count[0]+=1  # keep track of the current update count

        self.cores[self.n_cores-1].update_matrix(row_vector,col_vector,learning_rate)


        for ind in range(self.n_cores-1,0,-1): # goes from max to core 1, don't do carry on core zero

            if self.update_ctrs[ind]>=self.pc_params.carry_frequency[ind-1]:  # ind-1 as there is no carry freq for highest order bit
                #reset counter
                self.update_ctrs[ind]=0
                # read matrix serially
                matrix = self.cores[ind]._read_matrix() #Debug
                # matrix = self.cores[ind].serial_read()
                matrix1 = matrix.copy()

                #determine weights larger than threshold:
                threshold  = self.cores[ind].params.algorithm_params.weights.maximum*self.pc_params.carry_threshold

                dont_update = ncp.logical_and( (-threshold<matrix), (matrix<threshold) )
                matrix[dont_update]=0

                if self.pc_params.zero_reset == ZeroResetPCEnum.EXACT:#if resetting zero exactly
                    self.cores[ind].set_matrix(matrix1-matrix)
                else:
                    # # need to adjust learning rate to allow for large subtractions
                    # algorithm_params = self.cores[ind].params.algorithm_params
                    #
                    # if (algorithm_params.row_update.absmaxmin * algorithm_params.col_update.absmaxmin) < algorithm_params.weights.absmaxmin:  #TODO: can clip update if weights/updates have huge odd nonlinearity (adjust based on weight scaling) currently using range is 2X larger than needed without nonlinearity
                    #     internal_learning_rate = algorithm_params.weights.absmaxmin / (algorithm_params.row_update.absmaxmin * algorithm_params.col_update.absmaxmin)
                    # else:
                    #     internal_learning_rate = 1

                    # if self.pc_params.zero_reset == ZeroResetPCEnum.CALIBRATED:
                    #     update = self.nonlinear_update_scaling(matrix,-matrix,core=ind)
                    # else:
                    #     update = -matrix

                    # # self.cores[ind].serial_update(update/internal_learning_rate, internal_learning_rate, by_row=False)  # subtract values from current matrix
                    # self.cores[ind].serial_update(update, 1, by_row=False)  # subtract values from current matrix

                    desired_matrix = matrix1 - matrix
                    current_matrix = self.cores[ind]._read_matrix()
                    update_matrix = desired_matrix - current_matrix
                    self.cores[ind].update_matrix_burst(update_matrix,learning_rate=1)

                    # #DEBUG:#################
                    # ideal = matrix1-matrix
                    # actual = self.cores[ind]._read_matrix()

                if self.pc_params.exact_carries: # if computing the carry exactly
                    self.cores[ind-1].set_matrix(self.cores[ind-1]._read_matrix()+matrix)
                else:
                    # self.cores[ind-1].serial_update(matrix, by_row=False) # add values to the next core
                    self.cores[ind-1].update_matrix_burst(matrix,learning_rate=1)

                self.update_ctrs[ind-1]+=1
            else:
                break

    def update_matrix_burst(self, matrix0, learning_rate=1):
        r'''
        Updates the matrix given input row and column vectors.
        assumes serial updates are done by col (one col at a time)

        '''

        self.update_ctrs[self.n_cores-1]+=1
        self.update_count[0]+=1  # keep track of the current update count

        self.cores[self.n_cores-1].update_matrix_burst(matrix0,learning_rate=learning_rate)

        for ind in range(self.n_cores-1,0,-1): # goes from max to core 1, don't do carry on core zero

            if self.update_ctrs[ind]>=self.pc_params.carry_frequency[ind-1]:  # ind-1 as there is no carry freq for highest order bit
                #reset counter
                self.update_ctrs[ind]=0
                # read matrix serially
                matrix = self.cores[ind]._read_matrix() #Debug
                # matrix = self.cores[ind].serial_read()
                matrix1 = matrix.copy()

                #determine weights larger than threshold:
                threshold  = self.cores[ind].params.algorithm_params.weights.maximum*self.pc_params.carry_threshold

                dont_update = ncp.logical_and( (-threshold<matrix), (matrix<threshold) )
                matrix[dont_update]=0

                if self.pc_params.zero_reset == ZeroResetPCEnum.EXACT:#if resetting zero exactly
                    self.cores[ind].set_matrix(matrix1-matrix)
                else:
                    desired_matrix = matrix1 - matrix
                    current_matrix = self.cores[ind]._read_matrix()
                    update_matrix = desired_matrix - current_matrix
                    self.cores[ind].update_matrix_burst(update_matrix,learning_rate=1)

                if self.pc_params.exact_carries: # if computing the carry exactly
                    self.cores[ind-1].set_matrix(self.cores[ind-1]._read_matrix()+matrix)
                else:
                    # self.cores[ind-1].serial_update(matrix, by_row=False) # add values to the next core
                    self.cores[ind-1].update_matrix_burst(matrix,learning_rate=1)

                self.update_ctrs[ind-1]+=1
            else:
                break


    def _read_matrix(self):
        '''
        Read the internal matrix held by this core (debug method)
        The combined matrix is returned (i.e. the weights are summed across crossbars)
        No quantization or other errors are applied.
        '''

        result = self.cores[0]._read_matrix()

        # add result from lower order bits if needed (all scaling is already done)
        if self.pc_params.read_low_order_bits == True:
            for core in self.cores[1:]:
                result+= core._read_matrix()

        return result


    def _save_matrix(self):
        '''
        Save the internal matrix held by this core (debug method)
        Unlike _read_matrix, all data necessary to restore the matrix is provided.
        No quantization or other errors are applied.
        '''

        # combine all the matricies into a single large matrix
        return np.concatenate( *[core._save_matrix() for core in self.cores])


    def _restore_matrix(self, matrix):
        '''
        Restore an internal matrix held by this core (debug method)
        You should only use a matrix obtained from _save_matrix, as _read_matrix may remove needed values (e.g.: from an offset core).
        No quantization or other errors are applied.
        '''


        # split the concatenated matrix into individual matricies for each core
        matrix = np.split(matrix, self.pc_params.cores_per_weight)
        for ind in range(len(self.cores) ):
            self.cores[ind]._restore_matrix(matrix[ind])