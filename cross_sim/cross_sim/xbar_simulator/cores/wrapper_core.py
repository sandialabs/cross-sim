#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#


from abc import abstractmethod, ABCMeta

import numpy as np

from . import ICore
from .debug import print_debug_calls
from ..parameters import Parameters

class WrapperCore(ICore, metaclass=ABCMeta):
    '''
    Superclass for "wrapper" cores -- such as OffsetCore, PosNegCore, and BalancedCore.
    
    Instances of WrapperCore are always the outermost (algorithm-facing) ICore.
    
    The subclass must implement the _wrapper_* methods.
    '''

    def _given_wrapper_update_outer_limits(self):
        """
        Given wrapper clip constraints (inner clip constraints modified by the wrapper, e.g.: to make the coefficient
        limits not just be positive), and required outer clip constraints,
        creates the additional outer constraints, which get used for scaling

        :type params: Parameters
        """

        # save local copies of wrapper and outer clip constraints
        wrapper_cc = self.wrapper_params
        """:type: WrapperParams """
        outer_cc = self.params.algorithm_params
        """:type: AlgorithmParams """


        # scale factors to convert a number in wrapper scale to outer scale
        scale_matrix = outer_cc.weights.range / wrapper_cc.weights.range
        scale_row_in = outer_cc.row_input.range / wrapper_cc.row_input.range
        scale_col_in = outer_cc.col_input.range / wrapper_cc.col_input.range

        # outer outputs = wrapper output * undo matrix scaling * undo input scaling (col input for row output)
        outer_cc.row_output = wrapper_cc.row_output * scale_matrix * scale_col_in
        outer_cc.col_output = wrapper_cc.col_output * scale_matrix * scale_row_in



        # find the update scaling
        # use absmaxmin rather than range to find scaling factors for row/col_update
        # the max update is determined by the product of absmaxmin rather than range

        # find the max fraction of the weight range that can be changed in a single update
        update_fraction = wrapper_cc.row_update.absmaxmin * wrapper_cc.col_update.absmaxmin / wrapper_cc.weights.range


        # multiply by the outer coeficicient range to convert to the max outer update
        max_outer_update = update_fraction * outer_cc.weights.range


        # divide the updates between the row and column according to row_update_portion
        outer_cc.row_update.maximum = max_outer_update ** outer_cc.row_update_portion
        outer_cc.row_update.minimum = -(max_outer_update ** outer_cc.row_update_portion)

        outer_cc.col_update.maximum = max_outer_update ** (1 - outer_cc.row_update_portion)
        outer_cc.col_update.minimum = -(max_outer_update ** (1 - outer_cc.row_update_portion))

        return outer_cc

    def _given_outer_update_wrapper_limits(self):
        """
        Given outer clip constraints create wrapper constraints, which get used for scaling

        """

        # save local names of wrapper and outer clip constraints
        wrapper_cc = self.params.wrapper_params
        outer_cc = self.params.algorithm_params
        inner_cc = self.params.xbar_params

        if outer_cc.weights.minimum == 0:
            wrapper_cc.weights.minimum = 0
            wrapper_cc.weights.maximum = inner_cc.weights.maximum - inner_cc.weights.minimum

        else:
            # copy inner coefficient limits to wrapper and center around zero
            icc_c_m = inner_cc.weights.middle

            wrapper_cc.weights.minimum = inner_cc.weights.minimum - icc_c_m
            wrapper_cc.weights.maximum = inner_cc.weights.maximum - icc_c_m

        # copy inner row/col input limits to wrapper
        wrapper_cc.row_input.maximum = inner_cc.row_input.maximum
        wrapper_cc.row_input.minimum = inner_cc.row_input.minimum

        wrapper_cc.col_input.maximum = inner_cc.col_input.maximum
        wrapper_cc.col_input.minimum = inner_cc.col_input.minimum


        # scale factors to convert a number in wrapper scale to outer scale
        scale_matrix = outer_cc.weights.range / wrapper_cc.weights.range
        scale_row_in = outer_cc.row_input.range / wrapper_cc.row_input.range
        scale_col_in = outer_cc.col_input.range / wrapper_cc.col_input.range

        # wrapper outputs = outer output / matrix scaling / input scaling (col input for row output)
        wrapper_cc.row_output = outer_cc.row_output / (scale_matrix * scale_col_in)
        wrapper_cc.col_output = outer_cc.col_output / (scale_matrix * scale_row_in)

        # find the update scaling
        # use absmaxmin rather than range to find scaling factors for row/col_update
        # the max update is determined by the product of absmaxmin rather than range

        # find the max fraction of the weight range that can be changed in a single update
        update_fraction = outer_cc.row_update.absmaxmin * outer_cc.col_update.absmaxmin / outer_cc.weights.range

        # multiply by the wrapper coeficicient range to convert to the max wrapper update
        max_wrapper_update = update_fraction * wrapper_cc.weights.range
        # divide the updates between the row and column according to row_update_portion
        wrapper_cc.row_update.maximum = max_wrapper_update ** outer_cc.row_update_portion
        wrapper_cc.row_update.minimum = -(max_wrapper_update ** outer_cc.row_update_portion)
        wrapper_cc.col_update.maximum = max_wrapper_update ** (1 - outer_cc.row_update_portion)
        wrapper_cc.col_update.minimum = -(max_wrapper_update ** (1 - outer_cc.row_update_portion))

        return wrapper_cc

    def __init__(self, clipper_core_factory, params):
        """

        :param clipper_core_factory:
        :param params:
        :type params: Parameters
        :return:
        """

        self.params = params
        ''':type: Parameters'''

        self.clipper_core_factory = clipper_core_factory

        # choose whether to use outer to compute inner or inner to compute outer limits
        if self.params.algorithm_params.calculate_inner_from_outer is True:
            # computer wrapper constraints given outer
            self.params.wrapper_params = self._given_outer_update_wrapper_limits()

            # compute inner given wrapper  (shared parameter object automatically updates inner core params)
            self.params.xbar_params = self._given_wrapper_update_inner_limits()

        else:
            # computes the wrapper clip constraints based on the innper clip constraints and wrapper_core type
            self.params.wrapper_params = self._given_inner_update_wrapper_limits()  # equivalent to self.params.wrapper_params

            # update outer clip constraints to match inner constraints (using the just defined wrapper clip constraints)
            self.params.algorithm_params = self._given_wrapper_update_outer_limits()  # equivalent to self.params.algorithm_params

        # save reference to core for shorter access
        self.wrapper_params = self.params.wrapper_params
        self.algorithm_params = self.params.algorithm_params

        # save empty lists for analytics if analytics are turned on
        if self.params.analytics_params.store_weights ==True:
            self.weights_list = [] #list to store weights
            self.update_ctr = 0
            # store external update count for periodic carry
            self.external_update_counter=[0] # use a list so that this can be externally changed to point to an external list
            self.external_update_counter_list =[]

        if self.params.analytics_params.store_update_inputs:
            self.update_rows_list = []
            self.update_cols_list = []
            self.update_ctr = 0

        if self.params.analytics_params.store_row_inputs:
            self.row_inputs = []
            self.input_output_ctr = 0

        if self.params.analytics_params.store_col_inputs:
            self.col_inputs = []
            self.input_output_ctr = 0

        if self.params.analytics_params.store_row_outputs:
            self.row_outputs = []
            self.input_output_ctr = 0

        if self.params.analytics_params.store_col_outputs:
            self.col_outputs = []
            self.input_output_ctr = 0

        global ncp
        if self.params.numeric_params.useGPU:
            import cupy as cp
            ncp = cp
        else:
            ncp = np


    def nonlinear_update_scaling(self, matrix, update,row_vector,col_vector):
        """
        Returns what the the update should be given a desired update to account for nonlinearity.
        It is used to pass the average nonlinearity all the way to the top level cores.

        :param matrix:  return the nonlinearity (update scaling) at each value in matrix for an update of 'update'
        :return:
        """
        #scale to wrapper parameters
        matrix = self.algorithm_params.weights.scale_to(self.wrapper_params.weights, matrix)
        update = self.algorithm_params.weights.scale_to(self.wrapper_params.weights, update)
        # compute nonlinear update
        matrix  = self._wrapper_nonlinear_update_scaling(matrix, update,row_vector,col_vector)

        # unscale result
        return(self.algorithm_params.weights.scale_from(self.wrapper_params.weights,matrix) )


    def set_matrix(self, matrix, applyErrors=True):
        print_debug_calls('WrapperCore.set_matrix', matrix)

        self.nrows = matrix.shape[0]
        self.ncols = matrix.shape[1]

        if self.params.analytics_params.store_weights:
            if self.params.analytics_params.max_storage_cycles == 0 or self.params.analytics_params.max_storage_cycles>self.update_ctr:
                weights = matrix
                if self.params.analytics_params.all_weights == True:
                    self.weights_list.append(weights)
                else:
                    self.weights_list.append(weights[self.params.analytics_params.weight_rows, self.params.analytics_params.weight_cols] )
                self.external_update_counter_list.append(self.external_update_counter[0]) #store external weight index for periodic carry

            self.update_ctr+=1

        return self._wrapper_set_matrix(
            self.algorithm_params.weights.scale_to(self.wrapper_params.weights, matrix), applyErrors=applyErrors)

    def set_vmm_inputs(self, vector):
        print_debug_calls('WrapperCore.set_vmm_inputs', vector)

        if self.params.numeric_params.useGPU:
            import cupy as cp
            # assume vector already cupy
            vector = cp.copy(vector)

        return self._wrapper_set_vmm_inputs(
            self.algorithm_params.row_input.scale_to(self.wrapper_params.row_input, vector))

    def set_mvm_inputs(self, vector):
        print_debug_calls('WrapperCore.set_mvm_inputs', vector)

        if self.params.numeric_params.useGPU:
            # assume vector already cupy
            vector = ncp.copy(vector)

        if not self.params.convolution_parameters.is_conv_core:
            return self._wrapper_set_mvm_inputs(
                self.algorithm_params.col_input.scale_to(self.wrapper_params.col_input, vector))
        else:
            return self._wrapper_set_mvm_inputs(vector)

    def run_xbar_vmm(self, vector=None, output_integrator_scaling=1):
        """

        :param vector:
        :param output_integrator_scaling: the output integrator is scaled by this number (i.e. the integrator capacitance is changed)
        :return:
        """
        print_debug_calls('WrapperCore.run_xbar_vmm')

        if self.params.analytics_params.store_row_inputs:
            if self.params.analytics_params.max_storage_cycles == 0 or self.params.analytics_params.max_storage_cycles > self.input_output_ctr:
                self.row_inputs.append(vector)

        #set the vector if an argument is passed
        if vector is not None:
            self.set_vmm_inputs(vector)

        # run the vmm
        output = self._wrapper_run_xbar_vmm(output_integrator_scaling)
        # outputs come along columns, scale from wrapper constraints to outer constraints
        output = self.algorithm_params.col_output.scale_from(self.wrapper_params.col_output, output)

        if self.params.analytics_params.store_col_outputs:
            if self.params.analytics_params.max_storage_cycles == 0 or self.params.analytics_params.max_storage_cycles > self.input_output_ctr:
                self.col_outputs.append(output)

        if self.params.analytics_params.store_col_outputs or self.params.analytics_params.store_row_inputs: self.input_output_ctr += 1

        return output

    def run_xbar_mvm(self, vector=None, output_integrator_scaling=1):

        if self.params.analytics_params.store_col_inputs:
            if self.params.analytics_params.max_storage_cycles == 0 or self.params.analytics_params.max_storage_cycles > self.input_output_ctr:
                self.col_inputs.append(vector)

        #set the vector if an argument is passed
        if vector is not None:
            self.set_mvm_inputs(vector)

        # run MVM
        output = self._wrapper_run_xbar_mvm(output_integrator_scaling)

        # outputs come along rows, scale from wrapper constraints to outer constraints
        output = self.algorithm_params.row_output.scale_from(self.wrapper_params.row_output, output)

        if self.params.analytics_params.store_row_outputs:
            if self.params.analytics_params.max_storage_cycles == 0 or self.params.analytics_params.max_storage_cycles > self.input_output_ctr:
                self.row_outputs.append(output)

        if self.params.analytics_params.store_row_outputs or self.params.analytics_params.store_col_inputs: self.input_output_ctr +=1

        return output

    def update_matrix(self, row_vector, col_vector, learning_rate =1, can_overwrite_row=False, can_overwrite_col=False, dont_store_weights=False):
        """

        :param row_vector:
        :param col_vector:
        :param can_overwrite_row:
        :param can_overwrite_col:
        :param learning_rate: input the learning rate separately as the timed driver can be scaled independently of the quantization
        :param dont_store_weights:   don't store weights even if weight storage debugging is enabled (bypass for serial updates so only stored after last update)
        :return:
        """
        print_debug_calls('WrapperCore.update_matrix', row_vector, col_vector)

        # eliminated COWArray  #using COWArray caused extra writes, slowed down ~10%
        if not can_overwrite_row:
            row_vector = ncp.array(row_vector, dtype="float32")  #create a numpy copy
        if not can_overwrite_col:
            col_vector = ncp.array(col_vector, dtype="float32")  #create a numpy copy

        if self.params.analytics_params.store_update_inputs:
            if self.params.analytics_params.max_storage_cycles == 0 or self.params.analytics_params.max_storage_cycles>self.update_ctr:
                if not self.params.analytics_params.no_update_rows:
                    self.update_rows_list.append(row_vector.copy())
                if not self.params.analytics_params.no_update_cols:
                    self.update_cols_list.append(col_vector.copy())

        # scale row and col update vectors to the wrapper update vector scale

        row_vector = self.algorithm_params.row_update.scale_to(self.wrapper_params.row_update,
                                                           row_vector)
        col_vector = self.algorithm_params.col_update.scale_to(self.wrapper_params.col_update,
                                                           col_vector)

        result = self._wrapper_update_matrix(row_vector, col_vector, learning_rate)

        # print("weights = ",self._read_matrix())

        # store weights if needed
        if self.params.analytics_params.store_weights:
            if not dont_store_weights:  #don't store values during each serial update
                if self.params.analytics_params.max_storage_cycles == 0 or self.params.analytics_params.max_storage_cycles>self.update_ctr:
                    weights =self._read_matrix()
                    if self.params.analytics_params.all_weights == True:
                        self.weights_list.append(weights)
                    else:
                        self.weights_list.append(weights[self.params.analytics_params.weight_rows, self.params.analytics_params.weight_cols] )
                    self.external_update_counter_list.append(self.external_update_counter[0]) #store external weight index for periodic carry

        if self.params.analytics_params.store_update_inputs or self.params.analytics_params.store_weights: self.update_ctr+=1



        return result


    def update_matrix_burst(self, update_matrix, learning_rate =1, can_overwrite_mat=False, dont_store_weights=False):
        """

        :param update_matrix:
        :param can_overwrite_row:
        :param can_overwrite_col:
        :param learning_rate: input the learning rate separately as the timed driver can be scaled independently of the quantization
        :param dont_store_weights:   don't store weights even if weight storage debugging is enabled (bypass for serial updates so only stored after last update)
        :return:
        """
        print_debug_calls('WrapperCore.update_matrix', update_matrix)

        # eliminated COWArray  #using COWArray caused extra writes, slowed down ~10%
        # if not can_overwrite_mat:
            # update_matrix = np.array(update_matrix, dtype="float32")  #create a numpy copy

        # if self.params.analytics_params.store_update_inputs:
        #     if self.params.analytics_params.max_storage_cycles == 0 or self.params.analytics_params.max_storage_cycles>self.update_ctr:
        #         if not self.params.analytics_params.no_update_rows:
        #             self.update_rows_list.append(row_vector.copy())
        #         if not self.params.analytics_params.no_update_cols:
        #             self.update_cols_list.append(col_vector.copy())

        ## Scale update matrix to the wrapper update vector scale
        
        # Determine the algorithm range of the product
        AlgMax = self.algorithm_params.row_update.maximum*self.algorithm_params.col_update.maximum
        AlgMin = np.min([self.algorithm_params.row_update.maximum*self.algorithm_params.col_update.minimum,
            self.algorithm_params.row_update.minimum*self.algorithm_params.col_update.maximum])

        # Determine the wrapper range of the product
        WrapMax = self.wrapper_params.row_update.maximum*self.wrapper_params.col_update.maximum
        WrapMin = np.min([self.wrapper_params.row_update.maximum*self.wrapper_params.col_update.minimum,
            self.wrapper_params.row_update.minimum*self.wrapper_params.col_update.maximum])

        # This line functionally performs scale_to for the product of row and column values
        update_matrix *= (WrapMax - WrapMin)/(AlgMax - AlgMin)

        result = self._wrapper_update_matrix_burst(update_matrix, learning_rate)

        # print("weights = ",self._read_matrix())

        # store weights if needed
        if self.params.analytics_params.store_weights:
            if not dont_store_weights:  #don't store values during each serial update
                if self.params.analytics_params.max_storage_cycles == 0 or self.params.analytics_params.max_storage_cycles>self.update_ctr:
                    weights =self._read_matrix()
                    if self.params.analytics_params.all_weights == True:
                        self.weights_list.append(weights)
                    else:
                        self.weights_list.append(weights[self.params.analytics_params.weight_rows, self.params.analytics_params.weight_cols] )
                    self.external_update_counter_list.append(self.external_update_counter[0]) #store external weight index for periodic carry

        if self.params.analytics_params.store_update_inputs or self.params.analytics_params.store_weights: self.update_ctr+=1



        return result


    def serial_update(self, matrix, learning_rate =1, by_row=True):
        """
        Update the matrix serially, one row or column at a time.  Allows for non-neural updates.  Assumes the drivers should be maxed out
        (i.e max out row driver when applying analog updates to the columns)

        :param matrix: matrix of updates to apply
        :param by_row: apply updates one row at a time (else one column at a time if False)
        :type by_row: bool
        :return:
        """
        if by_row:
            max_row_update = self.params.algorithm_params.row_update.absmaxmin
            scaled_updates = matrix/max_row_update

            n_rows = matrix.shape[0]
            for row in range(n_rows):
                col_updates = scaled_updates[row,:]
                row_updates = np.zeros(n_rows)
                row_updates[row]=max_row_update
                if row !=n_rows-1:
                    dont_store_weights=True
                else:
                    dont_store_weights=False
                self.update_matrix(row_updates,col_updates,learning_rate=learning_rate,can_overwrite_row=True,can_overwrite_col=True, dont_store_weights=dont_store_weights)
        else:
            max_col_update = self.params.algorithm_params.col_update.absmaxmin
            scaled_updates = matrix/max_col_update

            n_cols = matrix.shape[1]
            for col in range(n_cols):
                row_updates = scaled_updates[:,col]
                col_updates = np.zeros(n_cols)
                col_updates[col]=max_col_update
                if col !=n_cols-1:
                    dont_store_weights=True
                else:
                    dont_store_weights=False

                self.update_matrix(row_updates,col_updates,learning_rate=learning_rate,can_overwrite_row=True,can_overwrite_col=True, dont_store_weights=dont_store_weights)


    def serial_read(self, by_row=True):
        """
        read the matrix out one row at a time, using VMMs
        :return:
        """
        result = ncp.zeros([self.nrows, self.ncols])
        if by_row:
            n_rows = self.nrows
            for row in range(n_rows):
                vector =  ncp.zeros(n_rows)
                vector[row]=self.algorithm_params.row_input.maximum
                self.set_vmm_inputs(vector)

                # run the vmm
                output = self._wrapper_run_xbar_vmm(output_integrator_scaling=self.algorithm_params.serial_read_scaling)
                # outputs come along columns, scale from wrapper constraints to outer constraints
                output = self.algorithm_params.col_output.scale_from(self.wrapper_params.col_output, output)
                result[row,:]=output/self.algorithm_params.row_input.maximum  # remove input scaling
        else:
            n_cols = self.ncols
            for col in range(n_cols):
                vector =  ncp.zeros(n_cols)
                vector[col]=self.algorithm_params.col_input.maximum
                self.set_mvm_inputs(vector)

                # run the mvm
                output = self._wrapper_run_xbar_mvm(output_integrator_scaling=self.algorithm_params.serial_read_scaling)
                # outputs come along rows, scale from wrapper constraints to outer constraints
                output = self.algorithm_params.row_output.scale_from(self.wrapper_params.row_output, output)
                result[:,col]=output

        return result


    def _read_matrix(self):
        return self.algorithm_params.weights.scale_from(self.wrapper_params.weights,
                                                         self._wrapper__read_matrix())

    def _save_matrix(self):
        return self._wrapper__save_matrix()

    def _restore_matrix(self, matrix):
        # return self._wrapper__restore_matrix(self.outer_core_value_constraints.weights.scale_to(self.wrapper_value_constraints.weights, matrix))
        return self._wrapper__restore_matrix(matrix)

    @abstractmethod
    def _given_wrapper_update_inner_limits(self):
        '''
        Given the wrapper constraints based on the outer constraints, find the derived inner constraints.

        For example, the PosNeg wrapper has double the coefficient dynamic range of the underlying core.
        '''
        raise NotImplementedError()

    @abstractmethod
    def _given_inner_update_wrapper_limits(self):
        '''
        Given the value constraints of the underlying core, this converts the value constraints into the ranges supported by the wrapper.
        
        For example, the PosNeg wrapper has double the coefficient dynamic range of the underlying core.
        '''
        raise NotImplementedError()


    @abstractmethod
    def _wrapper_nonlinear_update_scaling(self, matrix, update):
        """
        Returns what the the update should be given a desired update to account for nonlinearity.
        It is used to pass the average nonlinearity all the way to the top level cores.

        :param matrix:  return the nonlinearity (update scaling) at each value in matrix for an update of 'update'
        :return:
        """
        raise NotImplementedError()


    @abstractmethod
    def _wrapper_set_matrix(self, matrix):
        '''
        Wrapper-specific implementation of :meth:`set_matrix`
        '''
        raise NotImplementedError()

    @abstractmethod
    def _wrapper_set_vmm_inputs(self, vector):
        '''
        Wrapper-specific implementation of :meth:`set_vmm_inputs`
        '''
        raise NotImplementedError()

    @abstractmethod
    def _wrapper_set_mvm_inputs(self, vector):
        '''
        Wrapper-specific implementation of :meth:`set_mvm_inputs`
        '''
        raise NotImplementedError()

    @abstractmethod
    def _wrapper_run_xbar_vmm(self, output_integrator_scaling):
        '''
        Wrapper-specific implementation of :meth:`run_xbar_vmm`
        '''
        raise NotImplementedError()

    @abstractmethod
    def _wrapper_run_xbar_mvm(self, output_integrator_scaling):
        '''
        Wrapper-specific implementation of :meth:`run_xbar_mvm`
        '''
        raise NotImplementedError()

    @abstractmethod
    def _wrapper_update_matrix(self, row_vector, col_vector, learning_rate):
        '''
        Wrapper-specific implementation of :meth:`update_matrix`
        '''
        raise NotImplementedError()

    @abstractmethod
    def _wrapper_update_matrix_burst(self, update_matrix, learning_rate):
        '''
        Wrapper-specific implementation of :meth:`update_matrix`
        '''
        raise NotImplementedError()

    @abstractmethod
    def _wrapper__read_matrix(self):
        '''
        Wrapper-specific implementation of :meth:`_read_matrix`
        '''
        raise NotImplementedError()

    @abstractmethod
    def _wrapper__save_matrix(self):
        '''
        Wrapper-specific implementation of :meth:`_save_matrix`
        '''
        raise NotImplementedError()

    @abstractmethod
    def _wrapper__restore_matrix(self, matrix):
        '''
        Wrapper-specific implementation of :meth:`_restore_matrix`
        '''
        raise NotImplementedError()
