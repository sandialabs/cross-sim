#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np

from .clipper_core import ClipperCore
from ..parameters.parameter_defaults import WriteNoiseModelEnum, UpdateModelEnum
from warnings import warn


class HardwareCore(ClipperCore):
    '''
    An inner :py:class:`.ICore` that performs purely-numeric calculations
    '''

    def __init__(self, params):
        """

        :param params:
        :type params: Parameters
        :return:
        """
        self.matrix = None
        self.mcopy = None  # scratch space to store matrix during computations without repeatedly re-allocating space.
        self.vector_vmm = None
        self.vector_mvm = None
        ClipperCore.__init__(self, params)

    def set_matrix(self, matrix):
        matrix = self.clip_matrix(matrix)
        # Matrix values are from .1 to 1, resistance should be set to be proportional to these values
        current_matrix = self._read_matrix()
        diff = matrix - current_matrix
        minimum_error = matrix * self.params.hardware_params.set_matrix_error

        if diff > minimum_error:
            diff = diff / self.xbar_params.col_update.maximum
            n_cols = self.cols
            for col in range(n_cols):
                row_updates = diff[:, col]
                col_updates = np.zeros(n_cols)
                col_updates[col] = self.xbar_params.col_update.maximum
                self.update_matrix(row_updates, col_updates)
            # Call set_matrix again to do another update if needed
            self.set_matrix(matrix)

    def set_vmm_inputs(self, vector):
        vector = self.clip_vmm_inputs(vector)
        self.vector_vmm = vector

    def set_mvm_inputs(self, vector):
        # clip and quantize vector
        vector = self.clip_mvm_inputs(vector)
        self.vector_mvm = vector

    def run_xbar_vmm(self):

        # If using offset core, remove first element of 0 and use remaining elements to calculate current to subtract
        if self.subtract_current_in_xbar:
            vector = self.vector_vmm[1:]
            offset = np.ones(vector.shape) * self.xbar_params.weights.middle
            current_to_subtract = vector.dot(offset)
        else:
            vector = self.vector_vmm

        # Scale vector to voltage range
        # TODO: Determine what this value should be, vector right now is in range [-1,1]
        voltage_scaling = 1.0

        vector = vector/voltage_scaling

        # TODO: Replace this with labview code, put vector on each row and read col output
        result = vector.dot(self.matrix)

        # if using offset core subtract first row current from all outputs
        if self.subtract_current_in_xbar is True:
            result -= current_to_subtract
        return result

    def run_xbar_mvm(self):

        # If using offset core, remove first element of 0 and use remaining elements to calculate current to subtract
        if self.subtract_current_in_xbar:
            vector = self.vector_mvm[1:]
            offset = np.ones(vector.shape) * self.xbar_params.weights.middle
            current_to_subtract = vector.dot(offset)
        else:
            vector = self.vector_mvm

        # Scale vector to voltage range
        # TODO: Determine what this value should be, vector right now is in range [-1,1]
        voltage_scaling = 1.0

        vector = vector/voltage_scaling

        # TODO: Replace this with labview code, put vector on each col and read row output
        result = self.matrix.dot(vector)

        # if using offset core subtract first row current from all outputs
        if self.subtract_current_in_xbar is True:
            result -= current_to_subtract
        return result

    def update_matrix(self, row_vector, col_vector, learning_rate):
        row_vector, col_vector = self.clip_update_matrix_inputs(row_vector, col_vector)

        row_bits = self.xbar_params.row_update.bits
        col_bits = self.xbar_params.col_update.bits
        row_spacing = self.xbar_params.row_update.maximum / (2 ** row_bits - 1)
        col_spacing = self.xbar_params.col_update.maximum / (2 ** col_bits - 1)
        row_pulses = np.rint(row_vector/row_spacing)
        col_pulses = np.rint(col_vector/col_spacing)

        # row_pulses is number of pulses to apply for row, col_pulses is number of pulses to apply for columns

        def get_positives(x):
            if x > 0:
                return x
            else:
                return 0
        def get_negatives(x):
            if x < 0:
                return x
            else:
                return 0

        positive_rows = map(get_positives(), row_pulses)
        negative_rows = map(get_negatives(), row_pulses)

        # TODO: Use positive/negative row_pulses and col_pulses with labview to update matrix, need to separate into cycles


        update = np.outer(row_vector*learning_rate, col_vector)

        self.matrix += update

        # apply postpocessing / clip matrix values to limits and quantize
        self.matrix=self.xbar_params.weights.quantize(self.xbar_params.weight_clipping.clip(self.matrix))



    def _read_matrix(self):
        """
        read the matrix out one row at a time, using VMMs
        :return:
        """
        # TODO: Verify offset is being added correctly
        result = np.zeros([self.rows, self.cols])

        n_rows = self.rows
        for row in range(n_rows):
            vector = np.zeros(n_rows)
            vector[row] = self.xbar_params.row_input.maximum
            self.set_vmm_inputs(vector)

            # run the vmm
            output = self.run_xbar_vmm()
            # outputs come along columns, scale from measured current to resistance to proportional weight
            # output may also be in resistance already, then don't need to divide by voltage

            # Calculate resistance from Ohm's law, voltage is the maximum row_input
            output = output/self.xbar_params.row_input.maximum

            # Need min and max resistance values. Might be able to do this in labview side?
            min_resistance = 1000
            max_resistance = 2000
            range = max_resistance - min_resistance
            output = (output - min_resistance) / range

            # If using offset core add offset to front of each row
            if self.subtract_current_in_xbar:
                output = [self.xbar_params.weights.middle] + output

            # Result will be value in [0:1]
            #TODO: Verify this is correct way to read
            result[row, :] = output

        # If using offset core, add offset row to the front of the result
        if self.subtract_current_in_xbar:
            offset = np.ones(result[0].shape) * self.xbar_params.weights.middle
            result = offset + result

        return result


    def _save_matrix(self):
        return self.matrix.copy()


    def _restore_matrix(self, matrix):
        self.matrix = matrix.copy()



    def nonlinear_update_scaling(self, matrix, update):
        matrix = self.clip_matrix_nonlinear_update_scaling(matrix) #clip matrix
        raise NotImplementedError()
