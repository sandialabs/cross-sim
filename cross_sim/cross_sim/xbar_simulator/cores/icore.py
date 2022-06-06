#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

'''
Core interface

'''

from abc import abstractmethod

from ..utilities import DocStringInheritorAbstractBaseClass


class ICore(metaclass=DocStringInheritorAbstractBaseClass):
    '''
    Represents the minimum subset of functions that any MVM core must implement (and provide an end-user)
    
    There are also some convenience functions that were copied over from the (now-defunct) :py:class:`NeuralCore`.
    '''

    @abstractmethod
    def set_matrix(self, matrix, applyErrors=True):
        r'''
        Sets the matrix to use

        :math:`\mathbf{matrix}\leftarrow\mathtt{matrix}`
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def set_vmm_inputs(self, vector):
        r'''
        Sets the vector to use for :meth:`run_xbar_vmm`.

        :math:`\overrightarrow{vector\_vmm}\leftarrow\mathtt{vector}`
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def set_mvm_inputs(self, vector):
        r'''
        Sets the vector to use for :meth:`run_xbar_mvm`.
        
        :math:`\overrightarrow{vector\_mvm}\leftarrow\mathtt{vector}`
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def run_xbar_vmm(self):
        r'''
        Returns :math:`\overrightarrow{vector\_vmm}\cdot\mathbf{matrix}`
        '''
        raise NotImplementedError()

    @abstractmethod
    def run_xbar_mvm(self):
        r'''
        Returns :math:`\mathbf{matrix}\cdot\overrightarrow{vector\_mvm}`
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def update_matrix(self, row_vector, col_vector, learning_rate=1):    #, scale = 1.0
        r'''
        Updates the matrix given input row and column vectors.
        
        :math:`\mathbf{matrix}\leftarrow\mathbf{matrix}+\left(\overrightarrow{\mathtt{row\_vector}}\otimes\overrightarrow{\mathtt{col\_vector}}\right)`
        
        '''
#         Previously:
#         :param scale    Scales the vectors before applying them to the matrix (this has been removed.)
#         
#         :math:`\mathbf{matrix}\leftarrow\mathbf{matrix}+\mathtt{scale}\times\left(\overrightarrow{\mathtt{row\_vector}}\otimes\overrightarrow{\mathtt{col\_vector}}\right)`
#         
        raise NotImplementedError()
    
    @abstractmethod
    def _read_matrix(self):
        '''
        Read the internal matrix held by this core (debug method)
        
        Data is corrected for scaling at the level it is observed.
        
        No quantization or other errors are applied.
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def _save_matrix(self):
        '''
        Save the internal matrix held by this core (debug method)
        
        Unlike _read_matrix, all data necessary to restore the matrix is provided.

        No quantization or other errors are applied.
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def _restore_matrix(self, matrix):
        '''
        Restore an internal matrix held by this core (debug method)
        
        You should only use a matrix obtained from _save_matrix, as _read_matrix may remove needed values (e.g.: from an offset core).
        
        No quantization or other errors are applied.
        '''
        raise NotImplementedError()

    @abstractmethod
    def nonlinear_update_scaling(self, matrix, update):
        """
        Returns what the the update should be given a desired update to account for nonlinearity.
        It is used to pass the average nonlinearity all the way to the top level cores.

        :param matrix:  return the nonlinearity (update scaling) at each value in matrix for an update of 'update'
        :return:
        """
        raise NotImplementedError()


    def calibrate_weights(self, matrix, cycles = 1):
        """
        Calibrates the weights by performing two types simulations
        First simulates only a single row on to adjust the weights
        Second simulates all the rows on and only turns one row off
        Averaging the result of both simulations allows for an average input condition
        :rtype: None
        """
        import numpy as np
        self.set_matrix(matrix)
        corrected_weights = matrix.copy()
        rows = matrix.shape[0]
        #TODO: does not work when the bias is shifted from zero
        def sim_crossbar(driver_inputs, **kwargs):
            self.set_vmm_inputs(driver_inputs, **kwargs)
            results = self.run_xbar_vmm()
            return results.squeeze()
        
        orig_matrix = self._read_matrix()

        weight_error_old = np.zeros(orig_matrix.shape)
        weight_correction_old =np.zeros(orig_matrix.shape)

        for cycle in range(cycles):
    
            # **** first simulate a crossbar with all inputs on
    
            driver_inputs = np.ones(rows)
            simulated_all_on = sim_crossbar(driver_inputs)

            for row in range(rows):
                print("Calibrating Row " + str(row + 1) + " of " + str(self.nrows))

                target_weight_row = matrix[row, :]
                # corrected_weight_row = self.corrected_weights[row, :]
    
                # **** simulate only a single row on
    
                # create driver to drive only the row being calibrated
                driver_inputs = np.zeros(rows)
                driver_inputs[row] = 1

                # compute the sum, which will return the simulated weights of the current row
                simulated_weight_row_on = sim_crossbar(driver_inputs) - 0
    
                # **** simulate only a single row off
    
                # create driver to drive only the row being calibrated
                driver_inputs = np.ones(rows)
                driver_inputs[row] = 0
    
                # compute the sum, which will return the simulated weights of the current row
                simulated_weight_row_off = simulated_all_on - sim_crossbar(driver_inputs)
                
                #Compute average value for the row
                simulated_weight_row_average = (simulated_weight_row_on + simulated_weight_row_off)/2.0

                #Compute error in weights
                weight_error = target_weight_row - simulated_weight_row_average
                
    #             #TODO: this is not P!!!! Fix it! (for use with Isaac's formula)
    #             programmed_resistance = 1.0/weight_row      #This has a divide-by-zero problem that needs to be fixed!!!
    #             observed_resistance = 1.0/simulated_weight_row_average
    #             observed_parasitic_resistance = observed_resistance - programmed_resistance
    #             
    #             #Using Isaac's formula:
    #             self.corrected_weights[row,:] = weight_row/(1-(weight_row*observed_parasitic_resistance))

                #Using Sapan's formula (not Isaac's formula):
                if cycle == 0:
                    weight_correction = weight_error
                else:
                    weight_correction = weight_correction_old[row,:]*(weight_error/(weight_error_old[row,:]-weight_error))

                corrected_weights[row,:] += weight_correction

                #save errors from the previous cycle
                weight_error_old[row,:] = weight_error.copy()
                weight_correction_old[row,:] = weight_correction.copy()
            
            self.set_matrix(corrected_weights)
    
    def save_weights(self, filename):
        '''
        Convenience function to :meth:`_save_matrix` and then :meth:`numpy.savetxt` save it to a file.
        
        If necessary, the destination directory will be created via :meth:`os.makedirs`.
        
        :param filename:    File in which we should save the values
        :type filename: str
        '''
        import os
        os.makedirs(os.path.dirname(filename), exist_ok = True)
        import numpy as np
        np.savetxt(filename, self._save_matrix())
    
    def load_weights(self, filename):
        '''
        Convenience function to :meth:`numpy.loadtxt` from a file and then :meth:`_restore_matrix` it.
        
        :param filename    File from which we should load the values
        :type filename: str
        '''
        import numpy as np
        self._restore_matrix(np.loadtxt(filename))

