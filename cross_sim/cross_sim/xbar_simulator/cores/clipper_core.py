#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from abc import abstractmethod, ABCMeta

from . import ICore
from .debug import print_debug_calls
from ..parameters import Parameters

# class PostprocUnsupported(ValueError):
#     '''
#     An exception that is bubbled by :py:meth:`.XyceCore.update_matrix` (and maybe future cores?) if a postproc method is passed. The higher level core will catch this and retry the call without postprocessing.
#     '''
#     pass

class ClipperCore(ICore, metaclass = ABCMeta):
    '''
    Handles clipping and quantization of values going to and from the inner core (which is a subclass of :py:class:`ClipperCore`.)
    
    The subclass must implement the _clipper_* methods.
    '''
    



    def __init__(self, params):
        '''
        Create a Clipper Core
        
        :param params: all paramters
        :type params: Parameters
        :param inner_core: An instantiated inner core (an instance, not a class or factory!)
        :type inner_core: ICore
        '''

        self.params = params
        self.xbar_params = params.xbar_params
        # self._clipper__notify_core_value_constraints(self.xbar_params)

        #if this is true, the first neuron along a row/col should saturate at half the value of the others
        #TODO:  Implement this scaling in the Xyce core neuron capacitance, not only in the after the fact clipping (if this is useful)
        self.rescale_offset_neuron = False

        # if set to true by wrapper core, the first row current is subtracted prior to clipping.
        # TODO:  implent this in Xyce
        self.subtract_current_in_offset_xbar = False
        self.rows = None
        self.cols = None

        self.record_updates = params.analytics_params.record_updates
        if self.record_updates:
            self.Nupdates_total = params.analytics_params.Nupdates_total
            self.target_updates = None
            self.real_updates = None


    #################################################
    ######## Code below does input clipping

    def clip_matrix(self, matrix):
        self.rows = matrix.shape[0]
        self.cols = matrix.shape[1]
        return self.xbar_params.weight_clipping.clip(matrix) # overwrite input


    def clip_vmm_inputs(self, vector):
        return self.xbar_params.row_input.clip_and_quantize(vector)

    def clip_mvm_inputs(self, vector):
        return self.xbar_params.col_input.clip_and_quantize(vector)



    def clip_and_quantize_update_matrix_inputs(self, row_vector, col_vector):
        row_vector = self.xbar_params.row_update.clip_and_quantize(row_vector)
        col_vector = self.xbar_params.col_update.clip_and_quantize(col_vector)

        return row_vector, col_vector

    # @abstractmethod
    def clip_matrix_nonlinear_update_scaling(self, matrix):
        return self.xbar_params.weight_clipping.clip(matrix) #clip matrix

