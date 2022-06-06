#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

__author__ = 'sagarwa'

from .base import ParametersBase, Parameter
from . import parameter_defaults as params


class AnalyticsParameters(ParametersBase):

    if False:
        store_weights=float #store the weights after each update
        store_update_inputs=float #store the inputs to each update
        store_row_inputs=float #store all row inputs
        store_col_inputs=float #store all col inputs
        store_row_outputs=float #store all row outputs
        store_col_outputs=float #store all col outputs
        max_storage_cycles=float, # number of cycles for which to store internal results (to prevent memory cost from blowing up
        all_weights=bool# if true, store all weights, if false, store weights specified by weight_rows and weight _cols
        weight_rows= list #row index of weights to save
        weight_cols= list # col index of weights to save
        no_update_rows=bool
        no_update_cols=bool


    def __init__(self, param_root):
        #load the defaults
        ParametersBase.__init__(self,param_root, **params.AnalyticsDefaults)
