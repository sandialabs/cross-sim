#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

__author__ = 'sagarwa'

from .base import ParametersBase, Parameter
from . import parameter_defaults as params
from .valueconstraints import ClipQuantizeAndNoiseConstraints
import enum



class HardwareParameters(ParametersBase):
    '''
	Parameters for setting running a hardware crossbar
    '''

    if False:  # define parameter names here that are passed into the init function to help the IDE autocomplete
        binary_updates = bool
        relative_update_size = float
        set_matrix = ClipQuantizeAndNoiseConstraints

    def __init__(self, param_root):
        attributes = params.HardwareParametersDefaults.copy()  # load the default parameter names and values from parameter_defaults.py
        # load the defaults
        ParametersBase.__init__(self, param_root,
                                **attributes)  # create parameters with variable names and values in attributes


    # the following optional function is only called after all parameters are set from core_initalization.py.  This allows for complex derived parameters to be set
    # add call to core_initialization if used
    def manual_post_set(self):
        # set update ranges based on relative_update_size
        pass