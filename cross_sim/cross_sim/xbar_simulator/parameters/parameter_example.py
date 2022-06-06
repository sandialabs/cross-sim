#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from .base import ParametersBase, Parameter
from . import parameter_defaults as params
import enum


###########################
# In order to define the defaults value, the following dict and class should be added to parameter_defaults.py

# this defines the options for an enumerated parameter
class NewParam4Enum(enum.IntEnum):
    OPTION0 = 0
    OPTION1 = 1
    OPTION2 = 2
    OPTION_MONKEY = 3

# this defines all the default values
NewParametersDefaults = {
    'new_param1': 1.1,
    'new_param2': 3.14159,
    'new_param3': 3,
    'new_param4': NewParam4Enum.OPTION1
}

###############################
###############################


class NewParameters(ParametersBase):
    '''
	Example class for creating new parameters
    '''

    if False:  # define parameter names here that are passed into the init function to help the IDE autocomplete
        new_param1 = float
        new_param2 = float

    def __init__(self, param_root, **kwargs):
        # optionally including **kwargs allows for initial values to be passed when creating an instance of NewParameters
        attributes = params.NewParametersDefaults.copy()  # load the default parameter names and values from parameter_defaults.py
        attributes.update(kwargs)  # add the passed parameters to attributes
        # load the defaults
        ParametersBase.__init__(self, param_root,
                                **attributes)  # create parameters with variable names and values in attributes

    # parameter properties can further customized by defining the parameter as a Parameter object
    derived_param1 = Parameter(name="derived_param1", readonly=True, post_set=None)
    # Setting readonly = True makes the parameter readonly so that the user cannot write to it.  In order to override the readonly setting,
    # the variable self.override_readonly=True must exist in the object trying to write to the parameter.  Useful for derived parameters


    # post_set is a function that is called after the variable is assigned a value.  This allows for derived values to be computed.
    def post_set(self):
        # this is an example post set function that sets readonly parameter derived_param1=new_param3+1 whenever new_param3 is set
        self.override_readonly = True
        derived_param1 = self.new_param3 + 1
        self.override_readonly = False

    new_param3 = Parameter(name="new_param3", readonly=False, post_set=post_set)


    # if the parameter is an enum (i.e. one of a list of settings), using generate_enum_post_set allows the parameter to be set with a string
    # and checks that the string is valid from the list of possible strings set in NewParam4Enum in parameter_defaults.py
    new_param4 = Parameter(name="new_param4", readonly=False,
                           post_set=ParametersBase.generate_enum_post_set("new_param4", NewParam4Enum))


    # readonly derived parameters can also be set as follows:
    @property
    def derived_parameter2(self):
        return self.new_param3 - self.new_param1

    # the following optional function is only called after all parameters are set from core_initalization.py.  This allows for complex derived parameters to be set
    def manual_post_set(self):
        print(
            self.param_root.algorithm_params.weights.maximum / 2)  # can access parameters in other objects by referring to param_root

# error checking should be coded in verify_parameters in core_initalization.py
