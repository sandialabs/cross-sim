#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

'''
Parameters used by :py:class:`backprop`
'''
from .base import ParametersBase#, make_type_validator
from . import parameter_defaults as params



class BackpropParameters(ParametersBase):
    """
    These parameters are not used, backprop is currently a separate module built on top of xbar_simulator
    """

    # if False: # define parameters for easy code completion
    #     USE_WEIGHT_CALIBRATION=bool, #Specifies that weight calibration should be performed on loaded weights before they are used
    #     LOAD_CALIBRATED_WEIGHTS=bool, #Specifies that previously saved weights should be loaded
    #     random_seed = int
    #
    # def __init__(self, param_root):
    #     #load the defaults
    #     ParametersBase.__init__(self,param_root, **params.BackpropParameterDefaults)



# ******* the following would go in parameter_defaults if paramters are used
# BackpropParameterDefaults  ={
#     "USE_WEIGHT_CALIBRATION" : False, #Specifies that weight calibration should be performed on loaded weights before they are used
#     "LOAD_CALIBRATED_WEIGHTS" : False, #Specifies that previously saved weights should be loaded
#     "random_seed" : 233
#     #     ACTIVATE_TYPE = Parameter(default=ActivateTypes.SIGMOID, validate=make_type_validator(ActivateTypes))
#     #     ACTIVATE_SLOPE = Parameter(default=1.0)
#     #     PRECISION = np.float32
# }
