#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

'''
Created on Aug 31, 2015

@author: sagarwa
'''

import math

from .base import ParametersBase, Parameter
from . import parameter_defaults as params
from .parameter_defaults import DeviceModelEnum

class DeviceParameters(ParametersBase):
    '''
    These parameters specify how specific devices (memristor + access device, if enabled) behave
    ** do not add a manual_post_set due to execution order in core initialization
    '''

    if False:
        # define var names for easy code completion (these can be commented out, it's only to help the IDE)
        Gmin_relative = float # the starting point of the used conductance range (normalized to the range)
        Gmax_relative = float # the starting point of the used conductance range (normalized to the range)
        ACCESS_DEVICE_voltage_margin = float
        ACCESS_DEVICE_V0 = float
        rram_capacitance = float
        USE_ACCESS_DEVICE = bool
        USE_DEVICE_CAPACITANCE =bool
        GROUND_DEVICE_CAPACITANCE=bool
        model = DeviceModelEnum
        YAKOPCIC_A1 = float
        YAKOPCIC_A2 = float
        YAKOPCIC_B = float
        YAKOPCIC_VP = float
        YAKOPCIC_VN = float
        YAKOPCIC_AP = float
        YAKOPCIC_AN = float
        YAKOPCIC_XP = float
        YAKOPCIC_XN = float
        YAKOPCIC_ALPHAP = float
        YAKOPCIC_ALPHAN = float
        YAKOPCIC_ETA = float
        RESNOISE = bool
        RESSEED = float
        RESLAMBDA = float
        RESTD = float
        RESEPTD = float
        RESDELTA = float
        RESDELTAGRAD = float
        access_device_capacitance = float
        PEM_fxpdata=str
        PEM_fxmdata=str
        PEM_I1 = float
        PEM_I2 = float
        PEM_V1 = float
        PEM_V2 = float
        PEM_G0 = float
        PEM_VP = float
        PEM_VN = float
        PEM_d1 = float
        PEM_d2 = float
        PEM_C1 = float
        PEM_C2 = float

    model = Parameter(name="model", post_set=ParametersBase.generate_enum_post_set("model", DeviceModelEnum))


    def __init__(self, param_root):

        #load the defaults
        ParametersBase.__init__(self,param_root, **params.DeviceParameterDefaults)

    @property
    def ACCESS_DEVICE_I0(self):
        return 10e-9/(math.exp(self.ACCESS_DEVICE_voltage_margin/(2*self.ACCESS_DEVICE_V0))-1)
