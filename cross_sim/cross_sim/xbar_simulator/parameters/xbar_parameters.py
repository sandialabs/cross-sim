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


from .base import ParametersBase
from . import parameter_defaults as params

from .device_parameters import DeviceParameters

class XyceXbarParameters(ParametersBase):
    if False:
        # define parameters for easy code completion (these can be commented out, it's only to help the IDE)
        cell_spacing = float
        cell_resistance = float
        resistance_sigma = float
        cell_capacitance = float
        cell_inductance = float
        device = DeviceParameters
        USE_PARASITICS = bool
        LUMPED_PARASITICS = bool


    def __init__(self, param_root):
        attributes = params.XyceXbarParameterDefaults.copy()

        # create a device parameter set
        attributes['device']=DeviceParameters(param_root)

        # load the defaults
        ParametersBase.__init__(self,param_root, **attributes)


    @property
    def wire_resistivity(self):  # per unit length
        return self.cell_resistance/self.cell_spacing

    @property
    def wire_capacitance(self):  # per unit length
        return self.cell_capacitance/self.cell_spacing

    @property
    def wire_inductance(self):  # per unit length
        return self.cell_inductance/self.cell_spacing
