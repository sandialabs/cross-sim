#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#


from .base import ParametersBase, Parameter
from . import parameter_defaults as params
from .parameter_defaults import NeuronStyleEnum


class NeuronParameters(ParametersBase):

    if False:
        opamp_gain = float
        load_resistance = float
        integrator_capacitance = float
        input_impedance = float
        highz_impedance = float


    style = Parameter(name="style", post_set=ParametersBase.generate_enum_post_set("style", NeuronStyleEnum))

    def __init__(self, param_root):
        #load the defaults
        ParametersBase.__init__(self,param_root, **params.NeuronParameterDefaults)
