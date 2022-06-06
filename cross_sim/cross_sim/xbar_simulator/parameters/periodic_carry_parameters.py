#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#


from .base import ParametersBase, Parameter
from . import parameter_defaults as params
from .parameter_defaults import ZeroResetPCEnum
# from .parameter_defaults import NeuronStyleEnum


class PeriodicCarryParameters(ParametersBase):

    if False:
        use_periodic_carry=bool
        cores_per_weight=int
        number_base = float
        carry_threshold = float
        normalized_output_scale = float
        read_low_order_bits = bool
        carry_frequency = list
        ''':type: list of int'''
        exact_carries = bool
        zero_reset = ZeroResetPCEnum
        min_carry_update = float

    zero_reset = Parameter(name="zero_reset",
                           post_set=ParametersBase.generate_enum_post_set("zero_reset", ZeroResetPCEnum))

    # style = Parameter (name = "style", value = None, post_set= ParametersBase.generate_enum_post_set("style", NeuronStyleEnum) )

    def __init__(self, param_root):
        #load the defaults
        ParametersBase.__init__(self,param_root, **params.PeriodicCarryDefaults)
