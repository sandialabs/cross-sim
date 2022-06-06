#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#


from .base import ParametersBase, Parameter
from . import parameter_defaults as params
from .parameter_defaults import MemoryReadModelEnum


class MemoryParameters(ParametersBase):

    if False:
        Ileak = float

        write_time = float
        read_time = float

        Vword_read=float
        Vbit_read=float
        Vrow_read=float
        Vcol_read=float

        Vword_write=float
        Vbit_write=float
        Vrow_write=float
        Vcol_write=float

        drive_impedance=float
        read_impedance=float
        highz_impedance=float

    read_model = Parameter(name="read_model",
                           post_set=ParametersBase.generate_enum_post_set("read_model", MemoryReadModelEnum))

    def __init__(self, param_root):
        #load the defaults
        ParametersBase.__init__(self,param_root, **params.MemoryParameterDefaults)


    @property
    def read_eval_time(self):
        """
        Returns the time at which to evaluate the xyce sim output for outputs
        :return:
        """
        shared_params = self.param_root.xyce_parameters.shared_driver
        return shared_params.initial_wait_time+shared_params.rise_time+self.read_time/2

    @property
    def write_eval_time(self):
        """
        Returns the time at which to evaluate the xyce sim output for write outputs
        :return:
        """
        shared_params = self.param_root.xyce_parameters.shared_driver
        return shared_params.initial_wait_time+shared_params.rise_time+self.write_time/2
