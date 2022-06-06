#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

'''
They Xyce parameter structure is as follows:


* Xyce :py:class:`XyceParameters`
  
  * row_driver :py:class:`.DriverParameters`
  * col_driver :py:class:`.DriverParameters`
  * shared_driver :py:class:`.SharedDriverParameters`
  * row_neuron :py:class:`.NeuronParameters`
  * col_neuron :py:class:`.NeuronParameters`
  * xbar :py:class:`.XbarParameters`

    * device :py:class:`.DeviceParameters`
'''



from .base import ParametersBase, Parameter
from . import parameter_defaults as params
from .driver_parameters import DriverParameters, SharedDriverParameters
from .neuron_parameters import NeuronParameters
from .xbar_parameters import XyceXbarParameters
from .parameter_defaults import ParasiticCompensationEnum, XyceTimeSteppingEnum
import os, shutil
from tempfile import mkdtemp


class XyceParameters(ParametersBase):
    """

    """
    if False:
        # define var names for easy code completion (these can be commented out, it's only to help the IDE)
        row_driver = DriverParameters
        col_driver = DriverParameters
        shared_driver = SharedDriverParameters
        row_neuron = NeuronParameters
        col_neuron = NeuronParameters
        xbar = XyceXbarParameters
        xyce_path=str
        xyce_parallel_path = str
        xyce_run_command = str
        n_processors = int
        parallel_solve = bool
        # simulation_settle_time = float
        debug_output = bool
        calculate_power = bool
        output_rram_voltages = bool
        parasitic_compensation_parallelism = int
        print_all_time_steps = bool
        xyce_options=str
        parallel_timeout = float
        out_dir =str

    '''
    Generic parameters relevant to the running of Xyce, and to top-level objects (drivers, neurons, crossbar)

    '''

    parasitic_compensation = Parameter(name="parasitic_compensation",
                                       post_set=ParametersBase.generate_enum_post_set("parasitic_compensation",
                                                                                      ParasiticCompensationEnum))
    xyce_time_stepping = Parameter(name="xyce_time_stepping",
                                   post_set=ParametersBase.generate_enum_post_set("xyce_time_stepping",
                                                                                  XyceTimeSteppingEnum))

    del_out_dir = Parameter(name="del_out_dir",readonly=True)
    # @property
    # def read_sim_time_needed(self):
    #     return self.shared_driver.read_sim_time_needed


    def __init__(self, param_root):
        attributes = params.XyceParametersDefaults.copy()


        attributes['row_driver']=DriverParameters(param_root,**params.XyceRowDriverDefaults)
        attributes['col_driver']=DriverParameters(param_root,**params.XyceColDriverDefaults)
        attributes['shared_driver']=SharedDriverParameters(param_root)
        # attributes['shared_driver']=SharedDriverParameters(param_root,**params.XyceSharedDriverDefaults) #TODO: does not accept separate pos/neg defaults
        attributes['row_neuron']=NeuronParameters(param_root)
        attributes['col_neuron']=NeuronParameters(param_root)

        #TODO: Driver shared things (pulse widths, ...)
    
        attributes['xbar']=XyceXbarParameters(param_root)

        ParametersBase.__init__(self,param_root, **attributes)


    def manual_post_set(self):
        """
        create temp directory and set delete directory flag if needed
        :return:
        """
        self.override_readonly = True
        if self.out_dir is None:
            self.out_dir = mkdtemp()
            self.del_out_dir = True
        else:
            self.del_out_dir = False
        self.override_readonly = False

    def __del__(self):
        '''
        If we made up a name for the temporary directory, delete it as we're dying
        this is called when the object is destroyed
        '''

        if self.del_out_dir:
            if os.path.exists(self.out_dir):
                shutil.rmtree(self.out_dir)
