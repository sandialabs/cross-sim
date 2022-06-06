#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

__author__ = 'sagarwa'

# from parameters_simple.base import ParametersBase, Parameter
# import parameters_simple.parameter_defaults as params
# from parameters_simple.neuron_parameters import NeuronParameters
# from parameters_simple.xbar_parameters import XyceXbarParameters
# from parameters_simple.driver_parameters import SharedDriverParameters, DriverParameters
import copy

from .xyce_parameters import XyceParameters
from .valueconstraints import XbarParams, AlgorithmParams, WrapperParams #,ClipConstraints, QuantizationAndNoiseConstraints,ClipQuantizeAndNoiseConstraints,
from .backprop import BackpropParameters
from .numeric_params import NumericParams
from .hardware_parameters import HardwareParameters
from .memory_parameters import MemoryParameters
from enum import Enum
from .base import ParametersBase
from .periodic_carry_parameters import PeriodicCarryParameters
from .convolution_parameters import ConvolutionParameters
from .analytics_params import AnalyticsParameters
from .weight_error_parameters import WeightErrorParameters

'''
They Xyce parameter structure is as follows:


* Xyce :py:class:`XyceParameters`

  * row_driver :py:class:`.DriverParameters`
  * col_driver :py:class:`.DriverParameters`
  * shared_driver :py:class:`.SharedDriverParameters`
  * row_neuron :py:class:`.NeuronParameters`
  * col_neuron :py:class:`.NeuronParameters`
  * xbar :py:class:`.XyceXbarParameters`

    * device :py:class:`.DeviceParameters`

    param_root.xyce_parameters.row_driver
    param_root.xyce_parameters.col_driver
    param_root.xyce_parameters.shared_driver
    param_root.xyce_parameters.shared_driver.write_pos
    param_root.xyce_parameters.shared_driver.write_neg
    param_root.xyce_parameters.row_neuron
    param_root.xyce_parameters.col_neuron
    param_root.xyce_parameters.xbar
    param_root.xyce_parameters.xbar.device


'''


class Parameters(object):
    """
    This is master parameter class/object that will hold all parameters for a simulation
    """

    def __init__(self):
        self.xyce_parameters = XyceParameters(self)
        """:type: XyceParameters"""

        self.xbar_params = XbarParams(self)

        self.algorithm_params = AlgorithmParams(self)
        self.wrapper_params = WrapperParams(self)  # fully derived set of parameters for wrapper
        self.periodic_carry_params = PeriodicCarryParameters(self)
        self.convolution_parameters = ConvolutionParameters(self)
        self.weight_error_params = WeightErrorParameters(self)

        # self.backprop_parameters = BackpropParameters(self)
        self.numeric_params = NumericParams(self)
        self.memory_params = MemoryParameters(self)
        self.analytics_params = AnalyticsParameters(self)

        self.hardware_params = HardwareParameters(self)


    def copy(self):
        """
        returns a deep copy of itself
        :return params:
        :rtype: Parameters

        """
        params = copy.deepcopy(self)
        change_param_root(param=params,new_root=params)
        return params





def print_parameters(param, name="params",include_derived = False, output = "", exclude_error = False):
    """
    returns a string with all the stored parameters for printing

    :param param: the parameter object
    :param name: a name object the prefixes all printed outputs
    :param include_derived: also print parameters derived from others that do not need to be set
    :param output: the output string (used for paassing the string through recursive calls)
    :param exclude_error:  exclude noise/error parameters: normal_error_post, normal_error_pre and uniform_error_post
    :return:
    """
    from .base import Parameter

    if hasattr(param,"__dict__"): # check if it has subparameters
        if isinstance(param, Enum): # enums have dicts but we want the value
            output += name+" = \'"+str(param.name)+"\'\n"
        else:
            output+="\n"
            for key in sorted(param.__dict__.keys()):
                if key != "param_root" and key !="__objclass__" and key!="override_readonly" and \
                        ( (not exclude_error ) or (key!='normal_error_post' and key!='normal_error_pre' and key!='uniform_error_post')): #exlude noise params if set
                    #if derived/readonly parameters are included print everything
                    if include_derived is True:
                        output = print_parameters(param.__dict__[key], str(name+"."+key),include_derived,output, exclude_error)

                    #figure out if parameter is readonly by checking if it is in Parameter class and then seeing if it can be written to
                    elif isinstance(getattr(type(param),key,None), Parameter):
                        try:
                            setattr(param,key,getattr(param,key))
                            output = print_parameters(param.__dict__[key], str(name+"."+key),include_derived,output, exclude_error)
                        except AttributeError:
                            #dont print anything if parameter is readonly
                            pass
                    else:
                            #if not in parameter class continue printing
                            output = print_parameters(param.__dict__[key], str(name+"."+key),include_derived,output, exclude_error)
            output+="\n"

    else:
        #output value if it is a simple parameter
        if isinstance(param,str): # if it is a string add quotes
            output+=name+" = \'"+str(param)+"\'\n"
        else:
            output+=name+" = "+str(param)+"\n"

    return output

def change_param_root(param, new_root):
    """
    Change the param_root in a parameter object and all it's sub objects

    :param param:  the object in which to change the root
    :param new_root:  the enw root
    :return:
    """

    if isinstance(param, Parameters) or isinstance(param, ParametersBase):  # check if it is a parameter object
    # if hasattr(param,"__dict__"): # check if it has subparameters
        for key in sorted(param.__dict__.keys()):
            #change the root and recursively call function on all other keys
            if key == "param_root":
                setattr(param,key,new_root)
            else:
                change_param_root(param.__dict__[key], new_root)
    # do nothing if not a parameter object
