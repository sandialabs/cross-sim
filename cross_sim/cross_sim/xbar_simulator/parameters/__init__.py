#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

'''
Parameters defaults are stored in this package.
'''

#Fail fast with sensible error message if we do not have enumerations
try:
    from enum import Enum
except ImportError as ie:
    ie.msg = 'Please use python 3.4, or install enum34 from pypi. (Caused by: {0}.)'.format(ie.msg)
    raise


# from .valueconstraints import ClipConstraints, NormalError, UniformError, QuantizationAndNoiseConstraints, ClipQuantizeAndNoiseConstraints, RequiredCoreClipConstraints, SupplementalCoreClipConstraints, AllCoreClipConstraints, CoreQuantizationConstraints, CoreValueConstraints
#

from .all_parameters import Parameters, print_parameters
from .backprop import BackpropParameters
from .driver_parameters import DriverParameters
from .driver_parameters import SharedDriverParameters
from .neuron_parameters import NeuronParameters
from .xbar_parameters import XyceXbarParameters
from .xyce_parameters import XyceParameters
from .valueconstraints import XbarParams, AlgorithmParams, WrapperParams, ClipConstraints
from .device_parameters import DeviceParameters
from .analytics_params import  AnalyticsParameters
from .convolution_parameters import ConvolutionParameters
from .weight_error_parameters import WeightErrorParameters


from .parameter_defaults import DriverStyleEnum, DeviceModelEnum, \
    MemoryReadModelEnum, NeuronStyleEnum, WriteNoiseModelEnum, ParasiticCompensationEnum, XyceTimeSteppingEnum, UpdateModelEnum, ZeroResetPCEnum
