#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

#  This file contains functions used to intialize and error check a new core

from warnings import warn

from ..parameters import Parameters
from .numeric_core import NumericCore
from .offset_core import OffsetCore
from .balanced_core import BalancedCore
from .bitsliced_core import BitslicedCore
from .memory_core import MemoryCore
from .wrapper_core import WrapperCore
from .hardware_core import HardwareCore
from .icore import ICore
from ..parameters.parameter_defaults import ZeroResetPCEnum, UpdateModelEnum, CrossbarTypeEnum, SimTypeEnum
import numpy as np


def verify_parameters(params):
    """
    performs validation checks on the final set of parameters.  Do validation here to avoid repeated checks.  Runs manual post sets before validation
    :param params:
    :type params: Parameters
    :param inner:
    :return:
    """

    ###### place post sets here if needed
    params.convolution_parameters.post_set()

    # run any manual post sets:
    # RunManualPostSets(params) # manual_post_sets cannot depend on value of xbar_params.weights
    params.numeric_params.nonlinearity.manual_post_set()
    params.numeric_params.dG_lookup.manual_post_set()
    params.weight_error_params.manual_post_set()
    
    # check nonlinearity parameter
    if not (params.numeric_params.nonlinearity.alpha>=0):
        raise ValueError("The nonlinearity numeric parameter must be >=0")

    # check weights
    if params.xbar_params.weights.maximum != 1:
        raise ValueError("The max xbar weight must be 1 not "+str(params.xbar_params.weights.maximum))


    if params.algorithm_params.weights.maximum != -params.algorithm_params.weights.minimum:
        if (params.algorithm_params.weights.minimum==0 and params.algorithm_params.weights.maximum>0):
            if params.algorithm_params.crossbar_type!=CrossbarTypeEnum.OFFSET:
                raise ValueError("zero min algorithm weight only compatible with offset core")
        else:
            raise ValueError("Bad algorithm weights: must have max = - min or (max>0 & min=0)")

    # check lookup table parameters
    if params.numeric_params.dG_lookup.Gmin_relative<0 or \
        params.numeric_params.dG_lookup.Gmin_relative>=params.numeric_params.dG_lookup.Gmax_relative:
        raise ValueError("Lookup table paramter Gmin_relative must be > 0 and less than Gmax_relative")
    if params.numeric_params.dG_lookup.Gmin_relative>1:
        raise ValueError("Lookup table paramter Gmax_relative must be < 1 and greater than Gmin_relative")


    #### check periodic carry params
    if params.periodic_carry_params.cores_per_weight-1 != len(params.periodic_carry_params.carry_frequency):
        raise ValueError("The periodic carry fequency should be a list the same length as cores_per_weight")

    # if params.periodic_carry_params.zero_reset == ZeroResetPCEnum.CALIBRATED:
    #     if params.numeric_params.update_model != UpdateModelEnum.DG_LOOKUP:
    #         raise NotImplementedError ("The calibrated periodic carry is only implemented with the lookup table model")


def MakeCore2(params):
    """
    Creates the inner and outer cores.  A separate top level function is needed in case a periodic carry is set.


    :param params: All parameters
    :type params: Parameters

    :return: An outer core initialized with the appropriate inner core and parameters
    :rtype: WrapperCore
    :rtype: ICore or WrapperCore
    """

    # run checks for parameter validity (and run manual post sets) (re-run if using periodic carry)
    verify_parameters(params)

    if params.algorithm_params.sim_type==SimTypeEnum.NUMERIC:
        def inner_factory():
            return NumericCore(params)
        def inner_factory_independent():
            new_params = params.copy()
            verify_parameters(new_params)
            return NumericCore(new_params)
    else:
        raise ValueError("Inner core type "+str(params.algorithm_params.sim_type)+" is unknown")

    # set the outer core type
    if params.algorithm_params.crossbar_type==CrossbarTypeEnum.OFFSET:
        return OffsetCore(inner_factory, params)
    
    elif params.algorithm_params.crossbar_type==CrossbarTypeEnum.BALANCED:
        # If using multiple lookup table with balanced core, each of the two arrays needs its own params object to store
        # independent assignment matrices
        if params.numeric_params.update_model == UpdateModelEnum.DG_LOOKUP and params.numeric_params.dG_lookup.multi_LUT:
            return BalancedCore(inner_factory_independent, params)
        else:
            return BalancedCore(inner_factory, params)

    elif params.algorithm_params.crossbar_type==CrossbarTypeEnum.BITSLICED:
        return BitslicedCore(inner_factory_independent, params)

    else:
        raise ValueError("Outer core type "+str(params.algorithm_params.crossbar_type)+" is unknown should be OFFSET or BALANCED")

