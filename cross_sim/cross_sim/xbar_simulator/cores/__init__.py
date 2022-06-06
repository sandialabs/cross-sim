#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

'''
Cores used to simulate and model memristive crossbars and peripheral circuitry

Algorithm -> Wrapper Core (OffsetCore, BalancedCore, or PosNegCore) -> Clipper Core (XyceCore or NumericCore)


Data passing:
-------------

1 Sent from algorithm to WrapperCore

  * Scaled from outer [algorithm's] dynamic range (e.g.: -10,10) to wrapper's dynamic range (e.g.: -0.5,0.5)

2 Sent to OffsetCore/BalancedCore/PosNegcore/...

  * Converted to internal representation as needed to be sent to the inner core(s) (e.g.: positive values to positive core; negative values to negative core)
    (Note that conversion to the inner dynamic range (to 0,1) is a side-effect of this process)

3 Sent to ClipperCore

  * Clipping and quantization

4 Sent to XyceCore/NumericCore

  * Data is used


5 Returned to ClipperCore

  * Output is clipped and quantized
    *for VMM and MVM, clipping/quantization is done in offset/balanced core so the multiple outputs can be combined before clipping/quantization

6 Returned to OffsetCore/...

  * Output from core(s) is combined/post-processed (e.g.: subtraction for OffsetCore)

7 Returned to WrapperCore

  * Output is scaled back to outer dynamic range [as needed by the algorithm]

8 Output arrives back at algorithm

'''

import copy
from warnings import warn

from .icore import ICore
from .numeric_core import NumericCore
# from .xyce_core import XyceCore
from .clipper_core import ClipperCore
from .wrapper_core import WrapperCore
from .offset_core import OffsetCore
from .balanced_core import BalancedCore
from .memory_core import MemoryCore
from .convolution_core import ConvolutionCore
from ..parameters.all_parameters import change_param_root
from ..parameters import Parameters
from .core_initialization import verify_parameters, MakeCore2
try: # import if available / included in release
    from .pc_core import PeriodicCarryCore
except ImportError:
    pass

from ..parameters.parameter_defaults import CrossbarTypeEnum, SimTypeEnum


def MakeCore(params, inner=None, outer=None, dirname=None, Ncores=1):
    """
    Generate an :py:class:`.ICore`

    :param inner: DEPRECATED: use params setting.  The inner core to use: "Numeric" or "Xyce" a string is used as this is part of the API
    :type inner: str
    :param outer: DEPRECATED: use params setting.  The outer core to use: "Offset" or "Balanced" a string is used as this is part of the API
    :type outer: str

    :param params: All parameters
    :type params: Parameters

    :param dirname:  DEPRECATED: output directory for Xyce files

    :return: An outer core initialized with the appropriate inner core and parameters
    :rtype: ICore or WrapperCore
    """

    # create local copy of the parameters for the core
    params = params.copy()

    ### DEPRECATED
    # set the inner core type
    if inner is not None:
        if inner == "numeric" or inner=="NUMERIC":
            params.algorithm_params.sim_type=SimTypeEnum.NUMERIC
        else:
            raise ValueError("Inner core type "+str(inner)+" is unknown")

    ### DEPRECATED
    # set outer core type
    if outer is not None:
        if outer == "offset" or outer=="OFFSET":
            params.algorithm_params.crossbar_type = CrossbarTypeEnum.OFFSET
        elif outer == "balanced" or outer=="BALANCED":
            params.algorithm_params.crossbar_type = CrossbarTypeEnum.BALANCED
        elif outer == "bitscliced" or outer=="BITSLICED":
            params.algorithm_params.crossbar_type = CrossbarTypeEnum.BITSLICED
        elif outer =="memory" or outer=="MEMORY":
            params.algorithm_params.crossbar_type = CrossbarTypeEnum.MEMORY
        else:
            raise ValueError("Outer core type "+str(outer)+" is unknown should be offset, balanced or memory")

    # if dirname is not None:
        # params.xyce_parameters.out_dir=dirname

    # if using periodic carry core, return a periodic carry core instead of a wrapper core
    if params.periodic_carry_params.use_periodic_carry==True:
        verify_parameters(params)
        return PeriodicCarryCore(params)
    elif params.convolution_parameters.is_conv_core == True:
        verify_parameters(params)
        return ConvolutionCore(params,Ncores=Ncores)
    else:
        return MakeCore2(params)


def MakeMultiCore(params, inner=None, outer=None, dirname=None):
    """
    Generate an :py:class:`.ICore`

    Creates a special type of core that holds multiple nueral cores each with a unique params object
    Used to support splitting a large MVM into multiple arrays
    09/19/2020: Supportd only for ConvolutionCore

    :return: An outer core initialized with the appropriate inner core and parameters
    :rtype: ICore or WrapperCore
    """

    # create local copy of the parameters for the core

    if type(params) is not list:
        raise ValueError("MakeMultiCore cannot be called with a single params object")
    else:
        Ncores = len(params)

    paramsLocal = [None]*Ncores
    for k in range(Ncores):
        paramsLocal[k] = params[k].copy()
    params = paramsLocal

    for k in range(Ncores):
        verify_parameters(params[k])

    if params[0].convolution_parameters.is_conv_core:
        return ConvolutionCore(params,Ncores=Ncores)
    else:
        raise ValueError("MakeMultiCore should only be called for convolutional cores")