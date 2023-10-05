#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

# """
# Cores used to simulate and model memristive crossbars and peripheral circuitry

# Algorithm -> Wrapper Core (OffsetCore, BalancedCore, or BitslicedCore) -> Clipper Core (NumericCore)


# Data passing:
# -------------

# 1 Sent from AnalogCore to WrapperCore
#   - Scaled from outer [algorithm's] dynamic range (e.g.: -10,10) to wrapper's dynamic range (e.g.: -0.5,0.5)
# 2 Sent to OffsetCore/BalancedCore/BitslicedCore/...
#   - Converted to internal representation as needed to be sent to the inner core(s) (e.g.: positive values to positive core; negative values to negative core)
#     (Note that conversion to the inner dynamic range (to 0,1) is a side-effect of this process)
# 3 Sent to ClipperCore
#   - Clipping and quantization
# 4 Sent to NumericCore
#   - Data is used
# 5 Returned to ClipperCore
#   - Output is clipped and quantized for VMM and MVM, clipping/quantization is done in offset/balanced core so the multiple outputs can be combined before clipping/quantization
# 6 Returned to OffsetCore/BalancedCore/BitslicedCore
#   - Output from core(s) is combined/post-processed (e.g.: subtraction for OffsetCore)
# 7 Returned to WrapperCore
#   - Output is scaled back to outer dynamic range [as needed by the algorithm]
# 8 Output arrives back at algorithm
# """

import copy
from warnings import warn

from .icore import ICore
from .numeric_core import NumericCore
from .wrapper_core import WrapperCore
from .offset_core import OffsetCore
from .balanced_core import BalancedCore
from .bitsliced_core import BitslicedCore
from .analog_core import AnalogCore
from simulator.parameters import CrossSimParameters

try:  # import if available / included in release
    from .pc_core import PeriodicCarryCore
except ImportError:
    pass
