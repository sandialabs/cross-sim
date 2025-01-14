#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""CrossSim implementation of PyTorch layers.

This interface supports forward and backward computation of Linear, and [1-3]d
Convolutional layers. Forward operations support all CrossSim features, backward
operations are fully ideal. All implemented layers support analog or digital bias
additions and are fully compatible with other digital layers. Conversion to and from
Torch layers and profiling hooks are also provided.
"""

from .layer import AnalogLayer
from .linear import AnalogLinear
from .conv import AnalogConv1d, AnalogConv2d, AnalogConv3d
from .convert import (
    to_torch,
    from_torch,
    convertible_modules,
    analog_modules,
    inconvertible_modules,
    synchronize,
    reinitialize,
)
