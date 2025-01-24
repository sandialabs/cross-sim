#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""CrossSim implementation of Keras layers.

This interface supports forward computation of Dense, and [1-3]d Convolutional layers
including depthwise convolutions. Forward operations support all CrossSim features.
All implemented layers support analog or digital bias additions and are fully
compatible with other digital layers. Conversion to and from Torch layers and profiling
 hooks are also provided.
"""

from .layer import AnalogLayer
from .dense import AnalogDense
from .conv import (
    AnalogConv1D,
    AnalogConv2D,
    AnalogConv3D,
    AnalogDepthwiseConv1D,
    AnalogDepthwiseConv2D,
)
from .convert import (
    to_keras,
    from_keras,
    convertible_layers,
    analog_layers,
    inconvertible_layers,
    reinitialize,
)
