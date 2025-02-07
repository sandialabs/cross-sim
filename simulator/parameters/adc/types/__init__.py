#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
"""Defines parameters related to Pipeline and Cyclic ADC implementations."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

from .IdealADCParameters import IdealADCParameters
from .CyclicADCParameters import CyclicADCParameters
from .PipelineADCParameters import PipelineADCParameters
from .RampADCParameters import RampADCParameters
from .SarADCParameters import SarADCParameters
