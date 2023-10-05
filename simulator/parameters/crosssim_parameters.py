#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from dataclasses import dataclass

from .base_parameters import BaseParameters
from .core_parameters import CoreParameters
from .simulation_parameters import SimulationParameters
from .xbar_parameters import XbarParameters


@dataclass(repr=False)
class CrossSimParameters(BaseParameters):
    """Parameters CrossSim.

    Attributes:
        core (CoreParameters): Parameters that describe the core(s) used in the
            simulation.
        simulation (SimulationParameters): Non-physical simulation parameters
        xbar (XbarParameters): Parameters that describe the behavior of the crossbar
    """

    core: CoreParameters = None
    simulation: SimulationParameters = None
    xbar: XbarParameters = None
