#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from dataclasses import dataclass

from simulator.parameters.base import BaseParameters
from simulator.parameters.core import AnalogCoreParameters
from simulator.parameters.simulation import SimulationParameters
from simulator.parameters.xbar import GroupedXbarParameters


@dataclass(repr=False)
class CrossSimParameters(BaseParameters):
    """Parameters CrossSim.

    Attributes:
        core: Parameters that describe the core(s) used in the simulation.
        simulation: Non-physical simulation parameters
        xbar: Parameters that describe the behavior of the crossbar
    """

    core: AnalogCoreParameters = ("AnalogCore:default", "UnsignedCore")
    simulation: SimulationParameters = None
    xbar: GroupedXbarParameters = None

    def validate(self) -> None:
        """Validates the parameters."""
        self._recursive_validate(validate_self=False)
