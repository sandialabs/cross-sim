#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import logging

import numpy as np
import numpy.typing as npt

from simulator.circuits.adc.iadc import IADC
from simulator.backend import ComputeBackend

xp: np = ComputeBackend()
log = logging.getLogger(__name__)


class IdealADC(IADC):
    """An ideal ADC which performs perfect conversions."""

    def convert(self, vector: npt.ArrayLike) -> npt.NDArray:
        """Ideal case, converting is the identity function.

        Args:
            vector: Value to convert.

        Returns:
            Returns an array of the same value.
        """
        return xp.asarray(vector)

    def set_limits(self, matrix: npt.ArrayLike):
        """Sets limits for the ADC.

        Args:
            matrix: Unused in ideal case.
        """
        pass
