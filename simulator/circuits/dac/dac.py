#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""Digital to analog conversions for MVM and VMM operations.

This module implements an object responsible for performing DAC operations
on vectors used in either MVM or VMM operations.
"""

import logging

import numpy as np
import numpy.typing as npt

import simulator.circuits.dac.models as models  # noqa: F401

from simulator.backend.compute import ComputeBackend
from simulator.circuits.dac.idac import IDAC
from simulator.parameters.dac import PairedDACParameters
from simulator.parameters.core import CoreParameters

log = logging.getLogger(__name__)
xp: np = ComputeBackend()


class DAC:
    """Digital to analog converter object.

    Creates an object that performs DAC operations for both MVM and VMM
    operations.
    """

    def __init__(self, dac_params: PairedDACParameters, core_params: CoreParameters):
        """Initializes an ADC using the provided parameters.

        Args:
            dac_params: Parameters to describe DAC behavior.
            core_params: Parameters for the core the DAC is acting on.
        """
        log.info("Creating DAC for MVM of type %s", dac_params.mvm.model)
        self.mvm = IDAC(dac_params.mvm)
        log.info("Creating DAC for VMM of type %s", dac_params.vmm.model)
        self.vmm = IDAC(dac_params.vmm)

    def set_limits(self, matrix: npt.ArrayLike):
        """Sets the limits for a DAC.

        Args:
            matrix: Matrix to set limits based off of.
        """
        matrix = xp.asarray(matrix)
        self.mvm.set_limits(matrix)
        self.vmm.set_limits(matrix.T)

    def convert(self, vector: npt.ArrayLike) -> npt.NDArray:
        """Converts a digital vector to an analog value.

        Args:
            vector: Vector to be converted.

        Returns:
            npt.NDArray: Analog version of vector.
        """
        # QUESTION:
        # Is this an antipattern?
        # Having this warning in this function just makes it more likely to call
        # it via autocomplete.
        #
        # Might be better to not have these functions and have this class be
        # named PairedDAC, or similar.
        # @Curtis, 2024-01-25
        raise NotImplementedError(
            "Attemping to call DAC.convert directly, "
            "call mvm.convert or vmm.convert for the DAC model.",
        )

    def convert_sliced(self, vector: npt.ArrayLike, slice_size: int = 1) -> npt.NDArray:
        """Converts a sliced vector.

        Args:
            vector: Vector to be converted.
            slice_size: Size of slice to be converted.

        Raises:
            NotImplementedError: Should be called on dac.mvm or dac.vmm

        Returns:
            npt.NDArray: Analog version of the slice.
        """
        # QUESTION:
        # Is this an antipattern?
        # Having this warning in this function just makes it more likely to call
        # it via autocomplete.
        #
        # Might be better to not have these functions and have this class be
        # named PairedDAC, or similar.
        # @Curtis, 2024-01-25
        raise NotImplementedError(
            "Attemping to call DAC.convert_sliced directly, "
            "call mvm.convert_sliced or vmm.convert_sliced for the DAC model.",
        )
