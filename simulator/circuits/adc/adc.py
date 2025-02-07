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

# Importing the different adc models so _create_adc works correctly
import simulator.circuits.adc.models as models  # noqa: F401

from simulator.circuits.adc.iadc import IADC
from simulator.backend.compute import ComputeBackend
from simulator.parameters.adc import PairedADCParameters
from simulator.parameters.dac import PairedDACParameters
from simulator.parameters.core import CoreParameters
from simulator.parameters.simulation import SimulationParameters

xp: np = ComputeBackend()
log = logging.getLogger(__name__)


class ADC:
    """Analog to digital converter object.

    Creates an object that performs ADC operations for both MVM and VMM
    operations.
    """

    def __init__(
        self,
        adc_params: PairedADCParameters,
        dac_params: PairedDACParameters,
        core_params: CoreParameters,
        simulation_params: SimulationParameters,
        bitslice: int = 0,
    ):
        """Initializes an ADC using the provided parameters.

        Optimal tuning of an ADC may require additional infomation about
        the configuration of other hardware in the system. Therefore the
        DAC and core params must also be provided.

        Args:
            adc_params: Parameters to describe device behavior
            dac_params: DAC parameters used for "full precision guarantee"
            core_params: Core parameters used for "full precision guarantee"
            simulation_params: Additional parameters about the simulation
                configuration
            bitslice: Which bitslice the ADC belongs to. Defaults to 0.
        """
        log.info("Creating ADC for MVM of type %s", adc_params.mvm.model)
        self.mvm = IADC(
            adc_params=adc_params.mvm,
            dac_params=dac_params.mvm,
            core_params=core_params,
            simulation_params=simulation_params,
            bitslice=bitslice,
        )
        log.info("Creating ADC for VMM of type %s", adc_params.vmm.model)
        self.vmm = IADC(
            adc_params=adc_params.vmm,
            dac_params=dac_params.vmm,
            core_params=core_params,
            simulation_params=simulation_params,
            bitslice=bitslice,
        )

    def set_limits(self, matrix: npt.ArrayLike):
        """Sets the limits of the ADC for both MVM and VMM operations.

        Args:
            matrix: Matrix to set limit value with.
        """
        matrix = xp.asarray(matrix)
        self.mvm.set_limits(matrix)
        self.vmm.set_limits(matrix.T)

    def convert(self, vector: npt.ArrayLike) -> npt.NDArray:
        """Converts values from analog to digital.

        Expects that set_limits() has been called before.

        Args:
            vector: Vector to be converted.

        Raises:
            NotImplementedError: Raised when calling directly on the ADC.

        Returns:
            npt.NDArray: Digital version of the provided analog vector.
        """
        raise NotImplementedError(
            "Attemped to call ADC.convert directly."
            "Call mvm.convert or vmm.convert for the ADC model.",
        )
