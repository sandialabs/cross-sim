#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""Interface for ADC objects.

This module provides the interface for Analog-to-Digital Converters (ADC)
and a few methods utility methods.
"""

from abc import ABC, abstractmethod
import warnings
import logging

import numpy.typing as npt

from simulator.parameters.adc import ADCRangeLimits, ADCParameters
from simulator.parameters.dac import DACParameters

from simulator.parameters.core import (
    CoreParameters,
)
from simulator.parameters.simulation import SimulationParameters
from simulator.backend.compute import ComputeBackend
from simulator.backend.registry import register_subclasses, RegistryManager

xp = ComputeBackend()  # Represents either cupy or numpy
log = logging.getLogger(__name__)


@register_subclasses
class IADC(ABC):
    """Interface for ADC objects."""

    # TODO: Add device params
    def __new__(
        cls,
        adc_params: ADCParameters,
        dac_params: DACParameters,
        core_params: CoreParameters,
        simulation_params: SimulationParameters,
        bitslice: int,
    ):
        """Creates an uninitialized ADC of type requested by the ADC parameters.

        Args:
            adc_params: Parameters to describe ADC model
            dac_params: Unused, forwarded to __init__.
            core_params: Unused, forwarded to __init__.
            simulation_params: Unused, forwarded to __init__.
            bitslice: Unused, forwarded to __init__.

        Raises:
            ValueError: Raised when an unknown ADC model is specified

        Returns:
            IADC: An unintialized object following the IADC interface.
        """
        registry_manager = RegistryManager()
        adc_types = registry_manager.get(cls)
        adc_types[cls.__name__] = cls
        adc_types.pop("IADC", None)

        try:
            log.info("Creating new IADC object (model=%s)", adc_params.model)
            adc_class = adc_types[adc_params.model]
            log.info("IADC class selected = %s", adc_class)
            adc = super().__new__(adc_class)
            return adc
        except KeyError as e:
            raise ValueError(
                f"Invalid ADC model selected. "
                f"Model must be either the base class or a subclass of {cls.__name__}. "
                "Either define a new ADC model or set model to one of the following: "
                f"{list(adc_types.keys())}",
            ) from e

    def __init__(
        self,
        adc_params: ADCParameters,
        dac_params: DACParameters,
        core_params: CoreParameters,
        simulation_params: SimulationParameters,
        bitslice: int,
    ) -> None:
        """Initializes an object with a ADC interface.

        Optimal tuning of an ADC may require additional infomation about
        the configuration of other hardware in the system. Therefore the
        DAC and core params must also be provided.

        Args:
            adc_params: Parameters to describe device behavior
            dac_params: DAC parameters used for "full precision guarantee"
            core_params: Core parameters used for "full precision guarantee"
            simulation_params: Additional parameters about the simulation
                configuration
            bitslice: Which bitslice the ADC belongs to.
        """
        super().__init__()
        self.bits = adc_params.bits
        self.signed_input = dac_params.signed
        self.adc_params = adc_params
        self.dac_params = dac_params
        self.simulation_params = simulation_params
        self.core_params = core_params
        self.core_type = core_params.core_type
        self.adc_per_ibit = adc_params.adc_per_ibit

        self.Nbits_in = self.dac_params.bits
        # TODO: Where to place weight_bits?
        #       Is the core params the right place?
        #       @Curtis, 2023-11-27
        self.Nbits_w = self.adc_params.weight_bits
        self.ymin = 1 / pow(2, self.Nbits_w)

        # Used in certain cases
        if self.Nbits_in > 0:
            # TODO: Error prone - Conditionally defined attribute
            # (@Curtis, 2023-10-10)
            if self.signed_input:
                self.corr = pow(2, self.Nbits_in - 1) / (pow(2, self.Nbits_in - 1) - 1)
            else:
                self.corr = pow(2, self.Nbits_in) / (pow(2, self.Nbits_in) - 1)

        # Set the bit slice range parameter
        if self.core_type == "BitslicedCore" and self.bits > 0:
            Wbits = self.adc_params.weight_bits
            Nslices = self.core_params.bit_sliced.num_slices

            # NOTE: Assumes all subcores of bitsliced core are of the same type.
            self.bitsliced_core_type = self.core_params.subcores[0].core_type
            if self.adc_params.adc_range_option == ADCRangeLimits.CALIBRATED:
                self.calibrated_range = self.adc_params.calibrated_range[bitslice]
                # If bit 0, check that the limits of different slices differ
                # by powers of 2
                if bitslice == 0:
                    _verify_calibrated_bitslice_range(
                        Nslices=Nslices,
                        calibrated_range=self.adc_params.calibrated_range,
                    )
            if Wbits % Nslices == 0:
                self.Wbits_slice = int(Wbits / Nslices)
            elif self.bitsliced_core_type == "SignedCore":
                self.Wbits_slice = xp.ceil((Wbits - 1) / Nslices).astype(int)
            else:
                self.Wbits_slice = xp.ceil(Wbits / Nslices).astype(int)

    @abstractmethod
    def convert(self, vector: npt.ArrayLike) -> npt.NDArray:
        """Converts a vector from analog value to digital values.

        Returns a vector converted from one containing analog values
        to one containing digital values.
        Converts with respect to the limits set for the dac.
        """
        raise NotImplementedError

    @abstractmethod
    def set_limits(self, matrix: npt.ArrayLike):
        """Given a matrix, sets the maximum possible limits for the adc.

        Based on core type and adc_range_option.
        The default configuration uses "calibrated" with adc_per_ibit = False

        Args:
            matrix: Matrix to use to set ADC limits.
        """
        ######################
        # NON-BITSLICED CORE #
        ######################

        if self.core_params.core_type != "BitslicedCore":
            if (
                self.adc_params.adc_range_option is ADCRangeLimits.MAX
            ):  # Handled BALANCED and OFFSET
                self.max = matrix.shape[1]
                if self.signed_input:
                    self.min = -matrix.shape[1]
                elif (
                    self.core_params.core_type == "SignedCore"
                    and self.core_params.subtract_in_xbar
                ):
                    self.min = -matrix.shape[1]
                else:
                    self.min = 0

            elif self.adc_params.adc_range_option is ADCRangeLimits.GRANULAR:
                if self.signed_input or (
                    self.core_params.core_type == "SignedCore"
                    and self.core_params.subtract_in_xbar
                ):
                    self.min = -self.ymin * (pow(2, self.bits - 1) - 1) * self.corr
                    self.max = self.ymin * (pow(2, self.bits - 1) - 1) * self.corr
                else:
                    self.min = 0
                    self.max = self.ymin * (pow(2, self.bits) - 1) * self.corr

            elif self.adc_params.adc_range_option is ADCRangeLimits.CALIBRATED:
                self.min = self.adc_params.calibrated_range[0]
                self.max = self.adc_params.calibrated_range[1]

        ##################
        # BITSLICED CORE #
        ##################

        else:
            # TODO:
            # This has flexibility to change when we actually write bitsliced
            if self.bitsliced_core_type == "OffsetCore":
                if self.adc_params.adc_range_option == ADCRangeLimits.MAX:
                    # Bring # rows to nearest power of 2
                    ymax = pow(2, xp.round(xp.log2(matrix.shape[1])))
                    # Correct to make level separation a multiple
                    # of the min cell current
                    ymax *= pow(2, self.Wbits_slice) / (pow(2, self.Wbits_slice) - 1)

                    if self.signed_input:
                        # Further correction to ensure level separation is a
                        # multiple of the min cell current
                        # I think the line below assumes SignbitADC
                        ymax *= (pow(2, self.bits) - 2) / pow(2, self.bits)
                        # Correct for input bits
                        ymax *= pow(2, self.Nbits_in - 1) / (
                            pow(2, self.Nbits_in - 1) - 1
                        )
                        self.min = -ymax
                        self.max = ymax

                    else:
                        # Further correction to ensure level separation is a
                        # multiple of the min cell current
                        ymax *= (pow(2, self.bits) - 1) / pow(2, self.bits)
                        # Correct for input bits
                        ymax *= pow(2, self.Nbits_in) / (pow(2, self.Nbits_in) - 1)
                        self.min = 0
                        self.max = ymax

                elif self.adc_params.adc_range_option == ADCRangeLimits.GRANULAR:
                    ymin = 1 / (pow(2, self.Wbits_slice) - 1)
                    if self.signed_input:
                        # Apply correction
                        ymin *= pow(2, self.Nbits_in - 1) / (
                            pow(2, self.Nbits_in - 1) - 1
                        )
                        self.min = -ymin * (pow(2, self.bits - 1) - 1)
                        self.max = ymin * (pow(2, self.bits - 1) - 1)
                    else:
                        ymin *= pow(2, self.Nbits_in) / (pow(2, self.Nbits_in) - 1)
                        self.min = 0
                        self.max = ymin * (pow(2, self.bits) - 1)

                elif self.adc_params.adc_range_option == ADCRangeLimits.CALIBRATED:
                    self.min = self.calibrated_range[0]
                    self.max = self.calibrated_range[1]

            elif self.bitsliced_core_type == "SignedCore":
                if self.adc_params.adc_range_option == ADCRangeLimits.MAX:
                    self.min = -matrix.shape[1]
                    self.max = matrix.shape[1]

                elif self.adc_params.adc_range_option == ADCRangeLimits.GRANULAR:
                    ymin = 1 / (pow(2, self.Wbits_slice) - 1)
                    if self.signed_input:
                        ymin *= pow(2, self.Nbits_in - 1) / (
                            pow(2, self.Nbits_in - 1) - 1
                        )
                    else:
                        ymin *= pow(2, self.Nbits_in) / (pow(2, self.Nbits_in) - 1)
                    self.min = -ymin * (pow(2, self.bits - 1) - 1)
                    self.max = ymin * (pow(2, self.bits - 1) - 1)

                elif self.adc_params.adc_range_option == ADCRangeLimits.CALIBRATED:
                    self.min = self.calibrated_range[0]
                    self.max = self.calibrated_range[1]

        self.range = self.max - self.min


# Utility function used in several custom ADC models
def get_digit(x: int, n: int) -> int:
    """Returns the binary digit of a number."""
    return (x // 2**n) % 2


def _verify_calibrated_bitslice_range(
    Nslices: int,
    calibrated_range: list[list[float]],
) -> bool:
    """Check that the limits of different slices differ by powers of 2."""
    ranges = xp.zeros(Nslices)
    for i in range(Nslices):
        ranges[i] = calibrated_range[i][1] - calibrated_range[i][0]
    range_ratios = xp.zeros(Nslices - 1)
    for i in range(Nslices - 1):
        range_ratios[i] = xp.log2(ranges[i + 1] / ranges[i])
    diff = xp.abs(range_ratios - xp.round(range_ratios))
    if any(diff > 1e-6):
        warnings.warn(
            "Warning: Bit sliced core ADC ranges do not differ by powers of 2!",
            stacklevel=2,
        )
        return False
    return True
