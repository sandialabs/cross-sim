#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from abc import ABC, abstractmethod
import warnings
from simulator.parameters.xbar_parameters import ADCRangeLimits
from simulator.parameters.core_parameters import CoreStyle, BitSlicedCoreStyle
from simulator.backend import ComputeBackend

xp = ComputeBackend()  # Represents either cupy or numpy


class IADC(ABC):
    def __init__(
        self,
        adc_params,
        dac_params,
        core_params,
        simulation_params,
        bitslice,
    ) -> None:
        super().__init__()
        self.bits = adc_params.bits
        self.signed_input = dac_params.signed
        self.adc_params = adc_params
        self.dac_params = dac_params
        self.simulation_params = simulation_params
        self.core_params = core_params
        self.core_type = core_params.style
        self.adc_per_ibit = adc_params.adc_per_ibit

        self.Nbits_in = self.dac_params.bits
        self.Nbits_w = self.core_params.weight_bits
        self.ymin = 1 / pow(2, self.Nbits_w)

        # Used in certain cases
        if self.Nbits_in > 0:
            if self.signed_input:
                self.corr = pow(2, self.Nbits_in - 1) / (pow(2, self.Nbits_in - 1) - 1)
            else:
                self.corr = pow(2, self.Nbits_in) / (pow(2, self.Nbits_in) - 1)

        # Set the bit slice range parameter
        if self.core_type == CoreStyle.BITSLICED and self.bits > 0:
            Wbits = self.core_params.weight_bits
            Nslices = self.core_params.bit_sliced.num_slices

            self.bitsliced_core_type = self.core_params.bit_sliced.style
            if self.adc_params.adc_range_option == ADCRangeLimits.CALIBRATED:
                self.calibrated_range = self.adc_params.calibrated_range[bitslice]
                # If bit 0, check that the limits of different slices differ by powers of 2
                if bitslice == 0:
                    ranges = xp.zeros(Nslices)
                    for i in range(Nslices):
                        ranges[i] = (
                            self.adc_params.calibrated_range[i][1]
                            - self.adc_params.calibrated_range[i][0]
                        )
                    range_ratios = xp.zeros(Nslices - 1)
                    for i in range(Nslices - 1):
                        range_ratios[i] = xp.log2(ranges[i + 1] / ranges[i])
                    diff = xp.abs(range_ratios - xp.round(range_ratios))
                    if any(diff > 1e-6):
                        warnings.warn(
                            "Warning: Bit sliced core ADC ranges do not differ by powers of 2!",
                        )

            if Wbits % Nslices == 0:
                self.Wbits_slice = int(Wbits / Nslices)
            elif self.bitsliced_core_type == BitSlicedCoreStyle.BALANCED:
                self.Wbits_slice = xp.ceil((Wbits - 1) / Nslices).astype(int)
            else:
                self.Wbits_slice = xp.ceil(Wbits / Nslices).astype(int)

    @abstractmethod
    def convert(self, vector):
        """Returns a vector converted from one containing analog values
        to one containing digital values.
        Converts with respect to the limits set for the adc.
        """
        raise NotImplementedError

    @abstractmethod
    def set_limits(self, matrix):
        """Given a matrix, sets the maximum possible limits for the adc
        based on core type and adc_range_option.

        The default option is "calibrated" with adc_per_ibit = False
        """
        ###########################
        #### NON-BITSLICED CORE
        ###########################

        if self.core_type != CoreStyle.BITSLICED:
            if (
                self.adc_params.adc_range_option is ADCRangeLimits.MAX
            ):  # Handled BALANCED and OFFSET
                self.max = matrix.shape[1]
                if self.signed_input:
                    self.min = -matrix.shape[1]
                elif (
                    self.core_type == CoreStyle.BALANCED
                    and self.core_params.balanced.subtract_current_in_xbar
                ):
                    self.min = -matrix.shape[1]
                else:
                    self.min = 0

            elif self.adc_params.adc_range_option is ADCRangeLimits.GRANULAR:
                if self.signed_input or (
                    self.core_type == CoreStyle.BALANCED
                    and self.core_params.balanced.subtract_current_in_xbar
                ):
                    self.min = -self.ymin * (pow(2, self.bits - 1) - 1) * self.corr
                    self.max = self.ymin * (pow(2, self.bits - 1) - 1) * self.corr
                else:
                    self.min = 0
                    self.max = self.ymin * (pow(2, self.bits) - 1) * self.corr

            elif self.adc_params.adc_range_option is ADCRangeLimits.CALIBRATED:
                self.min = self.adc_params.calibrated_range[0]
                self.max = self.adc_params.calibrated_range[1]

        ###########################
        #### BITSLICED CORE
        ###########################

        else:
            if self.bitsliced_core_type == BitSlicedCoreStyle.OFFSET:
                if self.adc_params.adc_range_option == ADCRangeLimits.MAX:
                    # Bring # rows to nearest power of 2
                    ymax = pow(2, xp.round(xp.log2(matrix.shape[1])))
                    # Correct to make level separation a multiple of the min cell current
                    ymax *= pow(2, self.Wbits_slice) / (pow(2, self.Wbits_slice) - 1)

                    if self.signed_input:
                        # Further correction to ensure level separation is a multiple of the min cell current
                        # I think the line below assumes SignbitADC
                        ymax *= (pow(2, self.bits) - 2) / pow(2, self.bits)
                        # Correct for input bits
                        ymax *= pow(2, self.Nbits_in - 1) / (
                            pow(2, self.Nbits_in - 1) - 1
                        )
                        self.min = -ymax
                        self.max = ymax

                    else:
                        # Further correction to ensure level separation is a multiple of the min cell current
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

            elif self.bitsliced_core_type == BitSlicedCoreStyle.BALANCED:
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

    @classmethod
    def get_all_subclasses(cls: type, use_base=True) -> list[type]:
        """Returns all subclasses of a type
        Args:
            cls (type): Type to get subclasses of, ignored if use_base is True
            use_base (bool, optional): If set all subclasses of the ADC are returned as
                opposed to of the current type. Defaults to True.

        Returns:
            list[type]: A list of all BaseDevice classes.
        """
        root = cls
        if use_base:
            root = IADC
        subclasses = []
        for subclass in root.__subclasses__():
            subclasses.append(subclass)
            subclasses.extend(subclass.get_all_subclasses(use_base=False))
        return subclasses


# Utility function used in several custom ADC models
def get_digit(x, n):
    return (x // 2**n) % 2
