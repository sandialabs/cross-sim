#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy.typing as npt

from simulator.backend.registry import RegistryManager, register_subclasses
from simulator.parameters.dac import DACParameters

log = logging.getLogger(__name__)


@dataclass
class _InputBitslice:
    """Data from input bitslice outputs.

    Attributes:
        islice: Analog value of the input bitslice.
        idx: Index of the most significant bit of the input slice.
        correction_factor: Factor to multiply output by after ADC.
    """

    islice: npt.NDArray
    idx: int
    correction_factor: float

    @property
    def value(self):
        return self.islice * self.correction_factor * 2**self.idx


_InputBitslices = Iterator[_InputBitslice]


@register_subclasses
class IDAC(ABC):
    """Interface for digital to analog converts."""

    def __new__(
        cls,
        dac_params: DACParameters,
    ):
        """Creates an uninitialized DAC of type requested by the DAC parameters.

        Args:
            dac_params: Parameters to describe DAC model
            input_mapping_params: Unused, forwarded to __init__.

        Raises:
            ValueError: Raised when an unknown DAC model is specified

        Returns:
            IDAC: An unintialized object following the IDAC interface.
        """
        registry_manager = RegistryManager()
        dac_types = registry_manager.get(cls)
        dac_types[cls.__name__] = cls
        dac_types.pop("IDAC", None)

        try:
            log.info("Creating new IDAC object (model=%s)", dac_params.model)
            dac_class = dac_types[dac_params.model]
            log.info("IDAC class selected = %s", dac_class)
            dac = super().__new__(dac_class)
            return dac
        except KeyError as e:
            raise ValueError(
                f"Invalid DAC model selected. "
                f"Model must be either the base class or a subclass of {cls.__name__}. "
                "Either define a new DAC model or set model to one of the following: "
                f"{list(dac_types.keys())}",
            ) from e

    def __init__(
        self,
        dac_params: DACParameters,
    ):
        """Initializes an object using the DAC interface.

        Args:
            dac_params: Parameters for the DAC
            input_mapping_params: Parameters of inputs the DAC will use.
        """
        super().__init__()
        self.input_mapping_params = dac_params.mapping
        self.bits = dac_params.bits
        self.signed = dac_params.signed
        self.slice_size = dac_params.slice_size

    @abstractmethod
    def convert(self, vector: npt.ArrayLike) -> npt.NDArray:
        """Converts a vector from digital value to analog values.

        Returns a vector converted from one containing digital values
        to one containing analog values.
        Converts with respect to the limits set for the dac.
        """
        raise NotImplementedError

    @abstractmethod
    def convert_sliced(
        self,
        vector: npt.ArrayLike,
        slice_size: int | None = None,
    ) -> Iterator[_InputBitslices]:
        """Returns an iterator that converts slices of digital values to analog.

        Converts with respect to the limits set for the dac.

        Args:
            vector: Vector to be converted.
            slice_size: Size of slice to be converted.

        Returns:
            Iterator[_InputBitslice]: Output of the input bitslice slicing.
        """
        raise NotImplementedError

    @abstractmethod
    def set_limits(self, matrix: npt.ArrayLike):
        """Sets the limits of the DAC."""
        raise NotImplementedError
