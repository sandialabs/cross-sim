#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from abc import ABC, abstractmethod


class IDAC(ABC):
    def __init__(self, dac_params, core_params) -> None:
        super().__init__()
        self.core_params = core_params
        self.bits = dac_params.bits
        self.signed = dac_params.signed
        self.slice_size = dac_params.slice_size

    @abstractmethod
    def convert(self, vector):
        """Returns a vector converted from one containing digital values
        to one containing analog values.
        Converts with respect to the limits set for the dac.
        """
        raise NotImplementedError

    @abstractmethod
    def convert_sliced(self, vector, slice_size):
        """Returns a list of vectors of analog slice values converted from one containing
        digital values
        Converts with respect to the limits set for the dac.
        """
        raise NotImplementedError

    @abstractmethod
    def set_limits(self, matrix):
        """Sets the limits of the DAC. May depend on the input matrix but not required."""
        raise NotImplementedError

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
            root = IDAC
        subclasses = []
        for subclass in root.__subclasses__():
            subclasses.append(subclass)
            subclasses.extend(subclass.get_all_subclasses(use_base=False))
        return subclasses
