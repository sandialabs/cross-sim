#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from __future__ import annotations

import logging
from dataclasses import dataclass

from simulator.parameters.base import BaseParameters
from simulator.backend.registry import RegistryManager, register_subclasses

log = logging.getLogger(__name__)


@dataclass(repr=False)
class DeviceParameters(BaseParameters):
    """Parameters for a device composed of different read/program/drift models.

    Attributes:
        cell_bits: Programmable bit resolution of device conductance
        Rmin: Minimum programmable resistance of the device in ohms
        Rmax: Maximum programmable resistance of the device in ohms
        infinite_on_off_ratio: Whether to assume infinite conductance On/Off
            ratio. If True, simulates the case of infinite Rmax.
        clip_conductance: Whether to clip conductances between Gmin_norm and
            Gmax_norm. Defaults to False.
        read: Parameters for the device model used to simulate read errors.
        programming: Parameters for the device model used to simulate
            programming errors.
        drift: Parameters for the device model used to simulate drift errors.
    """

    cell_bits: int = 0
    Rmin: float = 1000
    Rmax: float = 10000
    time: int | float = 0
    infinite_on_off_ratio: bool = False
    clip_conductance: bool = False
    read_noise: DeviceModelParameters = None
    programming_error: DeviceModelParameters = None
    drift_error: DeviceModelParameters = None

    @property
    def Gmin_norm(self) -> float:
        """Returns the normalized minimum programmable device conductance.

        Normalization is performed dividing by the maximum programmable
        conductance.
        """
        gmin_norm = 0
        if not self.infinite_on_off_ratio:
            gmin_norm = self.Rmin / self.Rmax
        return gmin_norm

    @property
    def Gmax_norm(self) -> float:
        """Returns the normalized maximum programmable device conductance.

        Equal to 1 by definition.
        """
        return 1

    @property
    def Grange_norm(self) -> float:
        """Returns the normalized range of programmable device conductances.

        Normalization is performed using the max programmable resistance.
        """
        return self.Gmax_norm - self.Gmin_norm


@register_subclasses
@dataclass(repr=False)
class DeviceModelParameters(BaseParameters):
    """Parameters that describe device behavior.

    Attributes:
        enable: Flag to enable adding weight errors
        model: Name of device model to use. This must match the name of a child
            class of BaseDevice.
    """

    model: str = "IdealDevice"
    enable: bool = False

    def __new__(cls, *args, **kwargs):
        """Returns an unintialized instance of the class."""
        registry_manager = RegistryManager()
        key_name = "model"
        key_value = kwargs.get("model", cls.model)
        param_class = registry_manager.get_from_key(
            parent=DeviceModelParameters,
            key_name=key_name,
            key_value=key_value,
        )
        param = super().__new__(param_class)
        return param
