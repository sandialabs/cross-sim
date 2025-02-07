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

from simulator.parameters.device.device import (
    DeviceModelParameters,
)

log = logging.getLogger(__name__)


@dataclass(repr=False)
class GenericDeviceParameters(DeviceModelParameters):
    """Parameters for generic devices.

    Args:
        model: The class of device this parameter is for.
        magnitude: Model parameter for the generic model.
    """

    model: str = "GenericDevice"
    magnitude: float = 0


@dataclass(repr=False)
class NormalIndependentDeviceParameters(DeviceModelParameters):
    """Parameters for a generic normal independent device.

    Attributes:
        model: The class of device this parameter is for.
        magnitude: Standard deviation of the random conductance error that is
            applied either as programming error or read noise when using one of
            the generic device models. This is normalized either to the maximum
            device conductance
    """

    model: str = "NormalIndependentDevice"
    magnitude: float = 0


@dataclass(repr=False)
class NormalProportionalDeviceParameters(DeviceModelParameters):
    """Parameters for a generic normal proportional device.

    Attributes:
        model: The class of device this parameter is for.
        magnitude: Standard deviation of the random conductance error that is
            applied either as programming error or read noise when using one of
            the generic device models. This is normalized either to the target
            device conductance
    """

    model: str = "NormalProportionalDevice"
    magnitude: float = 0


@dataclass(repr=False)
class NormalInverseProportionalDeviceParameters(DeviceModelParameters):
    """Parameters for a generic normal inverse proportional device.

    Attributes:
        model: The class of device this parameter is for.
        magnitude: Standard deviation of the random conductance error that is
            applied either as programming error or read noise when using one of
            the generic device models.
    """

    model: str = "NormalInverseProportionalDevice"
    magnitude: float = 0


@dataclass(repr=False)
class UniformIndependentDeviceParameters(DeviceModelParameters):
    """Parameters for a generic uniform independent device.

    Attributes:
        model: The class of device this parameter is for.
        magnitude: Standard deviation of the random conductance error that is
            applied either as programming error or read noise when using one of
            the generic device models.
    """

    model: str = "UniformIndependentDevice"
    magnitude: float = 0


@dataclass(repr=False)
class UniformProportionalDeviceParameters(DeviceModelParameters):
    """Parameters for a generic uniform proportional device.

    Attributes:
        model: The class of device this parameter is for.
        magnitude: Standard deviation of the random conductance error that is
            applied either as programming error or read noise when using one of
            the generic device models.
    """

    model: str = "UniformProportionalDevice"
    magnitude: float = 0


@dataclass(repr=False)
class UniformInverseProportionalDeviceParameters(DeviceModelParameters):
    """Parameters for a generic uniform inverse proportional device.

    Attributes:
        model: The class of device this parameter is for.
        magnitude: Standard deviation of the random conductance error that is
            applied either as programming error or read noise when using one of
            the generic device models.
    """

    model: str = "UniformInverseProportionalDevice"
    magnitude: float = 0
