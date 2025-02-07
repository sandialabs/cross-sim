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

from simulator.devices.idevice import IDevice

from simulator.backend import ComputeBackend, RegistryManager, register_subclasses
from simulator.parameters.device import (
    DeviceParameters,
    DeviceModelParameters,
)

log = logging.getLogger(__name__)
xp: np = ComputeBackend()


@register_subclasses
class BaseDevice(IDevice):
    """Base class for a device model that is compliant with IDevice.

    As some device models may not have an implementation for each error type,
    this class adds default behavior of raising an exception for non-implemented
    methods.
    """

    def __new__(
        cls,
        device_params: DeviceParameters,
        model_params: DeviceModelParameters,
    ):
        """Creates an uninitialized Device requested by the Device parameters.

        Args:
            device_params: Parameters used to create the device.
            model_params: Unused, forwarded to __init__()

        Returns:
            BaseDevice: An unintialized BaseDevice object following the IDevice
                interface.
        """
        registry_manager = RegistryManager()
        device_types = registry_manager.get(cls)
        device_types[cls.__name__] = cls

        # Remove known invalid device types
        device_types.pop("BaseDevice", None)
        device_types.pop("GenericDevice", None)

        try:
            log.info("Creating new IDevice object (model=%s)", model_params.model)
            device_class = device_types[model_params.model]
            log.info("IDevice class selected = %s", device_class)
            device = super().__new__(device_class)
            return device
        except KeyError as e:
            raise ValueError(
                f"Invalid Device model selected. "
                f"Model must be either the base class or a subclass of {cls.__name__}. "
                "Either define a new Device or set model to one of the following: "
                f"{list(device_types.keys())}",
            ) from e

    def __init__(
        self,
        device_params: DeviceParameters,
        model_params: DeviceModelParameters,
    ):
        """Initialize device types, unpack device parameters into attributes.

        Args:
            device_params: Common device parameters.
            model_params: Parameters specific to the device model.
        """
        super().__init__()
        self.device_params = device_params
        self.model_params = model_params
        self.clip_conductance = device_params.clip_conductance
        self.cell_bits = device_params.cell_bits
        self.time = device_params.time

        # TODO: Currently unused
        self.initial_time = 0

    def read_noise(self, input_: npt.ArrayLike) -> npt.NDArray:
        """Returns a noisy version of a matrix after reading.

        Read noise represents error associated with the device conductances
        (the weights in the matrix) at the time of the MVM (e.g., the
        temperature could effect the conductance).

        The read noise is re-sampled on every MVM rather than at write time, as
        a result the implementation can become a bottleneck, especially on GPUs.

        Args:
            input_: The noiseless version of the conductance matrix.

        Returns:
            npt.NDArray: The noisy version of the conductance matrix after read.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented read noise",
        )

    def programming_error(self, input_: npt.ArrayLike) -> npt.NDArray:
        """Returns a noisy version of the conductance matrix after programming.

        Error that represents the inaccuracy of actually setting the
        conductances within the AnalogCore. The error is sampled when the matrix
        is set and so does not effect individual MVMs differently.

        Args:
            input_: The noiseless version of the conductance matrix.

        Returns:
            npt.NDArray: The noisy version of the conductance matrix after
                programming.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented programming error",
        )

    def drift_error(self, input_: npt.ArrayLike) -> npt.NDArray:
        """Returns a noisy version of the condutance matrix after drift error.

        Error representing the drift in the conductance of matrix over time.
        The drift model is assumed to be a complete model of device errors at
        the set time, i.e. includes the effect of programming errors.

        If time > 0, programming errors are not applied separately from drift

        Args:
            input_: The noiseless version of the conductance matrix.

        Returns:
            npt.NDArray: The noisy version of the conductance matrix after
                drift over time.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented drift error",
        )


# Ideal device
class IdealDevice(BaseDevice):
    """An ideal device which doesn't make errors."""

    def read_noise(self, input_: npt.ArrayLike) -> npt.NDArray:
        """Returns the input after an ideal read.

        Read noise represents error associated with the device conductances
        (the weights in the matrix) at the time of the MVM (e.g., the
        temperature could effect the conductance).

        Args:
            input_: The input to be read

        Returns:
            npt.NDArray: An identical numpy-like version of the input.
        """
        return xp.asarray(input_)

    def programming_error(self, input_: npt.ArrayLike) -> npt.NDArray:
        """Returns the input after an ideal array programming.

        Error that represents the inaccuracy of actually setting the
        conductances within the AnalogCore. The error is sampled when the matrix
        is set and so does not effect individual MVMs differently.

        Args:
            input_: The input to be programmed to an array.

        Returns:
            npt.NDArray: An idential numpy-like version of the input.
        """
        return xp.asarray(input_)

    def drift_error(self, input_: npt.ArrayLike, time: float) -> npt.NDArray:
        """Returns the input after an ideal drift at a specified time.

        Args:
            input_: The input to be programmed to an array.
            time: Time elapsed, unused in the ideal case.

        Returns:
            npt.NDArray: An idential numpy-like version of the input.
        """
        return xp.asarray(input_)
