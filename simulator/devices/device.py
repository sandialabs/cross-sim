#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from .idevice import IDevice
from .generic_device import (
    UniformIndependentDevice,
    UniformProportionalDevice,
    UniformInverseProportionalDevice,
    NormalProportionalDevice,
    NormalInverseProportionalDevice,
    NormalIndependentDevice,
)
from .custom.PCM_Joshi import PCMJoshi
from .custom.RRAM_Milo import RRAMMilo
from .custom.SONOS import SONOS
from simulator.circuits import array_simulator
from typing import Any

from simulator.backend import ComputeBackend

xp = ComputeBackend()


class Device(IDevice):
    def __init__(
        self,
        *args,
        read_noise_model,
        programming_error_model,
        drift_error_model,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._read_noise_model = read_noise_model
        self._programming_error_model = programming_error_model
        self._drift_error_model = drift_error_model

    def read_noise(self, input_, mask=None):
        if self.device_params.read_noise.enable:
            noisy_matrix = self._read_noise_model.read_noise(input_.copy())
            return self.clip_and_mask(noisy_matrix, mask)
        else:
            return input_

    def programming_error(self, input_, mask=None):
        if self.device_params.programming_error.enable:
            noised_input = self._programming_error_model.programming_error(
                input_.copy(),
            )
            return self.clip_and_mask(noised_input, mask)
        else:
            return input_

    def drift_error(self, input_, time, mask=None):
        if self.device_params.drift_error.enable:
            noised_input = self._drift_error_model.drift_error(input_.copy(), time)
            return self.clip_and_mask(noised_input, mask)
        else:
            return input_

    def clip_and_mask(self, input_, mask):
        if self.clip_conductance:
            input_ = input_.clip(self.Gmin_norm, self.Gmax_norm)
        else:
            # Regardless of whether clip conductance is on, G cannot be negative
            input_ = input_.clip(0, None)
        if mask is not None:
            input_ *= mask
        return input_

    @staticmethod
    def create_device(device_parameters: dict[str, Any]) -> IDevice:
        """Creates a device according to the specification by the device parameters
        Args:
            device_parameters (dict[str, Any]): Parameters to describe device behavior
        Raises:
            ValueError: Raised when an unknown read or write model is specified
        Returns:
            Device: A device using the parameters listed.
        """
        device_types = {
            subcls.__name__: subcls for subcls in Device.get_all_subclasses()
        }

        # Remove dummy Device type and classes which don't represent valid options
        device_types.pop("Device")
        device_types.pop("EmptyDevice")
        device_types.pop("GenericDevice")

        read_error_model = device_parameters.read_noise.model
        read_error_params = device_parameters.read_noise
        programming_error_model = device_parameters.programming_error.model
        programming_error_params = device_parameters.programming_error
        drift_model = device_parameters.drift_error.model
        drift_params = device_parameters.drift_error

        # Error checking for more user friendly exceptions
        message = ""
        if read_error_model not in device_types:
            message += f"Unknown read model: {read_error_model}.\n"
        if programming_error_model not in device_types:
            message += f"Unknown programming error model: {programming_error_model}.\n"
        if drift_model not in device_types:
            message += f"Unknown drift model: {drift_model}.\n"
        if message:
            message += (
                "Either define a model by that name or select from the"
                + f"{len(device_types)} existing options: {list(device_types)}"
            )
            raise ValueError(message)

        # Create the custom device and return it
        read = device_types[read_error_model](device_parameters, read_error_params)
        programming = device_types[programming_error_model](
            device_parameters,
            programming_error_params,
        )
        drift = device_types[drift_model](device_parameters, drift_params)

        device = Device(
            device_parameters,
            read_error_params,
            read_noise_model=read,
            programming_error_model=programming,
            drift_error_model=drift,
        )

        # Try all three methods on a dummy input to make sure they are valid
        # This ensures failure at initialization rather than runtime
        device._read_noise_model.read_noise(xp.zeros(1))
        device._programming_error_model.programming_error(xp.zeros(1))
        device._drift_error_model.drift_error(xp.zeros(1), 0)

        return device
