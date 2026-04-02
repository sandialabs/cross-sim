#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#


from .idevice import IDevice
from .generic_device import (  # noqa: F401
    UniformIndependentDevice,
    UniformProportionalDevice,
    UniformInverseProportionalDevice,
    NormalProportionalDevice,
    NormalInverseProportionalDevice,
    NormalIndependentDevice,
)
from .custom.PCM_Joshi import PCMJoshi  # noqa: F401
from .custom.RRAM_Milo import RRAMMilo  # noqa: F401
from .custom.RRAM_Wan import RRAMWan  # noqa: F401
from .custom.SONOS import SONOS  # noqa: F401
from .custom.Analytical_Devices import CubicDevice  # noqa: F401

from typing import Any

from simulator.backend import ComputeBackend

xp = ComputeBackend()


class Device(IDevice):
    """Device object used by cores to model device errors."""

    def __init__(
        self,
        *args,
        read_noise_model,
        programming_error_model,
        drift_error_model,
        nonlinear_IV_model,
        **kwargs,
    ):
        """Initializes a device to model errors."""
        super().__init__(*args, **kwargs)
        self._read_noise_model = read_noise_model
        self._programming_error_model = programming_error_model
        self._drift_error_model = drift_error_model
        self._nonlinear_IV_model = nonlinear_IV_model

    def read_noise(self, input_, mask=None):
        """Returns a version of a matrix after reading."""
        if self.device_params.read_noise.enable:
            noisy_matrix = self._read_noise_model.read_noise(input_.copy())
            return self.clip_and_mask(noisy_matrix, mask)
        else:
            return input_

    def programming_error(self, input_, mask=None):
        """Returns a version of the conductance matrix after programming."""
        if self.device_params.programming_error.enable:
            noised_input = self._programming_error_model.programming_error(
                input_.copy(),
            )
            return self.clip_and_mask(noised_input, mask)
        else:
            return input_

    def drift_error(self, input_, time, mask=None):
        """Returns a version of the condutance matrix after drift error."""
        if self.device_params.drift_error.enable:
            noised_input = self._drift_error_model.drift_error(input_.copy(), time)
            return self.clip_and_mask(noised_input, mask)
        else:
            return input_

    def nonlinear_current(self, Gmat, Vmat):
        """Returns the current through every cell given the current cell states.

        This is used to simulate nonlinear I-V characteristics. This method does
        not give a conductance output, but a current output.
        """
        if self.device_params.nonlinear_IV.enable:
            if Vmat.ndim >= Gmat.ndim:
                return self._nonlinear_IV_model.nonlinear_current(Gmat, Vmat)
            else:
                raise ValueError(
                    "In nonlinear_current, Vmat cannot have fewer dimensions than Gmat."
                )
        else:
            return Gmat * Vmat

    def nonlinear_current_sum(self, Gmat, Vterm):
        """Returns the current sums accounting for I-V nonlinearity."""
        if self.device_params.nonlinear_IV.enable:
            return self._nonlinear_IV_model.nonlinear_current_sum(Gmat, Vterm)
        else:
            return xp.matmul(Gmat, Vterm)

    def clip_and_mask(self, input_, mask):
        """Applies clipping and masking to the conductance matrix."""
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
        """Creates a device according to the device parameters
        Args:
            device_parameters (dict[str, Any]): Parameters to describe device
                behavior
        Raises:
            ValueError: Raised when an unknown read or write model is specified
        Returns:
            Device: A device using the parameters listed.
        """
        device_types = {
            subcls.__name__: subcls for subcls in Device.get_all_subclasses()
        }

        # Remove dummy Device type and classes which don't represent valid
        # options
        device_types.pop("Device")
        device_types.pop("EmptyDevice")
        device_types.pop("GenericDevice")

        read_error_model = device_parameters.read_noise.model
        read_error_params = device_parameters.read_noise
        programming_error_model = device_parameters.programming_error.model
        programming_error_params = device_parameters.programming_error
        drift_model = device_parameters.drift_error.model
        drift_params = device_parameters.drift_error
        nonlinear_IV_model = device_parameters.nonlinear_IV.model
        nonlinear_IV_params = device_parameters.nonlinear_IV

        # Error checking for more user friendly exceptions
        message = ""
        if read_error_model not in device_types:
            message += f"Unknown read model: {read_error_model}.\n"
        if programming_error_model not in device_types:
            message += f"Unknown programming error model: {programming_error_model}.\n"
        if drift_model not in device_types:
            message += f"Unknown drift model: {drift_model}.\n"
        if nonlinear_IV_model not in device_types:
            message += f"Unknown I-V nonlinearity model: {nonlinear_IV_model}.\n"
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
        nonlinear_IV = device_types[nonlinear_IV_model](
            device_parameters, nonlinear_IV_params
        )

        device = Device(
            device_parameters,
            read_error_params,
            read_noise_model=read,
            programming_error_model=programming,
            drift_error_model=drift,
            nonlinear_IV_model=nonlinear_IV,
        )

        # Try all three methods on a dummy input to make sure they are valid
        # This ensures failure at initialization rather than runtime
        device._read_noise_model.read_noise(xp.zeros(1))
        device._programming_error_model.programming_error(xp.zeros(1))
        device._drift_error_model.drift_error(xp.zeros(1), 0)

        return device
