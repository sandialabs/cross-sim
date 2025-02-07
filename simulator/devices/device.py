#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import logging

import numpy as np

import simulator.devices.models as models  # noqa: F401

from simulator.parameters.device import DeviceParameters
from simulator.devices.base_device import BaseDevice
from simulator.backend.compute import ComputeBackend
from simulator.devices.idevice import IDevice

log = logging.getLogger(__name__)
xp: np = ComputeBackend()


class Device(IDevice):
    """Device object used by cores to model device errors.

    A device that models read noise, programming noise and drift error noise.
    For each error type, a differerent model can be used independent of the
    other models.

    For example, read noise and programming noise error can be configured to be
    ideal models (e.g. they produce no error), but drift error can be configured
    to be a non-ideal model.

    Attributes:
        cell_bits: Refer to `DeviceParameters.cell_bits`.
        clip_conductances: Refer to `DeviceParameters.clip_conductances`.
        time: Refer to `DeviceParameters.time`.
    """

    def __init__(self, device_params: DeviceParameters):
        """Initializes a composite device to model errors.

        Error modeling is provided for read noise, programming error, and drift.

        Args:
            device_params: Parameters for the device to create
        """
        super().__init__()
        self.device_params = device_params
        self.cell_bits = device_params.cell_bits
        self.clip_conductance = device_params.clip_conductance
        self.time = device_params.time

        self._read_noise_model = BaseDevice(
            device_params=device_params,
            model_params=device_params.read_noise,
        )
        self._programming_error_model = BaseDevice(
            device_params=device_params,
            model_params=device_params.programming_error,
        )
        self._drift_error_model = BaseDevice(
            device_params=device_params,
            model_params=device_params.drift_error,
        )

    def read_noise(self, input_, mask=None):
        """Error representing device conductances during an MVM.

        Read noise represents error associated with the device conductances
        (the weights in the matrix) at the time of the MVM (e.g., the
        temperature could effect the conductance).
        The read noise is re-sampled on every MVM rather than at write time, as
        a result the implementation can become a bottleneck (especially when
        running on GPUs).
        """
        if self.device_params.read_noise.enable:
            noisy_matrix = self._read_noise_model.read_noise(input_.copy())
            return self._clip_and_mask(noisy_matrix, mask)
        else:
            return input_

    def programming_error(self, input_, mask=None):
        """Error representing inaccuracy of actually setting conductances.

        Error that represents the inaccuracy of actually setting the
        conductances within the AnalogCore. The error is sampled when the matrix
        is set and so does not effect individual MVMs differently.
        """
        if self.device_params.programming_error.enable:
            noised_input = self._programming_error_model.programming_error(
                input_.copy(),
            )
            return self._clip_and_mask(noised_input, mask)
        else:
            return input_

    def drift_error(self, input_, time, mask=None):
        """Error representing the drift in the conductance of matrix over time.

        The drift model is assumed to be a complete model of device errors at
        the set time, i.e. includes the effect of programming errors.
        If time > 0, programming errors are not applied separately from drift
        errors.
        """
        if self.device_params.drift_error.enable:
            noised_input = self._drift_error_model.drift_error(input_.copy(), time)
            return self._clip_and_mask(noised_input, mask)
        else:
            return input_

    def _clip_and_mask(self, input_, mask):
        if self.clip_conductance:
            input_ = input_.clip(self.Gmin_norm, self.Gmax_norm)
        else:
            # Regardless of whether clip conductance is on, G cannot be negative
            input_ = input_.clip(0, None)
        if mask is not None:
            input_ *= mask
        return input_
