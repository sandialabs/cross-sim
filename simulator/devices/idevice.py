#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import logging
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from simulator.backend.compute import ComputeBackend
from simulator.parameters.device import DeviceParameters

xp: np = ComputeBackend()  # Represents either cupy or numpy
log = logging.getLogger(__name__)


class IDevice(ABC):
    """Interface for device objects.

    Partial implementations of the interface (e.g., only implementing
    read_error) should do one of the following:
        1. Inherit from IdealDevice
        2. Inherit from EmptyDevice
        3. Manually raise NotImplementedError in unimplemented functions
    """

    device_params: DeviceParameters
    cell_bits: int
    Rmin: float
    Rmax: float
    time: int | float
    infinite_on_off_ratio: bool
    clip_conductance: bool

    # TODO: Some properties pulled from param, others defined here
    #       Would be cleaner if we were consistent.
    #       (@Curtis, 2023-10-25)
    @property
    def Gmin_norm(self) -> float:
        """Returns the normalized minimum programmable device conductance.

        Normalization is performed dividing by the maximum programmable
        conductance.
        """
        return self.device_params.Gmin_norm

    @property
    def Gmax_norm(self) -> float:
        """Returns the normalized maximum programmable device conductance.

        Equal to 1 by definition.
        """
        return self.device_params.Gmax_norm

    @property
    def Grange_norm(self) -> float:
        """Returns the normalized range of programmable device conductances.

        Normalization is performed using the max programmable resistance.
        """
        return self.device_params.Grange_norm

    @property
    def levels(self) -> int:
        """Number of conductance levels that can be targeted."""
        return 2**self.cell_bits

    @property
    def on_off_ratio(self):
        """Returns the on/off ratio of the device."""
        if self.Gmin_norm > 0:
            return 1 / self.Gmin_norm
        else:
            return 1e20

    @abstractmethod
    def read_noise(self, input_: npt.ArrayLike) -> npt.NDArray:
        """Error representing device conductances during an MVM.

        Read noise represents error associated with the device conductances
        (the weights in the matrix) at the time of the MVM (e.g., the
        temperature could effect the conductance).
        The read noise is re-sampled on every MVM rather than at write time, as
        a result the implementation can become a bottleneck (especially when
        running on GPUs).
        """
        raise NotImplementedError

    @abstractmethod
    def programming_error(self, input_: npt.ArrayLike) -> npt.NDArray:
        """Error representing inaccuracy of actually setting conductances.

        Error that represents the inaccuracy of actually setting the
        conductances within the AnalogCore. The error is sampled when the matrix
        is set and so does not effect individual MVMs differently.
        """
        raise NotImplementedError

    @abstractmethod
    def drift_error(self, input_: npt.ArrayLike) -> npt.NDArray:
        """Error representing the drift in the conductance of matrix over time.

        The drift model is assumed to be a complete model of device errors at
        the set time, i.e. includes the effect of programming errors.
        If time > 0, programming errors are not applied separately from drift
        errors.
        """
        raise NotImplementedError

    def _apply_conductance_quantization(self, input_: npt.ArrayLike) -> npt.NDArray:
        # set qmult:  multiply by this factor to convert every level to an
        # absolute range of 1
        # this allows quantization to be done by rounding
        input_ = xp.asarray(input_)

        # The first level maps to a nonzero current
        qmult = (self.levels - 1) / (self.Gmax_norm - self.Gmin_norm)
        input_ -= self.Gmin_norm  # shift min to zero
        input_ *= qmult  # multiply by a quantization factor to allow for rounding
        input_ = xp.backend.rint(input_, out=input_)
        input_ /= qmult
        input_ += self.Gmin_norm  # shift zero back
        return input_

    def apply_write_error(self, input_: npt.ArrayLike) -> npt.NDArray:
        """Default implementation of apply write errors function.

        IMPORTANT: assumes that drift error includes programming error.

        Args:
            input_: Input to apply write error to.

        Returns:
            npt.NDArray: Matrix with write errors applied.
        """
        # TODO: Should we pass in time as an arg?
        #       See SONOS
        # (@Curtis, 2023-10-24)
        if self.cell_bits > 0:
            input_ = self._apply_conductance_quantization(input_)

        if self.time == 0:
            # If time is 0,
            #   1) If programming error is not IdealDevice, apply programming
            #      error
            #   2) else, apply drift error evaluated at t=0
            if self.device_params.programming_error.model != "IdealDevice":
                noisy_matrix = self.programming_error(input_)
            else:
                noisy_matrix = self.drift_error(input_, self.time)
        else:
            noisy_matrix = self.drift_error(input_, self.time)
        return noisy_matrix
