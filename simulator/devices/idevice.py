#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#


from abc import ABC, abstractmethod

from simulator.backend import ComputeBackend

xp = ComputeBackend()  # Represents either cupy or numpy


class IDevice(ABC):
    """IDevice is the Abstract Base class for the error models for devices.
    Partial implementations of the interface (e.g., only
    implementing read_error) should either inherit from IdealDevice or define
    the remaining errors with NotImplementedError.
    """

    def __init__(
        self,
        device_params,
        weight_error_params,
    ):
        """Common init for device types, unpack device parameters to attributes.

        Args:
            device_params (DeviceParameters): Common device paramters
            weight_error_params (WeightErrorParameters | DriftErrorParameters):
                Device parameters specific to read noise
        """
        super().__init__()
        self.device_params = device_params
        self.Gmin_norm = device_params.Gmin_norm
        self.Gmax_norm = device_params.Gmax_norm
        self.Grange_norm = device_params.Grange_norm
        self.clip_conductance = device_params.clip_conductance
        self.cell_bits = device_params.cell_bits
        self.time = device_params.time
        self.magnitude = weight_error_params.magnitude

        # Currently unused
        self.initial_time = 0

        # On/off ratio
        if self.Gmin_norm > 0:
            self.on_off_ratio = 1 / self.Gmin_norm
        else:
            self.on_off_ratio = 1e20

        # Number of conductance levels that can be targeted
        self.levels = 2**self.cell_bits

    def apply_write_error(self, input_):
        """Default implementation of apply write errors function,
        IMPORTANT: assumes that drift error includes programming error.

        Args:
            input_ (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.cell_bits > 0:
            input_ = self._apply_conductance_quantization(input_)

        if self.time == 0:
            # If time is 0,
            #   1) If programming error is not IdealDevice,
            #       apply programming error
            #   2) else, apply drift error evaluated at t=0
            if self.device_params.programming_error.model != "IdealDevice":
                noisy_matrix = self.programming_error(input_)
            else:
                noisy_matrix = self.drift_error(input_, self.time)
        else:
            noisy_matrix = self.drift_error(input_, self.time)

        return noisy_matrix

    def _apply_conductance_quantization(self, input_):
        # set qmult:  multiply by this factor to convert every level to an
        # absolute range of 1 this allows quantization to be done by rounding

        # The first level maps to a nonzero current
        qmult = (self.levels - 1) / (self.Gmax_norm - self.Gmin_norm)
        input_ -= self.Gmin_norm  # shift min to zero
        input_ *= qmult  # multiply by a quantization factor to allow  rounding
        input_ = xp.backend.rint(input_, out=input_)
        input_ /= qmult
        input_ += self.Gmin_norm  # shift zero back
        return input_

    @abstractmethod
    def read_noise(self, input_):
        """Returns a noisy version of a matrix after reading.

        Read noise represents error associated with the device conductances
        (the weights in the matrix) at the time of the MVM (e.g., the
        temperature could effect the conductance).

        The read noise is re-sampled on every MVM rather than at write time, as
        a result the implementation can become a bottleneck, especially on GPUs.
        """
        raise NotImplementedError

    @abstractmethod
    def programming_error(self, input_):
        """Returns a noisy version of the conductance matrix after programming.

        Error that represents the inaccuracy of actually setting the
        conductances within the AnalogCore. The error is sampled when the matrix
        is set and so does not effect individual MVMs differently.
        """
        raise NotImplementedError

    @abstractmethod
    def drift_error(self, input_):
        """Returns a noisy version of the condutance matrix after drift error.

        Error representing the drift in the conductance of matrix over time.
        The drift model is assumed to be a complete model of device errors at
        the set time, i.e. includes the effect of programming errors.
        """
        raise NotImplementedError

    @abstractmethod
    def nonlinear_current(self, Gmat, Vmat):
        """Returns the current through every cell given the current cell states.

        This is used to simulate nonlinear I-V characteristics. This method does
        not give a conductance output, but a current output.

        Args:
            Gmat: The (normalized) conductance matrix where the conductance is
                defined at some nominal voltage.
            Vmat: A matrix of (normalized) applied voltages across every device.
        """
        raise NotImplementedError

    @abstractmethod
    def nonlinear_current_sum(self, Gmat, Vterm):
        """Returns the current sums accounting for I-V nonlinearity.

        This calculation is only possible if cell currents are not coupled: i.e.
        there are no parasitic resistances and the output line is at virtual
        ground. If implemented, this can give a significant speedup over using
        nonlinear_current() and summing the currents in the circuit solver.
        To enable this path, must have
        params.simulation.fast_nonlinear_IV = True.

        Args:
            Gmat: The (normalized) conductance matrix where the conductance is
                defined at some nominal voltage.
            Vterm: A vector or matrix of (normalized) applied terminal voltages.
        """
        raise NotImplementedError

    @classmethod
    def get_all_subclasses(cls: type, use_base=True) -> list[type]:
        """Returns all subclasses of a type
        Args:
            cls (type): Type to get subclasses of, ignored if use_base is True
            use_base (bool, optional): If set all subclasses of the BaseDevice
                are returned as opposed to of the current type.
                Defaults to True.

        Returns:
            list[type]: A list of all BaseDevice classes.
        """
        root = cls
        if use_base:
            root = IDevice
        subclasses = []
        for subclass in root.__subclasses__():
            subclasses.append(subclass)
            subclasses.extend(subclass.get_all_subclasses(use_base=False))
        return subclasses


# EmptyDevice: mostly kept as a way to hack around implementing the interface
class EmptyDevice(IDevice):
    """Device which implements IDevice with the default behavior of raising
    an exception for non-implemented methods.
    """

    def read_noise(self, input_):
        """Returns a version of a matrix after reading."""
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented read noise",
        )

    def programming_error(self, input_):
        """Returns a version of the conductance matrix after programming."""
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented programming error",
        )

    def drift_error(self, input_):
        """Returns a version of the condutance matrix after drift error."""
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented drift error",
        )

    def nonlinear_current(self, Gmat, Vmat):
        """Returns the current through every cell."""
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented I-V nonlinearity",
        )

    def nonlinear_current_sum(self, Gmat, Vterm):
        """Returns the current sums accounting for I-V nonlinearity."""
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented nonlinear current sums.",
            "Please either implement it, or set "
            "params.simulation.disable_fast_nonlinear_IV = True.",
        )


# Ideal device
class IdealDevice(IDevice):
    """An ideal device which doesn't make errors."""

    def read_noise(self, input_):
        """Returns the input after an ideal read."""
        return input_

    def programming_error(self, input_):
        """ ""Returns the input after an ideal array programming."""
        return input_

    def drift_error(self, input_, time):
        """ ""Returns the input after an ideal drift at a specified time."""
        return input_

    def nonlinear_current(self, Gmat, Vmat):
        """Returns the current through every cell with ideal cells."""
        if Vmat.ndim == Gmat.ndim:
            return Gmat * Vmat
        else:
            # Account for the fact that Vmat can have more dimensions than Gmat
            return Gmat[(..., *(None,) * (Vmat.ndim - Gmat.ndim))] * Vmat

    def nonlinear_current_sum(self, Gmat, Vterm):
        """Returns the current sums with ideal cells."""
        return xp.matmul(Gmat, Vterm)
