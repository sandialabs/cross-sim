#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#


from .idevice import EmptyDevice
from abc import ABC, abstractmethod

from simulator.backend import ComputeBackend

xp = ComputeBackend()  # Represents either cupy or numpy


class AbstractError(ABC):
    """An interface for creating random arrays from a given distribution."""

    def __init__(self, magnitude):
        """Initialize an Abstract Error Interface."""
        self.magnitude = magnitude

    @abstractmethod
    def create_error(self, matrix):
        """Create a random matrix by sampling from a distribution.

        Args:
            matrix: The normalized conductance matrix from whichthe random
                matrix is created

        Returns:
            np.nd_matrix : A 2d array of random values with the same shape as
                the input matrix
        """
        pass


class UniformError(AbstractError):
    """Error with a uniform distribution."""

    def create_error(self, matrix):
        """Create a random matrix by sampling from a uniform distribution."""
        distribution = xp.random.uniform(size=matrix.shape)
        shifted_and_scaled_distribution = (distribution - 0.5) * self.magnitude
        return shifted_and_scaled_distribution


class NormalError(AbstractError):
    """Error with a normal distribution."""

    def create_error(self, matrix, sigma: float | int = None):
        """Create an array of random values sampled from a normal distribution.

        Args:
            matrix: The matrix that error will be applied to
            sigma: std of error to apply to normalized conductances.
                Defaults to None.
        """
        sigma = sigma if sigma is not None else self.magnitude
        Rall = xp.random.normal(scale=sigma, size=matrix.shape)
        return Rall


class AbstractScalar(ABC):
    """Scale a value by a specific value."""

    @abstractmethod
    def scale_and_add(self, random_matrix, input_matrix):
        """Scale the input random_matrix.

        Args:
            input_matrix: 2d matrix of normalized conductances
            random_matrix: normalized conductances with random errors applied,
                same shape as input_matrix
        """
        raise NotImplementedError


class Independent(AbstractScalar):
    """Apply an error that is conductance invariant."""

    def scale_and_add(self, random_matrix, input_matrix):
        """Apply a conductance invariant error."""
        return input_matrix + random_matrix


class Proportional(AbstractScalar):
    """Apply an error that scales proportionally with conductance."""

    def scale_and_add(self, random_matrix, input_matrix):
        """Apply an error that scales with conductance."""
        return input_matrix * (1 + random_matrix)


class Inverse(AbstractScalar):
    """Apply an error that scales inversely with conductance.

    If conductance is zero, no error is applied.
    """

    def scale_and_add(self, random_matrix, input_matrix):
        """Apply an error that scales inversely with conductance."""
        signs = xp.backend.sign(input_matrix)
        magnitudes = xp.backend.abs(input_matrix)

        def inverse_func(x):
            return 1 / x if x > 0 else 0

        scaled_magnitudes = xp.backend.vectorize(inverse_func)(magnitudes)
        scaled_random_matrix = random_matrix * scaled_magnitudes * signs
        return input_matrix + scaled_random_matrix


class GenericDevice(EmptyDevice):
    """Abstract base class for composed devices.

    Attributes:
        distribution_type (AbstractError) : A class which generates the random
            matrices based on the shape of the input matrix.
        scalar_type (AbstractScalar) : The class which scales the random matrix
    """

    distribution_type: AbstractError
    scalar_type: AbstractScalar

    def __init__(self, device_params, weight_error_params):
        """Initialize a generic device with a specified error profile."""
        super().__init__(device_params, weight_error_params)
        self.distribution = self.distribution_type(self.magnitude)
        self.scalar = self.scalar_type()

    def _apply_error(self, matrix):
        error = self.distribution.create_error(matrix)
        return self.scalar.scale_and_add(error, matrix)

    def read_noise(self, input_):
        """Apply generic device read noise."""
        return self._apply_error(input_)

    def programming_error(self, input_):
        """Apply generic device programming error."""
        return self._apply_error(input_)


class NormalIndependentDevice(GenericDevice):
    """A device with state independent error and a normal distribution."""

    distribution_type = NormalError
    scalar_type = Independent


class NormalProportionalDevice(GenericDevice):
    """A device with proportional error and a normal distribution."""

    distribution_type = NormalError
    scalar_type = Proportional


class NormalInverseProportionalDevice(GenericDevice):
    """A device with inversely proportional error and a normal distribution."""

    distribution_type = NormalError
    scalar_type = Inverse


class UniformIndependentDevice(GenericDevice):
    """A device with state independent error and a uniform distribution."""

    distribution_type = UniformError
    scalar_type = Independent


class UniformProportionalDevice(GenericDevice):
    """A device with proportional error and a uniform distribution."""

    distribution_type = UniformError
    scalar_type = Proportional


class UniformInverseProportionalDevice(GenericDevice):
    """A device with inversely proportional error and a uniform distribution."""

    distribution_type = UniformError
    scalar_type = Inverse
