#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import logging

import numpy as np

from simulator.devices.base_device import BaseDevice
from simulator.parameters.device import (
    DeviceParameters,
)
from simulator.parameters.device.types import GenericDeviceParameters
from abc import ABC, abstractmethod

from simulator.backend import ComputeBackend

xp: np = ComputeBackend()  # Represents either cupy or numpy
log = logging.getLogger(__name__)


class AbstractError(ABC):
    """An interface for creating random arrays from a given distribution."""

    def __init__(self, magnitude):
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
    def create_error(self, matrix):
        """Create a random matrix by sampling from a uniform distribution."""
        distribution = xp.random.uniform(size=matrix.shape)
        shifted_and_scaled_distribution = (distribution - 0.5) * self.magnitude
        return shifted_and_scaled_distribution


class NormalError(AbstractError):
    def create_error(self, matrix, sigma: float | int = None):
        """Create an array of random values sampled from a normal distribution.

        Args:
            sigma: std of error to apply to normalized conductances.
                Defaults to None.
        """
        sigma = sigma if sigma is not None else self.magnitude
        Rall = xp.random.normal(scale=sigma, size=matrix.shape)
        return Rall


class AbstractScalar(ABC):
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
        return input_matrix + random_matrix


class Proportional(AbstractScalar):
    """Apply an error that scales proportionally with conductance."""

    def scale_and_add(self, random_matrix, input_matrix):
        return input_matrix * (1 + random_matrix)


class Inverse(AbstractScalar):
    """Apply an error that scales inversely with conductance.

    If conductance is zero, no error is applied.
    """

    def scale_and_add(self, random_matrix, input_matrix):
        signs = xp.backend.sign(input_matrix)
        magnitudes = xp.backend.abs(input_matrix)

        def inverse_func(x):
            return 1 / x if x > 0 else 0

        scaled_magnitudes = xp.backend.vectorize(inverse_func)(magnitudes)
        scaled_random_matrix = random_matrix * scaled_magnitudes * signs
        return input_matrix + scaled_random_matrix


class GenericDevice(BaseDevice):
    """Abstract base class for composed devices.

    Attributes:
        distribution_type: A class which generates the random matrices based on
            the shape of the input matrix.
        scalar_type: The class which scales the random matrix
    """

    distribution_type: AbstractError
    scalar_type: AbstractScalar

    def __init__(
        self,
        device_params: DeviceParameters,
        model_params: GenericDeviceParameters,
    ):
        super().__init__(device_params, model_params)
        self.magnitude = model_params.magnitude
        self.distribution = self.distribution_type(self.magnitude)
        self.scalar = self.scalar_type()

    def _apply_error(self, matrix):
        error = self.distribution.create_error(matrix)
        return self.scalar.scale_and_add(error, matrix)

    def read_noise(self, input_):
        return self._apply_error(input_)

    def programming_error(self, input_):
        return self._apply_error(input_)


class NormalIndependentDevice(GenericDevice):
    distribution_type = NormalError
    scalar_type = Independent


class NormalProportionalDevice(GenericDevice):
    distribution_type = NormalError
    scalar_type = Proportional


class NormalInverseProportionalDevice(GenericDevice):
    distribution_type = NormalError
    scalar_type = Inverse


class UniformIndependentDevice(GenericDevice):
    distribution_type = UniformError
    scalar_type = Independent


class UniformProportionalDevice(GenericDevice):
    distribution_type = UniformError
    scalar_type = Proportional


class UniformInverseProportionalDevice(GenericDevice):
    distribution_type = UniformError
    scalar_type = Inverse
