#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""Base class for CrossSim layer implementations.

AnalogLayer provides basic infrastructure that is used by common layer implementations
for both PyTorch and Keras.
"""

from __future__ import annotations

from abc import abstractmethod, ABC

from simulator import AnalogCore, CrossSimParameters
import numpy.typing as npt


class AnalogLayer(ABC):
    """Base class for CrossSim layer implementations.

    AnalogLayer is the base class for CrossSim layer implementations. The layer
    implementations internally contain an AnalogCore object and logic to form a 2D
    matrix and input driver to implement the layer functionality.
    Implementing class constructors should follow the the following pattern:
    `AnalogLinear(CrossSimParameters, *base layer arg, **analog specific args)`

    Attributes:
        core: An AnalogCore object which will implement the layer functionality.
        params:
            CrossSimParameters object or list of CrossSimParameters (for layers
            requiring multiple arrays) for the layer. If a list, the length must match
            the number of arrays used within AnalogCore.
    """

    core: AnalogCore
    params: CrossSimParameters

    @abstractmethod
    def form_matrix(
        self,
        weight: npt.ArrayLike,
        bias: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
        """Builds 2D weight matrix for programming into the array.

        Args:
            weight: Numpy ndarray (or similar) of the layer weights
            bias:
                Numpy ndarray (or similar) of the layer bias. Can be None if the layer
                has no bias or an analog_bias is not used by the layer.

        Returns:
            2D numpy ndarray of the matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, M_input: npt.ArrayLike) -> npt.NDArray:
        """Runs the layer forward operation.

        Args:
            M_input:
                Numpy ndarray (or similar) of the layer input. Shape and order of input
                dimensions should follow the PyTorch ordering

        Returns:
            Numpy ndarray result of the layer operation. Should follow PyTorch shape
            and dimension.
        """
        raise NotImplementedError

    def set_matrix(self, matrix: npt.ArrayLike, verbose=False) -> None:
        """Programs a matrix into the layer's internal AnalogCore.

        See AnalogCore.set_matrix for details.

        Args:
            matrix: Numpy ndarray to be programmed into the array.
            verbose: Boolean flag to enable verbose print statements.
        """
        self.core.set_matrix(matrix, verbose=verbose)

    def get_matrix(self) -> npt.NDArray:
        """Returns the programmed 2D analog array.

        Returns:
            Numpy array of the 2D array with non-idealities applied.
        """
        return self.core.get_matrix()

    # A few properties to pass through from AnalogCore so this can be treated
    # like an AnalogCore for manipulation and debugging purposes.
    @property
    def max(self):  # noqa: A003
        """Internal AnalogCore's defined maximum matrix value.

        Can be defined either using percentile or explict min/max scaling. See
        AnalogCore documentation for details.
        """
        return self.core.max

    @property
    def min(self):  # noqa: A003
        """Internal AnalogCore's defined minimum matrix value.

        Can be defined either using percentile or explict min/max scaling. See
        AnalogCore documentation for details.
        """
        return self.core.min

    @property
    def shape(self):
        """Shape of 2D Matrix representing the layer.

        Shape represents the original matrix shape (from form_matrix) and does not
        include any matrix transformations performed with AnalogCore or subcores.
        """
        return self.core.shape

    @property
    def Ncores(self):
        """Number of partitions in the internal AnalogCore."""
        return self.core.Ncores

    @property
    def cores(self):
        """List of partitions in the internal AnalogCore."""
        return self.core.cores

    @property
    def num_cores_row(self):
        """Number of partitions in the horizontal (row) direction."""
        return self.core.num_cores_row

    @property
    def num_cores_col(self):
        """Number of partitions in the vertical (column) direction."""
        return self.core.num_cores_col

    @property
    def dtype(self):
        """Datatype of the matrix stored in the internal AnalogCore."""
        return self.core.dtype

    def __setitem__(self, key, value):
        """Forward setitem on the layer to the internal AnalogCore.

        Used primarily for PyTorch layer synchronization by setting values with masks.
        """
        self.core.__setitem__(key, value)
