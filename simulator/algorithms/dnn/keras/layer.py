#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""Base class for all CrossSim Keras layers.

AnalogLayer is the base class for all CrossSim equivalents of keras layers.
AnalogLayer provides a common interface for CrossSim keras layers and basic
infrastructure.
"""


from __future__ import annotations

from abc import abstractmethod, ABC

from simulator import CrossSimParameters

from keras.layers import Layer

import numpy.typing as npt
from typing import Any


class AnalogLayer(Layer, ABC):
    """Base class for CrossSim keras layers.

    AnalogLayer is the base class for keras layers which internally use
    CrossSim for the forward pass. Implementing classes will provide
    implementations for the specific matrix formation and forward inference.

    Implementing  classes should inherit from the
    original keras layer and AnalogLayer as follows
    `AnalogDense(AnalogLayer, keras.layers.Dense)` with the keras base layer last
    to ensure proper module resolution order. Implementing classes should exactly
    match the output of the original keras layer when all error models are disabled
    for the forward function. Keras AnalogLayers do not currently support backward
    operations.

    Attributes:
        core:
            An AnalogCore or similar object for CrossSim simulation.
            self.core must provide a set_matrix and get_matrix function. In most cases
            these should be thin wrappers around the implementations in AnalogCore
        params:
            CrossSimParameters object or list of CrossSimParameters (for layers
            requiring multiple arrays) for the AnalogLinear layer. If a list, the
            length must match the number of arrays used within AnalogCore.
        analog_bias:
            Boolean indicating if the bias of the layer is part of the analog
            array or stored and computing digitally.
        bias_rows:
            Integer indicating the number of rows to use to implement the bias
            within the array. 0 implies a digital bias.
    """

    def __init__(
        self,
        params: CrossSimParameters | list[CrossSimParameters],
        bias_rows: int = 0,
        **kwargs,
    ) -> None:
        """Initialize analog-specific attributes of AnalogLayer.

        Generic AnalogLayer handling, passes kwargs to the base keras layer for further
        initialization. kwargs should contain all arguments for the keras layer.

        Args:
            params:
                CrossSimParameters object or list of CrossSimParameters (for layers
                requiring multiple arrays) for the AnalogLinear layer. If a list, the
                length must match the number of arrays used within AnalogCore.
            bias_rows:
                Integer indicating the number of rows to use to implement the bias
                within the array. 0 implies a digital bias.
            **kwargs: All arguments for base keras layer.

        """
        super().__init__(**kwargs)

        if isinstance(params, CrossSimParameters):
            self.params = params.copy()
        elif isinstance(params, list):
            self.params = params[0].copy()

        self.bias_rows = bias_rows
        self.analog_bias = self.use_bias and bias_rows > 0

        # Keras layers have a few gpu-specific things needed so set this here
        self.useGPU = self.params.simulation.useGPU
        if self.useGPU:
            # There is a potential race condition because Keras doesn't use the default
            # stream for weights so the dlpack zero-copy transfer can create a data
            # race. Grab the stream here so we can use it synchronize with it later.
            import cupy as cp

            self.stream = cp.cuda.get_current_stream()

    def set_weights(self, weights: list[npt.NDArray]) -> None:
        """Sets the analog matrix based on a list of Numpy arrays.

        Wraps Layer.set_weights function setting the matrix based on the new weights.

        Args:
            weights: List of numpy arrays, one for each weight variable.
        """
        super().set_weights(weights)
        self.core.set_matrix(self.form_matrix())

    def get_config(self) -> dict[str, Any]:
        """Returns the config of the object.

        Wraps layer.get_config() adding the new params for Analog layers.

        Returns:
            Dictionary specifying the layer configuration.
        """
        base_config = super().get_config()
        config = {
            "params": self.params,
            "bias_rows": self.bias_rows,
        }
        return {**base_config, **config}

    def get_matrix(self) -> npt.NDArray:
        """Returns the 2D matrix progammed into the array.

        Returns:
            Numpy array of the 2D weight matrix with non-idealities applied.
        """
        return self.core.get_matrix()

    @abstractmethod
    def build(self, input_shape) -> None:
        """Create weights and core based on the input spec.

        Base class build will create weight variables based on the input shape and
        initialize those variables. Subclass implementations should create self.core in
        their implementations of build since the shape of the weight matrix isn't known
        until build is called.

        Args:
            input_shape: keras input_shape object
        """
        super().build(input_shape)

    @abstractmethod
    def call(self, inputs: npt.ArrayLike) -> npt.ArrayLike:
        """Calls the layer operation."""
        raise NotImplementedError

    @abstractmethod
    def form_matrix(self) -> npt.NDArray:
        """Builds 2D weight matrix for programming into the array.

        Returns:
            2D Numpy Array of the matrix.

        """
        raise NotImplementedError

    @abstractmethod
    def get_core_weights(self) -> list[npt.NDArray]:
        """Gets the weight and bias values with errors applied.

        Implementations should use self.get_matrix and reshape the result into the
        original weights.

        Returns:
            List of numpy arrays with errors applied. CrossSim version of get_weights
        """
        raise NotImplementedError

    @abstractmethod
    def reinitialize(self) -> None:
        """Rebuilds the layer's internal core object.

        Allows parameters to be updated within a layer without rebuilding the
        layer. This will resample all initialization-time errors
        (e.g. programming error)  even if the models were not be changed.
        Alternatively,  reinitialize can be used to directly resample
        initialization-time errors.
        """
        raise NotImplementedError

    @classmethod
    def from_keras(
        cls,
        layer: Layer,
        params: CrossSimParameters,
        bias_rows: int = 0,
    ) -> AnalogLayer:
        """Returns an analog version of a Keras layer.

        Creates a new Analog Layer with the same attributes as the original
        Keras layer.

        Arguments:
            layer: The Keras layer to copy
            params:
                CrossSimParameters object or list of CrossSimParameters objects
                (for layers requiring multiple arrays) for the analog layer.
            bias_rows:
                Integer indictating the number of analog rows to use for the bias.
                0 indicates a digital bias. Ignored if layer does not have a bias.

        Returns:
            A new analog equivalent layer.

        """
        analog_args = {
            "params": params,
            "bias_rows": bias_rows,
        }
        analog_layer = cls(**analog_args, **layer.get_config())

        if layer.built:
            analog_layer.build_from_config(layer.get_build_config())
            analog_layer.set_weights(layer.get_weights())

        return analog_layer

    @classmethod
    def to_keras(cls, layer: AnalogLayer, physical_weights: bool = False) -> Layer:
        """Returns a Keras layer from an analog layer.

        By default weights will be copied directly from the layer's weight
        attributes and therefore will not include non-idealities.
        If idealities are desired, use physical_weights = True.

        Arguments:
            layer: AnalogLayer to copy
            physical_weights:
                Bool indicating whether the torch layer should have ideal weights or
                weights with programming error applied.

        Returns:
            A new torch equivalent to the analog layer.

        """
        # This is a little ugly, in general the keras layer should be bases[-1]
        # So we are hard coding it until someone needs us to do something
        # different.
        config = layer.get_config()
        del config["params"]
        del config["bias_rows"]

        keras_layer = cls.__bases__[-1](**config)

        if layer.built:
            keras_layer.build_from_config(layer.get_build_config())
            if physical_weights:
                keras_layer.set_weights(layer.get_core_weights())
            else:
                keras_layer.set_weights(layer.get_weights())

        return keras_layer
