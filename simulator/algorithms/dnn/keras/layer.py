#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from __future__ import annotations

from abc import abstractmethod, ABC

from simulator import CrossSimParameters
from keras.layers import Layer


class AnalogLayer(Layer, ABC):
    def __init__(
        self,
        params: CrossSimParameters,
        bias_rows: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.params = params.copy()
        self.bias_rows = bias_rows
        self.analog_bias = self.use_bias and bias_rows > 0

    def set_weights(self, weights) -> None:
        super().set_weights(weights)
        self.core.set_matrix(self.form_matrix())

    def get_config(self):
        base_config = super().get_config()
        config = {
            "params": self.params,
            "bias_rows": self.bias_rows,
        }
        return {**base_config, **config}

    def get_matrix(self):
        return self.core.get_matrix()

    @abstractmethod
    def build(self, input_shape):
        super().build(input_shape)

    @abstractmethod
    def call(self, inputs):
        raise NotImplementedError

    @abstractmethod
    def form_matrix(self):
        raise NotImplementedError

    @abstractmethod
    def get_core_weights(self):
        raise NotImplementedError

    @abstractmethod
    def reinitialize(self):
        raise NotImplementedError

    @classmethod
    def from_keras(
        cls,
        layer: Layer,
        params: CrossSimParameters,
        bias_rows: int = 0,
    ) -> AnalogLayer:
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
        # This is a little ugly, in general the keras layer should be bases[1]
        # So we are hard coding it until someone needs us to do something
        # different.
        config = layer.get_config()
        del config["params"]
        del config["bias_rows"]

        keras_layer = cls.__bases__[1](**config)

        if layer.built:
            keras_layer.build_from_config(layer.get_build_config())
            if physical_weights:
                keras_layer.set_weights(layer.get_core_weights())
            else:
                keras_layer.set_weights(layer.get_weights())

        return keras_layer
