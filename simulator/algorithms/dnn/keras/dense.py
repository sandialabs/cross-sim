#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#


from __future__ import annotations

from .layer import AnalogLayer
from simulator import AnalogCore

from keras.layers import Dense
from keras import ops


class AnalogDense(AnalogLayer, Dense):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def build(self, input_shape) -> None:
        super().build(input_shape)
        self.core = AnalogCore(self.form_matrix(), self.params)

    def call(self, inputs):
        if inputs.ndim < 3:
            x = self._apply_linear(inputs)
        else:
            x = self._apply_linear(
                ops.reshape(inputs, (-1, inputs.shape[-1])),
            ).reshape(
                (*inputs.shape[:-1], -1),
            )

        if self.activation is not None:
            x = self.activation(x)
        return x

    def form_matrix(self):
        if not self.analog_bias:
            return self.get_weights()[0]
        else:
            bias_expanded = self.get_weights()[1] / self.bias_rows
            return ops.vstack(
                (self.get_weights()[0], ops.tile(bias_expanded, (self.bias_rows, 1))),
            )

    def get_core_weights(self):
        """Returns weight and biases with programming errors."""
        matrix = self.get_matrix()
        weight = matrix[: self.get_weights()[0].shape[0]]
        wlist = [weight]
        if self.use_bias is True:
            bias = self.bias.numpy()
            wlist.append(bias)
        return wlist

    def reinitialize(self):
        """Creates a new core object from layer and CrossSimParameters."""
        # Since bias_rows can change we need to recompute analog_bias
        # Also means we can't just the matrix from the old core, need to call
        # form_matrix again.
        self.analog_bias = self.bias is not None and self.bias_rows > 0
        self.core = AnalogCore(self.form_matrix(), self.params)

    def _apply_linear(self, x):
        if not self.analog_bias:
            out = self.core.rdot(x)
            if self.use_bias:
                return out + self.bias
            else:
                return out
        else:
            return self.core.rdot(
                ops.hstack((x, ops.ones((x.shape[0], self.bias_rows)))),
            )
