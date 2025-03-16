#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""CrossSim version of keras.layer.Dense.

AnalogDense provides a CrossSim-based forward using analog MVM.
"""

from __future__ import annotations

from .layer import AnalogLayer
from simulator.algorithms.dnn.analog_linear import AnalogLinear

from keras.layers import Dense
from keras import ops

from tensorflow.experimental.dlpack import from_dlpack

import numpy.typing as npt


class AnalogDense(AnalogLayer, Dense):
    """CrossSim implementation of keras.layer.Dense.

    See AnalogLayer for description of CrossSim-specific documentation.
    See keras.layer.Dense for layer functionality documentation.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes AnalogLinear and underlying keras.layer.Dense layer."""
        super().__init__(**kwargs)

        # Initialize the counter for input profiling
        self.last_input = 0

    def build(self, input_shape) -> None:
        """Create layer's AnalogCore.

        Args:
            input_shape: keras input_shape object
        """
        super().build(input_shape)

        self.core = AnalogLinear(
            self.params,
            self.input_spec.axes[-1],
            self.units,
            self.bias_rows,
        )
        self.core.set_matrix(self.form_matrix())

    def call(self, inputs: npt.ArrayLike) -> npt.NDArray:
        """Dense layer forward operation."""
        if inputs.ndim < 3:
            input_ = inputs
        else:
            input_ = ops.reshape(inputs, (-1, inputs.shape[-1]))

        # There are two considerations with the following code:
        # 1) Purely CPU implementations of keras (when no GPU is avaliable)
        # cause a pointer misalignment when using dlpack, so only use dlpack for when
        # part of the application is resident on the GPU. Additionally because Keras
        # uses a non-default stream for gpu operations an explicit stream synchronize
        # is required to avoid a race condition.
        # 2) Tensorflow from_dlpack doesn't support some stride patterns, specifically
        # those coming from transposed tensors. Since AnalogLinear.apply uses
        # transposes we need to first transpose the result before from_dlpack will work.
        # Importantly the stream synchronize has to happen before we undo the transpose
        # since it must be before any subsequent keras operations on the result.
        out = self.core.apply(input_)
        if self.useGPU:
            out = from_dlpack(out.T.toDlpack())
            self.stream.synchronize()
            out = ops.transpose(out)

        if self.use_bias and not self.analog_bias:
            out += self.bias

        if self.activation is not None:
            out = self.activation(out)

        if inputs.ndim < 3:
            return out
        else:
            return ops.reshape(out, (*inputs.shape[:-1], -1))

    def form_matrix(self) -> npt.NDArray:
        """Builds 2D weight matrix for programming into the array.

        Returns:
            2D Numpy Array of the matrix including analog bias if using.

        """
        if self.use_bias:
            weight, bias = self.get_weights()
        else:
            weight = self.get_weights()[0]
            bias = None

        return self.core.form_matrix(weight.T, bias)

    def get_core_weights(self) -> list[npt.NDArray]:
        """Gets the weight and bias values with errors applied.

        Returns:
            List of numpy arrays with errors applied. CrossSim version of get_weights.
        """
        weight, bias = self.core.get_core_weights()
        weight = weight.transpose((1, 0))

        if self.use_bias:
            if not self.analog_bias:
                bias = self.bias.numpy()
            return [weight, bias]
        else:
            return [weight]

    def reinitialize(self) -> None:
        """Rebuilds the layer's internal core object.

        Allows parameters to be updated within a layer without rebuilding the
        layer. This will resample all initialization-time errors
        (e.g. programming error)  even if the models were not be changed.
        Alternatively,  reinitialize can be used to directly resample
        initialization-time errors.
        """
        # Since bias_rows can change we need to recompute analog_bias
        # Also means we can't just the matrix from the old core, need to call
        # form_matrix again.
        self.analog_bias = self.bias is not None and self.bias_rows > 0
        self.core = AnalogLinear(
            self.params,
            self.input_spec.axes[-1],
            self.units,
            self.bias_rows,
        )
        self.core.set_matrix(self.form_matrix())
