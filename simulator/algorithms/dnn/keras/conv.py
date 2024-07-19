#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from __future__ import annotations

from .layer import AnalogLayer

from simulator.algorithms.dnn.analog_convolution import (
    AnalogConvolution1D,
    AnalogConvolution2D,
    AnalogConvolution3D,
)

from keras.layers import Conv1D, Conv2D, Conv3D
from keras import ops
import numpy as np


class AnalogBaseConv(AnalogLayer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Keras has already handled base incompatibliities
        # Check for CrossSim-specific ones
        if any(d > 1 for d in self.dilation_rate):
            raise NotImplementedError(
                "AnalogConv2D does not support dilated convolutions",
            )

    def build(self, input_shape) -> None:
        super().build(input_shape)

        self.core = self.core_func(**self.conv_dict())
        self.core.set_matrix(self.form_matrix())

    def call(self, inputs):
        # put shape into a consistent shape for "same" padding
        input_ = ops.transpose(inputs, self._pre_call_input_order)

        # padding logic based on the jax.lax padtype_to_pads function
        # https://github.com/google/jax/blob/main/jax/_src/lax/lax.py#L4839
        if self.padding == "same":
            # Explicitly convert this to a tuple for ops.negative
            in_shape = tuple(input_.shape[2:])
            # Ceiling division:
            out_shape = -ops.floor_divide(ops.negative(in_shape), self.strides)

            pad_sizes = (
                max(d, 0)
                for d in (out_shape - 1) * self.strides + self.kernel_size - in_shape
            )
            pad_width = [
                (pad_size // 2, pad_size - pad_size // 2) for pad_size in pad_sizes
            ]
            # pad_width only covers the kernel dims, need to add no
            # padding for the leading dimensions (batch, in_channels)
            input_ = ops.pad(input_, [(0, 0), (0, 0), *pad_width])

        out = self.core.apply_convolution(input_)

        if self.use_bias and not self.analog_bias:
            # Since everything is in channel_first order at this point we don't
            # need to handle both possibilities
            # Code taken from keras base_conv bias reshaping
            bias_shape = (1, self.filters) + (1,) * self.rank
            out += ops.reshape(self.bias, bias_shape)

        out = ops.transpose(out, self._post_call_output_order)

        if self.activation is not None:
            out = self.activation(out)
        return out

    def form_matrix(self):
        if self.use_bias:
            weight, bias = self.get_weights()
        else:
            weight = self.get_weights()[0]
            bias = None

        return self.core.form_matrix(ops.transpose(weight, self._weight_order), bias)

    def conv_dict(self):
        raise NotImplementedError(
            "AnalogBaseConv should not be instantiated directly.",
            "Subclasses must implement conv_dict for the convolution type.",
        )

    def get_core_weights(self):
        ...

    def reinitialize(self):
        """Creates a new core object from layer and CrossSimParameters."""
        # Since bias_rows can change we need to recompute analog_bias

        self.analog_bias = self.bias is not None and self.bias_rows > 0

        # Shape might have changed need to call form_matrix again.
        self.core = self.core_func(**self.conv_dict())
        self.core.set_matrix(self.form_matrix())


class AnalogConv1D(AnalogBaseConv, Conv1D):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Set up the Conv1D specific values
        # This is the function used for the actual computation and assorted
        # tuples used for reordering for consistency with the backend compute
        # functions
        self.core_func = AnalogConvolution1D

        if self.data_format == "channels_first":
            self._pre_call_input_order = (0, 1, 2)
            self._post_call_output_order = (0, 1, 2)
        elif self.data_format == "channels_last":
            self._pre_call_input_order = (0, 2, 1)
            self._post_call_output_order = (0, 2, 1)

        self._weight_order = (2, 1, 0)

    def call(self, inputs):
        if self.padding == "causal":
            inputs = ops.pad(inputs, self._compute_causal_padding())
        return super().call(inputs)

    def conv_dict(self) -> dict[str, int | tuple[int, ...]]:
        d = {
            "params": self.params,
            "Nic": self.input_spec.axes[-1]
            if self.data_format == "channels_last"
            else self.input_spec.axes[1],
            "Noc": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.strides,
            "dilation": self.dilation_rate,
            "groups": self.groups,
            "bias_rows": self.bias_rows,
        }

        return d

    def get_core_weights(self):
        """Returns weight and biases with programming errors."""
        matrix = self.get_matrix()
        tmat = matrix.T

        if self.data_format == "channels_last":
            in_channels = self.input_spec.axes[-1]
        else:
            in_channels = self.input_spec.axes[1]
        inputs = ops.prod(self.kernel_size) * in_channels
        input_range = inputs // self.groups

        if self.groups <= 1:
            weight = tmat[:inputs].reshape(
                -1,
                *self.kernel_size,
                self.filters,
            )
        else:
            layer_counter = 0
            final_mat = []
            # When self.groups = g > 1, the rows in the matrix are split into g rows.
            # In order for the transposeto work, we need to combine the split rows
            # back together. This loop steps through the transposed matrix and combines
            # all of the corresponding split rows, then vstacks them back together
            for i in range(input_range):
                for g in range(self.groups):
                    layer_counter += tmat[i + (g * input_range)]
                final_mat.append(layer_counter)
                layer_counter = 0
            final_mat = np.vstack(final_mat)
            weight = final_mat.reshape(
                -1,
                *self.kernel_size,
                self.filters,
            )

        weight = np.transpose(
            weight,
            (
                1,
                0,
                2,
            ),
        )
        wlist = [weight]

        if self.use_bias:
            if self.analog_bias:
                bias = tmat[
                    slice(
                        inputs,
                        (inputs + self.bias_rows),
                        1,
                    )
                ].sum(0)
            else:
                bias = self.bias.numpy()
            wlist.append(bias)
        return wlist


class AnalogConv2D(AnalogBaseConv, Conv2D):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Set up the Conv2D specific values
        # This is the function used for the actual computation and assorted
        # tuples used for reordering for consistency with the backend compute
        # functions
        self.core_func = AnalogConvolution2D

        if self.data_format == "channels_first":
            self._pre_call_input_order = (0, 1, 2, 3)
            self._post_call_output_order = (0, 1, 2, 3)
        elif self.data_format == "channels_last":
            self._pre_call_input_order = (0, 3, 1, 2)
            self._post_call_output_order = (0, 2, 3, 1)

        self._weight_order = (3, 2, 0, 1)

    def conv_dict(self) -> dict[str, int | tuple[int, ...]]:
        d = {
            "params": self.params,
            "Nic": self.input_spec.axes[-1]
            if self.data_format == "channels_last"
            else self.input_spec.axes[1],
            "Noc": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.strides,
            "dilation": self.dilation_rate,
            "groups": self.groups,
            "bias_rows": self.bias_rows,
        }

        return d

    def get_core_weights(self):
        """Returns weight and biases with programming errors."""
        matrix = self.get_matrix()
        tmat = matrix.T

        if self.data_format == "channels_last":
            in_channels = self.input_spec.axes[-1]
        else:
            in_channels = self.input_spec.axes[1]
        inputs = ops.prod(self.kernel_size) * in_channels
        input_range = inputs // self.groups

        if self.groups <= 1:
            weight = tmat[:inputs].reshape(
                -1,
                *self.kernel_size,
                self.filters,
            )
        else:
            layer_counter = 0
            final_mat = []
            # When self.groups = g > 1, the rows in the matrix are split into g rows.
            # In order for the transposeto work, we need to combine the split rows
            # back together. This loop steps through the transposed matrix and combines
            # all of the corresponding split rows, then vstacks them back together
            for i in range(input_range):
                for g in range(self.groups):
                    layer_counter += tmat[i + (g * input_range)]
                final_mat.append(layer_counter)
                layer_counter = 0
            final_mat = np.vstack(final_mat)
            weight = final_mat.reshape(
                -1,
                *self.kernel_size,
                self.filters,
            )

        weight = np.transpose(weight, (1, 2, 0, 3))
        wlist = [weight]

        if self.use_bias:
            if self.analog_bias:
                bias = tmat[
                    slice(
                        inputs,
                        (inputs + self.bias_rows),
                        1,
                    )
                ].sum(0)
            else:
                bias = self.bias.numpy()
            wlist.append(bias)
        return wlist


class AnalogConv3D(AnalogBaseConv, Conv3D):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Set up the Conv3D specific values
        # This is the function used for the actual computation and assorted
        # tuples used for reordering for consistency with the backend compute
        # functions
        self.core_func = AnalogConvolution3D

        if self.data_format == "channels_first":
            self._pre_call_input_order = (0, 1, 2, 3, 4)
            self._post_call_output_order = (0, 1, 2, 3, 4)
        elif self.data_format == "channels_last":
            self._pre_call_input_order = (0, 4, 1, 2, 3)
            self._post_call_output_order = (0, 2, 3, 4, 1)

        self._weight_order = (4, 3, 0, 1, 2)

    def conv_dict(self) -> dict[str, int | tuple[int, ...]]:
        d = {
            "params": self.params,
            "Nic": self.input_spec.axes[-1]
            if self.data_format == "channels_last"
            else self.input_spec.axes[1],
            "Noc": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.strides,
            "dilation": self.dilation_rate,
            "groups": self.groups,
            "bias_rows": self.bias_rows,
        }

        return d

    def get_core_weights(self):
        """Returns weight and biases with programming errors."""
        matrix = self.get_matrix()
        tmat = matrix.T

        if self.data_format == "channels_last":
            in_channels = self.input_spec.axes[-1]
        else:
            in_channels = self.input_spec.axes[1]
        inputs = ops.prod(self.kernel_size) * in_channels
        input_range = inputs // self.groups

        if self.groups <= 1:
            weight = tmat[:inputs].reshape(
                -1,
                *self.kernel_size,
                self.filters,
            )
        else:
            layer_counter = 0
            final_mat = []
            # When self.groups = g > 1, the rows in the matrix are split into g rows.
            # In order for the transposeto work, we need to combine the split rows
            # back together. This loop steps through the transposed matrix and combines
            # all of the corresponding split rows, then vstacks them back together
            for i in range(input_range):
                for g in range(self.groups):
                    layer_counter += tmat[i + (g * input_range)]
                final_mat.append(layer_counter)
                layer_counter = 0
            final_mat = np.vstack(final_mat)
            weight = final_mat.reshape(
                -1,
                *self.kernel_size,
                self.filters,
            )

        weight = np.transpose(weight, (1, 2, 3, 0, 4))
        wlist = [weight]

        if self.use_bias:
            if self.analog_bias:
                bias = tmat[
                    slice(
                        inputs,
                        (inputs + self.bias_rows),
                        1,
                    )
                ].sum(0)
            else:
                bias = self.bias.numpy()
            wlist.append(bias)
        return wlist
