#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""CrossSim version of N-dimensional keras.layers.[Depthwise]Conv[N]D layers.

[Depthwise]AnalogConv[N]D provides a CrossSim-based forward using Analog MVM backed by
AnalogConvolution.
"""

from __future__ import annotations

from .layer import AnalogLayer

from simulator.algorithms.dnn.analog_convolution import (
    AnalogConvolution1D,
    AnalogConvolution2D,
    AnalogConvolution3D,
)

from keras.layers import Conv1D, Conv2D, Conv3D, DepthwiseConv1D, DepthwiseConv2D
from keras import ops
from tensorflow.experimental.dlpack import from_dlpack

import numpy.typing as npt


class AnalogBaseConv(AnalogLayer):
    """CrossSim base class for N-dimensional keras.layer.[Depthwise]Conv[N]D layers.

    Implementing classes must declare the following attributes:
        core_func:
            A class which will be used to implement self.core. Typically
            AnalogConvolution[N]d
        _pre_call_input_order:
            Tuple of dimension orders to convert inputs from keras specified order into
            order expected by core_func call
        _post_call_output_order:
            Tuple of dimension orders to put outputs from core_func call into keras
            specified order
        _weight_order:
            Tuple of dimension orders to convert weight matrix from keras specified
            order into order expected by core_func form_matrix
        _get_core_weight_order:
            Tuple of dimension orders to convert weights from core_func matrix order
            into keras specified order for get_core_weights function

    See AnalogLayer for description of CrossSim-specific documentation.
    See keras.layer.[Depthwise]Conv[N]D for layer functionality documentation.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes AnalogConv[N]D and underlying keras.layer.Conv[N]D layer.

        See AnalogLayer for description of CrossSim-specific documentation.
        See keras.layer.Conv[N]D for layer functionality documentation.
        """
        super().__init__(**kwargs)

        # Keras has already handled base incompatibliities
        # Check for CrossSim-specific ones
        if any(d > 1 for d in self.dilation_rate):
            raise NotImplementedError(
                "{} does not support dilated convolutions".format(
                    self.__class__.__name__,
                ),
            )

    def build(self, input_shape) -> None:
        """Create layer's core using core_func.

        Args:
            input_shape: keras input_shape object
        """
        super().build(input_shape)

        self.core = self.core_func(**self._conv_dict())
        self.core.set_matrix(self.form_matrix())

    def call(self, inputs: npt.ArrayLike) -> npt.ArrayLike:
        """Convolution layer forward operation."""
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

        # Purely CPU implementations of keras (when no GPU is avaliable)
        # cause a pointer misalignment when using dlpack, so only use dlpack for when
        # part of the application is resident on the GPU. Additionally because Keras
        # uses a non-default stream for gpu operations an explicit stream synchronize
        # is required to avoid a race condition.
        out = self.core.apply(input_)
        if self.useGPU:
            out = from_dlpack(out.toDlpack())
            self.stream.synchronize()

        if self.use_bias and not self.analog_bias:
            # Since everything is in channel_first order at this point we don't
            # need to handle both possibilities
            # Code derived from keras base_conv bias reshaping
            bias_shape = (1, out.shape[1]) + (1,) * self.rank
            out += ops.reshape(self.bias, bias_shape)

        out = ops.transpose(out, self._post_call_output_order)

        if self.activation is not None:
            out = self.activation(out)
        return out

    def form_matrix(self) -> npt.NDArray:
        """Builds 2D weight matrix for programming into the array.

        Matrix formation is deferred to the internal AnalogConvolution.

        Returns:
            2D Numpy Array of the matrix including analog bias if using.

        """
        if self.use_bias:
            weight, bias = self.get_weights()
        else:
            weight = self.get_weights()[0]
            bias = None

        return self.core.form_matrix(ops.transpose(weight, self._weight_order), bias)

    def get_core_weights(self) -> list[npt.NDArray]:
        """Gets the weight and bias values with errors applied.

        Returns:
            List of numpy arrays with errors applied. CrossSim version of get_weights
        """
        weight, bias = self.core.get_core_weights()
        weight = weight.transpose(self._get_core_weight_order)

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

        self.analog_bias = self.bias is not None and self.bias_rows > 0

        # Shape might have changed need to call form_matrix again.
        self.core = self.core_func(**self._conv_dict())
        self.core.set_matrix(self.form_matrix())

    def _conv_dict(self) -> dict[str, int | tuple[int, ...]]:
        """Maps keras attribute names to the arguments expected by AnalogConvolution."""
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


class AnalogDepthwiseConv(AnalogBaseConv):
    """CrossSim base class for N-dimensional keras.layer.DepthwiseConv[N]D layers.

    Keras depthwise convolutions have some small shape and attribute differences vs
    conventional convolution layers. This handles the differences, all CrossSim-specific
    attributes and methods are the same as AnalogBaseConv.

    """

    def form_matrix(self) -> npt.NDArray:
        """Builds 2D weight matrix for programming into the array.

        Matrix formation is deferred to the internal AnalogConvolution. DepthwiseConvs
        use a different shape convention than standard convolutions so additional
        reshaping is needed to make compatible with AnalogConvolution asusmptions.

        Returns:
            2D Numpy Array of the matrix including analog bias if using.

        """
        if self.use_bias:
            weight, bias = self.get_weights()
        else:
            weight = self.get_weights()[0]
            bias = None

        weight_ = weight.reshape(*self.kernel_size, -1, self.core.Noc)
        return self.core.form_matrix(ops.transpose(weight_, self._weight_order), bias)

    def get_core_weights(self) -> list[npt.NDArray]:
        """Gets the weight and bias values with errors applied.

        Returns:
            List of numpy arrays with errors applied. CrossSim version of get_weights.
        """
        weight, bias = self.core.get_core_weights()

        weight = weight.transpose(self._get_core_weight_order)
        weight = weight.reshape((*self.kernel_size, -1, self.depth_multiplier))

        if self.use_bias:
            if not self.analog_bias:
                bias = self.bias.numpy()
            return [weight, bias]
        else:
            return [weight]

    def _conv_dict(self) -> dict[str, int | tuple[int, ...]]:
        """Maps keras attribute names to the arguments expected by AnalogConvolution."""
        Nic = (
            self.input_spec.axes[-1]
            if self.data_format == "channels_last"
            else self.input_spec.axes[1]
        )

        d = {
            "params": self.params,
            "Nic": Nic,
            "Noc": Nic * self.depth_multiplier,
            "kernel_size": self.kernel_size,
            "stride": self.strides,
            "dilation": self.dilation_rate,
            "groups": Nic,
            "bias_rows": self.bias_rows,
        }

        return d


class AnalogBaseConv1D:
    """Base class for all AnalogConv1Ds (Depthwise and Conventional)."""

    def __init__(self, **kwargs) -> None:
        """Initialize the core func and reordering tuples for 1D conv layers."""
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
        self._get_core_weight_order = (2, 1, 0)


class AnalogBaseConv2D:
    """Base class for all AnalogConv2Ds (Depthwise and Conventional)."""

    def __init__(self, **kwargs) -> None:
        """Initialize the core func and reordering tuples for 2D conv layers."""
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
        self._get_core_weight_order = (2, 3, 1, 0)


class AnalogConv1D(AnalogBaseConv, AnalogBaseConv1D, Conv1D):
    """CrossSim implemenation of Keras Conv1D layer."""

    def call(self, inputs: npt.ArrayLike) -> npt.ArrayLike:
        """Conv1D forward operation.

        Conv1D layers support a special form of padding ("causal") which must be
        specially handled.
        """
        if self.padding == "causal":
            inputs = ops.pad(inputs, self._compute_causal_padding())
        return super().call(inputs)


class AnalogDepthwiseConv1D(AnalogDepthwiseConv, AnalogBaseConv1D, DepthwiseConv1D):
    """CrossSim implemenation of Keras DepthwiseConv1D layer."""


class AnalogConv2D(AnalogBaseConv, AnalogBaseConv2D, Conv2D):
    """CrossSim implemenation of Keras Conv2D layer."""


class AnalogDepthwiseConv2D(AnalogDepthwiseConv, AnalogBaseConv2D, DepthwiseConv2D):
    """CrossSim implemenation of Keras DepthwiseConv2D layer."""


class AnalogConv3D(AnalogBaseConv, Conv3D):
    """CrossSim implemenation of Keras Conv3D layer."""

    def __init__(self, **kwargs) -> None:
        """Initializes AnalogConv3D and underlying keras.layer.Conv3D layer.

        See AnalogLayer for description of CrossSim-specific documentation.
        See AnalogBaseConv for CrossSim Conv specific documentation.
        See keras.layer.Conv3D for layer functionality documentation.
        """
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
        self._get_core_weight_order = (2, 3, 4, 1, 0)
