#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""CrossSim version of N-dimensional torch.nn.Conv[N]d layers.

AnalogConv[N] provides a CrossSim-based forward and backward using Analog MVM backed by
AnalogConvolution.
"""

from __future__ import annotations

from .layer import AnalogLayer

from simulator import CrossSimParameters
from simulator.algorithms.dnn.analog_convolution import (
    AnalogConvolution,
    AnalogConvolution1D,
    AnalogConvolution2D,
    AnalogConvolution3D,
)
from torch import Tensor, from_dlpack
from torch.nn import Conv1d, Conv2d, Conv3d, Parameter
from torch.nn.modules.conv import _ConvNd
from torch.nn.functional import pad
from torch.autograd import Function
from torch import ops


# MRO for Conv layers: Analog1d, AnalogNd, Conv1d, ConvNd, AnalogLayer, Module
class _AnalogConvNd(_ConvNd, AnalogLayer):
    """CrossSim base class for N-dimensional torch.nn.Conv[N]d layers.

    Implementing classes must declare the following attributes:
        core_func:
            A class which will be used to implement self.core. Typically
            AnalogConvolution[N]d
        _forward_padding: An N-length tuple of zeros for padding

    See AnalogLayer for description of CrossSim-specific documentation.
    See torch.nn.Conv[N]d for layer functionality documentation.
    """

    def __init__(
        self,
        params: CrossSimParameters | list[CrossSimParameters],
        # Base ConvNd layer arguments
        in_channels: int,
        out_channels: int,
        # Need int | tuple because this is called before the torch.nn.ConvXd
        # init which handles converting into tuples and error checking
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...],
        padding: str | int | tuple[int, ...],
        dilation: int | tuple[int, ...],
        # skip transposed for now
        # skip output_padding for now
        groups: int,
        bias: bool,
        padding_mode: str,
        device,
        dtype,
        # Additional arguments for AnalogConvNd
        bias_rows: int,
    ) -> None:
        """Initializes AnalogConv[N]d and underlying torch.nn.Conv[N]d layer.

        Args:
            params:
                CrossSimParameters object or list of CrossSimParameters (for layers
                requiring multiple arrays) for the AnalogLinear layer. If a list, the
                length must match the number of arrays used within AnalogCore.
            in_channels: See torch.nn.Conv[N]d in_channels argument.
            out_channels: See See torch.nn.Conv[N]d out_channels argument.
            kernel_size: See torch.nn.Conv[N]d kernel_size argument.
            stride: See torch.nn.Conv[N]d stride argument.
            padding: See torch.nn.Conv[N]d padding argument.
            dilation: See torch.nn.Conv[N]d dilation argument.
            groups: See torch.nn.Conv[N]d groups argument.
            bias: See torch.nn.Conv[N]d bias argument.
            padding_mode: See torch.nn.Conv[N]d padding_mode argument.
            device: See torch.nn.Conv[N]d device argument.
            dtype: See torch.nn.Conv[N]d dtype argument.
            bias_rows:
                Integer indicating the number of rows to use to implement the bias
                within the array. 0 implies a digital bias. Ignored if bias is false.
        """
        device_ = AnalogLayer._set_device(
            device,
            params[0] if isinstance(params, list) else params,
        )

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device_,
            dtype,
        )

        if isinstance(params, CrossSimParameters):
            self.params = params.copy()
        elif isinstance(params, list):
            self.params = params[0].copy()

        # Torch has already handled base incompatibilities
        # Check for CrossSim-specific ones
        if any(d > 1 for d in self.dilation):
            raise NotImplementedError(
                "{} does not support dilated convolutions".format(
                    self.__class__.__name__,
                ),
            )

        if isinstance(params, CrossSimParameters):
            self.params = params.copy()
        elif isinstance(params, list):
            self.params = params[0].copy()

        self.bias_rows = bias_rows

        # Easier to track analog bias than digital bias
        # Avoids conditionals on bias and digital_bias
        self.analog_bias = bias and bias_rows > 0

        self.core = self.core_func(
            self.params,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.groups,
            self.bias_rows,
        )
        self.core.set_matrix(self.form_matrix().detach())

    def form_matrix(self) -> Tensor:
        """Builds 2D weight matrix for programming into the array.

        Returns:
            2D Torch Tensor of the matrix.

        """
        # Intentionally moving this onto the CPU
        weight_ = self.weight.detach().cpu()
        if self.analog_bias:
            return from_dlpack(self.core.form_matrix(weight_, self.bias.detach().cpu()))
        else:
            return from_dlpack(self.core.form_matrix(weight_))

    def forward(self, x: Tensor) -> Tensor:
        """AnalogConv[N]d forward operation.

        See AnalogConvGrad.forward for details.
        """
        # Use the same padding logic as torch.nn.ConvNd.
        # https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py#L453
        # Special-casing zeros just because pad expects "constant" not "zeros"

        if self.padding_mode != "zeros":
            x_ = pad(x, self._reversed_padding_repeated_twice, self.padding_mode)
        else:
            x_ = pad(x, self._reversed_padding_repeated_twice)

        out = AnalogConvGrad.apply(
            x_,
            self.weight,
            self.bias,
            self.core,
            self.bias_rows,
            self.stride,
            # Since we already padded here, we can simply set padding to zeros
            self._foward_padding,
            self.dilation,
            self.output_padding,
            self.groups,
            self.training,
        )

        return out

    def reinitialize(self) -> None:
        """Rebuilds the layer's internal core object.

        Allows parameters to be updated within a layer without rebuilding the
        layer. This will resample all initialization-time errors
        (e.g. programming error)  even if the models were not be changed.
        Alternatively,  reinitialize can be used to directly resample
        initialization-time errors.
        """
        # If we're being called and there is no core (called during init),
        # just bail out, init will handle it.
        if not hasattr(self, "core"):
            return

        # Since bias_rows can change we need to recompute analog_bias

        self.analog_bias = self.bias is not None and self.bias_rows > 0

        # Shape might have changed need to call form_matrix again.
        self.core = self.core_func(
            self.params,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.groups,
            self.bias_rows,
        )
        self.core.set_matrix(self.form_matrix().detach())

    @classmethod
    def from_torch(
        cls,
        layer: _ConvNd,
        params: CrossSimParameters | list[CrossSimParameters],
        bias_rows: int = 0,
    ) -> _AnalogConvNd:
        """Build AnalogConv[N]d from a Conv[N]d layer.

        Creates a new AnalogConv[N]d layer with the same attributes as the original
        torch layer.

        Args:
            layer: torch.nn.Conv[N]d layer to copy.
            params:
                CrossSimParameters object or list of CrossSimParameters (for layers
                requiring multiple arrays) for the AnalogConv[N]d layer. If a list, the
                length must match the number of arrays used within AnalogCore.
            bias_rows:
                Integer indicating the number of analog rows to use for the bias.
                0 indicates a digital bias. Ignored if layer does not have a bias.

        Returns:
            AnalogConv[N]d layer with the same weights and properties as input
            Conv[N]d layer.
        """
        device = AnalogLayer._set_device(
            layer.weight.device,
            params[0] if isinstance(params, list) else params,
        )

        analog_layer = cls(
            params,
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
            layer.bias is not None,
            layer.padding_mode,
            # Adopt the convention that device and dtype are based on the device and
            # dtype of the weight matrix for conversion. Technically misses some
            # possible (but extremely silly) use cases.
            device,
            layer.weight.dtype,
            bias_rows,
        )
        analog_layer.weight = layer.weight
        analog_layer.bias = layer.bias

        return analog_layer

    @classmethod
    def to_torch(
        cls,
        layer: _AnalogConvNd,
        physical_weights: bool = False,
        device=None,
        dtype=None,
    ) -> _ConvNd:
        """Creates a torch Conv[N]d layer from an AnalogConv[N]d.

        Args:
            layer: AnalogConv[N]d layer to copy.
            physical_weights:
                Bool indicating whether the torch layer should have ideal weights or
                weights with programming error applied.
            device:
                The device where the layer will be placed. See torch.device for
                additional documentation. If None, the device will be set based on the
                layer's CrossSimParameters object.
            dtype:
                The dtype of the layer weights. See torch.dtype for additional
                documentation. If None dtype will be set based on AnalogCore.dtype.

        Returns:
            torch.nn.Linear with the same properties and weights (potentially with
            errors) as the AnalogConv[N]d layer.
        """
        if not device:
            if layer.params.simulation.useGPU:
                device = "cuda:{}".format(layer.params.simulation.gpu_id)
            else:
                device = "cpu"

        if not dtype:
            dtype = AnalogLayer._numpy_to_torch_dtype_dict[layer.core.dtype]

        torch_layer = cls.__bases__[1](
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
            layer.bias is not None,
            layer.padding_mode,
            device,
            dtype,
        )
        if physical_weights:
            w, b = layer.get_core_weights()
        else:
            w = layer.weight.clone().detach()
            if layer.bias is not None:
                b = layer.bias.clone().detach()
            else:
                b = None
        torch_layer.weight = Parameter(w)

        # Need this check otherwise adding a Parameter(None) creates an empty
        # tensor. So in this case, gives the layer a bias that is empty.
        if b is not None:
            torch_layer.bias = Parameter(b)

        return torch_layer


# _AnalogConvNd listed first to allow it to handle params and bias_rows
class AnalogConv1d(_AnalogConvNd, Conv1d):
    """CrossSim implementation of torch.nn.Conv1d.

    See AnalogLayer for description of CrossSim-specific documentation.
    See torch.nn.Conv1d for layer functionality documentation.
    """

    def __init__(
        self,
        params: CrossSimParameters,
        # Base Conv1d layer arguments
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int,],
        stride: int | tuple[int,] = 1,
        padding: str | int | tuple[int,] = 0,
        dilation: int | tuple[int,] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        # Additional arguments for AnalogConv1d
        bias_rows: int = 0,
    ) -> None:
        """Initializes AnalogConv1d and underlying torch.nn.Conv1d layer.

        Args:
            params:
                CrossSimParameters object or list of CrossSimParameters (for layers
                requiring multiple arrays) for the AnalogLinear layer. If a list, the
                length must match the number of arrays used within AnalogCore.
            in_channels: See torch.nn.Conv1d in_channels argument.
            out_channels: See See torch.nn.Conv1d out_channels argument.
            kernel_size: See torch.nn.Conv1d kernel_size argument.
            stride: See torch.nn.Conv1d stride argument.
            padding: See torch.nn.Conv1d padding argument.
            dilation: See torch.nn.Conv1d dilation argument.
            groups: See torch.nn.Conv1d groups argument.
            bias: See torch.nn.Conv1d bias argument.
            padding_mode: See torch.nn.Conv1d padding_mode argument.
            device: See torch.nn.Conv3d device argument.
            dtype: See torch.nn.Conv3d dtype argument.
            bias_rows:
                Integer indicating the number of rows to use to implement the bias
                within the array. 0 implies a digital bias. Ignored if bias is false.
        """
        self.core_func = AnalogConvolution1D
        # Rank-entry tuple of zeros for use in forward for padding
        self._foward_padding = (0,)

        super().__init__(
            params,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
            bias_rows,
        )


# _AnalogConvNd listed first to allow it to handle params and bias_rows
class AnalogConv2d(_AnalogConvNd, Conv2d):
    """CrossSim implementation of torch.nn.Conv2d.

    See AnalogLayer for description of CrossSim-specific documentation.
    See torch.nn.Conv2d for layer functionality documentation.
    """

    def __init__(
        self,
        params: CrossSimParameters,
        # Base Conv2d layer arguments
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: str | int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        # Additional arguments for AnalogConv2d
        bias_rows: int = 0,
    ) -> None:
        """Initializes AnalogConv2d and underlying torch.nn.Conv2d layer.

        Args:
            params:
                CrossSimParameters object or list of CrossSimParameters (for layers
                requiring multiple arrays) for the AnalogLinear layer. If a list, the
                length must match the number of arrays used within AnalogCore.
            in_channels: See torch.nn.Conv2d in_channels argument.
            out_channels: See See torch.nn.Conv2d out_channels argument.
            kernel_size: See torch.nn.Conv2d kernel_size argument.
            stride: See torch.nn.Conv2d stride argument.
            padding: See torch.nn.Conv2d padding argument.
            dilation: See torch.nn.Conv2d dilation argument.
            groups: See torch.nn.Conv2d groups argument.
            bias: See torch.nn.Conv2d bias argument.
            padding_mode: See torch.nn.Conv2d padding_mode argument.
            device: See torch.nn.Conv2d device argument.
            dtype: See torch.nn.Conv2d dtype argument.
            bias_rows:
                Integer indicating the number of rows to use to implement the bias
                within the array. 0 implies a digital bias. Ignored if bias is false.
        """
        self.core_func = AnalogConvolution2D
        # Rank-entry tuple of zeros for use in forward for padding
        self._foward_padding = (0, 0)

        super().__init__(
            params,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
            bias_rows,
        )


# _AnalogConvNd listed first to allow it to handle params and bias_rows
class AnalogConv3d(_AnalogConvNd, Conv3d):
    """CrossSim implementation of torch.nn.Conv3d.

    See AnalogLayer for description of CrossSim-specific documentation.
    See torch.nn.Conv3d for layer functionality documentation.
    """

    def __init__(
        self,
        params: CrossSimParameters,
        # Base Conv3d layer arguments
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: str | int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        # Additional arguments for AnalogConv3d
        bias_rows: int = 0,
    ) -> None:
        """Initializes AnalogConv3d and underlying torch.nn.Conv3d layer.

        Args:
            params:
                CrossSimParameters object or list of CrossSimParameters (for layers
                requiring multiple arrays) for the AnalogLinear layer. If a list, the
                length must match the number of arrays used within AnalogCore.
            in_channels: See torch.nn.Conv3d in_channels argument.
            out_channels: See See torch.nn.Conv3d out_channels argument.
            kernel_size: See torch.nn.Conv3d kernel_size argument.
            stride: See torch.nn.Conv3d stride argument.
            padding: See torch.nn.Conv3d padding argument.
            dilation: See torch.nn.Conv3d dilation argument.
            groups: See torch.nn.Conv3d groups argument.
            bias: See torch.nn.Conv3d bias argument.
            padding_mode: See torch.nn.Conv3d padding_mode argument.
            device: See torch.nn.Conv3d device argument.
            dtype: See torch.nn.Conv3d dtype argument.
            bias_rows:
                Integer indicating the number of rows to use to implement the bias
                within the array. 0 implies a digital bias. Ignored if bias is false.
        """
        self.core_func = AnalogConvolution3D
        # Rank-entry tuple of zeros for use in forward for padding
        self._foward_padding = (0, 0, 0)

        super().__init__(
            params,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
            bias_rows,
        )


class AnalogConvGrad(Function):
    """Gradient implementation for CrossSim ConvNd layer.

    Forward direction is implemented using CrossSim. Backward is ideal.
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        core: AnalogConvolution,
        bias_rows: int,
        stride: tuple,
        padding: tuple,
        dilation: tuple,
        output_padding: tuple,
        groups: int,
        training: bool,
    ) -> Tensor:
        """CrossSim-based Conv forward.

        Forward takes both the layer core object as well as the true weights
        and bias to save for the gradient computation. Convolutional layer
        properties are also passed for the backward computation.

        Args:
            ctx: torch gradient context
            x:
                3/4-D torch tensor of the layer input. Trailing dimension must
                match conv.in_channels, conv.kernel_size[0],
                conv.kernel_size[1]).
            weight:
                4D torch tensor of the layer weight. Shape matches
                torch.nn.conv2D
            bias: 1D torch tensor of the layer bias
            core: ConvolutionCore object for the operation
            bias_rows:
                Integer number of rows used for the bias (0 meaning digital
                bias)
            stride: ConvNd.stride parameter
            padding: ConvNd.padding parameter, must be int tuple
            dilation: ConvNd.dilation parameter
            output_padding: ConvNd.output_padding parameter
            groups: ConvNd.groups parameter
            training: ConvNd.training parameter

        Returns:
            2d/3d, 3d/4d, or 4d/5d torch tensor result for 1d, 2d, and 3d inputs
            respectively. Trailing dimension matches torch.nn.ConvNd.
        """
        analog_bias = bias is not None and bool(bias_rows)

        if training:
            w, b = core.get_core_weights()
            w = from_dlpack(w)
            if not analog_bias:
                b = bias
            else:
                b = from_dlpack(b)
            ctx.save_for_backward(x, w, b)
            # Stuff all the convolution parameters into ctx
            ctx.stride = stride
            ctx.padding = padding
            ctx.dilation = dilation
            ctx.output_padding = output_padding
            ctx.groups = groups

        out = from_dlpack(core.apply(x.detach()))

        if bias is not None and not analog_bias:
            # Bias must be expanded from 1D to out.ndim for correct
            # broadcasting. Should be (bias.shape, 1...) with 1s equal to the
            # total number of dimensions of the convolution. This is always
            # weight.ndim - 2 because 2 dims for in and out channels.
            # For batched inputs, we need a leading 1 on the shape and we can
            # detect a batched input if the output has less dimensions than the
            # weights
            if out.ndim < weight.ndim:
                bias_shape = bias.shape + (1,) * (weight.ndim - 2)
            else:
                bias_shape = (1,) + bias.shape + (1,) * (weight.ndim - 2)
            return out + bias.reshape(bias_shape)
        else:
            return out

    @staticmethod
    def backward(ctx, grad_output):
        """Backward implementation of a ConvNd layer.

        Based on internal torch convolution_backward.
        """
        (x, weight, bias) = ctx.saved_tensors

        output_mask = (
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            bias is not None and ctx.needs_input_grad[2],
        )

        gi, gw, gb = ops.aten.convolution_backward(
            # Maybe hacky? convolution_backward expects grad_output, input,
            # and weight to have the same dimension. For unbatched inputs this
            # isn't true, so just unsqueeze them to make it match.
            # Seems to work, not sure if this is the right approach.
            grad_output.unsqueeze(0) if grad_output.ndim < weight.ndim else grad_output,
            x.unsqueeze(0) if x.ndim < weight.ndim else x,
            weight,
            bias.shape if (bias is not None) else None,
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            False,  # transposed
            ctx.output_padding,
            ctx.groups,
            # Output mask determines which gradients to compute in
            # (input, weight, bias) order
            output_mask,
        )

        # No gradients with respect to core, bias_rows, or the other inputs
        return gi, gw, gb, None, None, None, None, None, None, None, None
