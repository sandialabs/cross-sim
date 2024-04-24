#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""CrossSim version of Torch.nn.Conv2d.

AnalogConv2d provides a CrossSim-based forward using Analog MVM backed by
ConvolutionCore. Backward is implemented using an ideal torch backward
(unquantized). The layer supports a single analog bias row (no split bias rows)
"""

from __future__ import annotations

from .layer import AnalogLayer

from simulator import CrossSimParameters
from simulator.algorithms.dnn.analog_convolution import (
    AnalogConvolution,
    AnalogConvolution2D,
)
from torch import Tensor, zeros, arange, from_dlpack
from torch.nn import Conv2d, Parameter
from torch.nn.functional import pad
from torch.autograd import Function
from torch import ops


class AnalogConv2d(Conv2d, AnalogLayer):
    """ """

    def __init__(
        self,
        params: CrossSimParameters,
        # Base Conv2D layer arguments
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: str | int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        # Additional arguments for AnalogConv2d
        bias_rows: int = 0,
    ) -> None:
        """ """
        # TODO: Do we need to handle device and dtype
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
        )

        # Torch has already handled base incompatibilities
        # Check for CrossSim-specific ones

        if self.dilation[0] != 1 or self.dilation[1] != 1:
            raise NotImplementedError(
                "AnalogConv2d does not support dilated convolutions",
            )

        self.params = params.copy()
        self.bias_rows = bias_rows

        # Easier to track analog bias than digital bias
        # Avoids conditionals on bias and digital_bias
        self.analog_bias = bias and bias_rows > 0

        self.weight_mask = (
            slice(0, out_channels, 1),
            slice(0, self.kernel_size[0] * self.kernel_size[1] * in_channels, 1),
        )
        self.bias_mask = (
            slice(None, None, 1),
            slice(self.weight_mask[1].stop, self.weight_mask[1].stop + bias_rows, 1),
        )

        self.core = AnalogConvolution2D(
            self.params,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.groups,
            int(self.bias_rows and bias),
        )
        self.core.set_matrix(self.form_matrix().detach())

    def form_matrix(self) -> Tensor:
        """ """
        # AnalogConvolution expects weight matrices with the order
        # (Kx, Ky, Nic, Noc), torch uses (Noc, Nic, Kx, Ky)
        # Intentionally moving this onto the CPU
        weight_ = self.weight.detach().cpu()
        if self.analog_bias:
            return from_dlpack(self.core.form_matrix(weight_, self.bias.detach().cpu()))
        else:
            return from_dlpack(self.core.form_matrix(weight_))

    def forward(self, x: Tensor) -> Tensor:
        """AnalogConv2d forward operation.

        See AnalogConvGrad.forward for details.
        """
        # Use the same padding logic as torch.nn.Conv2d.
        # https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py#L453
        # Special-casing zeros just because pad expects "constant" not "zeros"

        if self.padding_mode != "zeros":
            x_ = pad(x, self._reversed_padding_repeated_twice, self.padding_mode)
        else:
            x_ = pad(x, self._reversed_padding_repeated_twice)

        # Now use x_ to set Nwindows
        # This is dangerous because params can be copied so the params object
        # won't be reflected but all current uses of Nwindows refer to the
        # object directly.
        # Still, be very careful with this pattern, its kind of a hack
        if self.padding == "same":
            Nox, Noy = x_.shape[0] // self.stride[0], x_.shape[1] // self.stride[1]
        else:
            Nox = 1 + (x_.shape[0] - self.kernel_size[0]) // self.stride[0]
            Noy = 1 + (x_.shape[1] - self.kernel_size[1]) // self.stride[1]

        for row_list in self.core.cores:
            for core in row_list:
                core.params.simulation.convolution.Nwindows = Nox * Noy

        out = AnalogConvGrad.apply(
            x_,
            self.weight,
            self.bias,
            self.core,
            self.bias_rows,
            self.stride,
            # Since we already padded here, we can simply set padding to (0,0)
            (0, 0),
            self.dilation,
            self.output_padding,
            self.groups,
        )

        return out

    def get_core_weights(self):
        """Returns weight and biases with programming errors."""
        matrix = self.get_matrix()
        if self.groups == 1:
            weight = matrix[self.weight_mask].reshape(self.weight.shape)
        else:
            weight = zeros(self.weight.shape)
            weights_per_out = (
                self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
            )
            outs_per_group = self.out_channels // self.groups
            for i in range(self.out_channels):
                group = i // outs_per_group
                weight[i] = matrix[
                    i,
                    group * weights_per_out : (group + 1) * weights_per_out,
                ].reshape(self.weight.shape[1:])
        if self.analog_bias:
            bias = matrix[self.bias_mask].sum(1)
        else:
            bias = self.bias
        return (weight, bias)

    def reinitialize(self) -> None:
        """Creates a new core object from layer and CrossSimParameters."""
        # If we're being called and there is no core (called during init),
        # just bail out, init will handle it.
        if not hasattr(self, "core"):
            return

        # Since bias_rows can change we need to recompute analog_bias

        self.analog_bias = self.bias is not None and self.bias_rows > 0

        # Shape might have changed need to call form_matrix again.
        self.core = AnalogConvolution2D(
            self.params,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.groups,
            int(self.bias_rows and self.bias is not None),
        )
        self.core.set_matrix(self.form_matrix().detach())

    @classmethod
    def from_torch(
        cls,
        layer: Conv2d,
        params: CrossSimParameters,
        bias_rows: int = 0,
    ) -> AnalogConv2d:
        """ """
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
            bias_rows,
        )
        analog_layer.weight = layer.weight
        analog_layer.bias = layer.bias

        return analog_layer

    @classmethod
    def to_torch(cls, layer: AnalogConv2d, physical_weights: bool = False):
        """Creates a torch Conv2d layer from AnalogConv2d.

        Args:
            layer: AnalogConv2d layer to be converted
            physical_weights:
                Bool indicating whether the torch layer should have ideal or
                weights with programming error applied.
        """
        torch_layer = Conv2d(
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
            layer.bias is not None,
            layer.padding_mode,
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


class AnalogConvGrad(Function):
    """Gradient implementation for CrossSim Conv2D layer.

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
            stride: Conv2D.stride parameter
            padding: Conv2D.padding parameter, must be int tuple
            dilation: Conv2D.dilation parameter
            output_padding: Conv2D.output_padding parameter
            groups: Conv2D.groups parameter

        Returns:
            3/4-D torch tensor result. Trailing dimension matches torch.Conv2D.
        """
        ctx.save_for_backward(x, weight, bias)
        # Stuff all the convolution parameters into ctx
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.output_padding = output_padding
        ctx.groups = groups

        analog_bias = bias is not None and bool(bias_rows)

        out = from_dlpack(core.apply_convolution(x.detach()))

        if bias is not None and not analog_bias:
            if out.ndim == 3:
                return out + bias.expand(*reversed(out.shape)).permute(
                    *arange(out.ndim - 1, -1, -1),
                )
            elif out.ndim == 4:
                return out + bias.expand(
                    (out.shape[0], out.shape[3], out.shape[2], out.shape[1]),
                ).permute(0, 3, 2, 1)
        else:
            return out

    @staticmethod
    def backward(ctx, grad_output):
        """Ideal backward implementation of a Conv2D layer.

        Uses ideal (unquantized and unnoised) weights (and biases with analog
        bias) for the gradients. Based on internal torch convolution_backward.
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
        return gi, gw, gb, None, None, None, None, None, None, None
