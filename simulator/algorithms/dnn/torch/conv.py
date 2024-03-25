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
from simulator.algorithms.dnn.convolution import Convolution
from torch import Tensor, zeros, arange, stack
from torch.nn import Conv2d, Parameter
from torch.autograd import Function
from torch import ops

import numpy as np


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
        if bias_rows > 1:
            raise NotImplementedError("Conv2d does not support multiple bias rows")

        if self.stride[0] != self.stride[1]:
            raise NotImplementedError(
                "AnalogConv2d does not support different horizontal and vertical "
                "strides",
            )

        if self.dilation[0] != 1 or self.dilation[1] != 1:
            raise NotImplementedError(
                "AnalogConv2d does not support dilated convolutions",
            )

        if self.padding_mode != "zeros":
            raise NotImplementedError(
                "AnalogConv2d does not support padding_mode other than 'zeros'",
            )

        # TODO: implement
        if self.padding == "valid":
            raise NotImplementedError("AnalogConv2d does not support 'valid' padding")

        self.params = params.copy()
        self.bias_rows = bias_rows

        # Easier to track analog bias than digital bias
        # Avoids conditionals on bias and digital_bias
        self.analog_bias = bias and bias_rows > 0

        self.depthwise = False
        if self.groups != 1:
            if (
                self.groups == self.in_channels
                and self.out_channels == self.in_channels
            ):
                self.depthwise = True
            else:
                raise NotImplementedError(
                    "AnalogConv2d does not support grouped convolutions other than "
                    "depthwise convolutions (groups = in_channels = out_channels)",
                )

        self.weight_mask = (
            slice(0, out_channels, 1),
            slice(0, self.kernel_size[0] * self.kernel_size[1] * in_channels, 1),
        )
        self.bias_mask = (
            slice(None, None, 1),
            slice(self.weight_mask[1].stop, self.weight_mask[1].stop + bias_rows, 1),
        )

        # Build convParams and sync the params object
        convParams = self._build_conv_dict()
        self._synchronize_params()

        self.core = Convolution(convParams, params=self.params)
        self.core.set_matrix(self.form_matrix().detach().numpy())

    def form_matrix(self) -> Tensor:
        """ """
        bias_rows = self.bias_rows if self.analog_bias else 0
        matrix = zeros(
            (
                self.out_channels,
                self.kernel_size[0] * self.kernel_size[1] * self.in_channels
                + bias_rows,
            ),
        )
        weight_ = self.weight.detach()
        Kx, Ky = self.kernel_size
        Nic = self.in_channels
        Noc = self.out_channels
        groups = self.groups

        # Ensure weight.shape is (Noc, Nic, Kx, Ky)
        if self.weight.shape != (Noc, Nic // groups, Kx, Ky):
            raise ValueError(
                "Expected weight shape",
                (Noc, Nic // groups, Kx, Ky),
                "got",
                self.weight.shape,
            )

        if not self.depthwise:
            for i in range(Noc):
                # For some reason (possibly indexing related?
                # See: https://github.com/pytorch/pytorch/issues/29973 )
                # torch.Tensor indexing is much slower than numpy.
                # It ends up being >5x faster to move things into numpy,
                # do the indexing there and back to a torch tensor vs a
                # native torch.Tensor implementation.
                #
                # Long run this is going to move into Convolution as a shared
                # resource for both Torch and Keras, so for now leaving it
                # hacky and ugly.
                weight_np = weight_.detach().numpy()
                submat = Tensor(
                    np.array(
                        [weight_np[i, k, :, :].flatten() for k in range(Nic)],
                    ).flatten()
                )
                if self.analog_bias:
                    matrix[i, :-1] = submat
                else:
                    matrix[i, :] = submat
        else:
            for i in range(Noc):
                matrix[i, (i * Kx * Ky) : ((i + 1) * Kx * Ky)] = weight_[
                    i,
                    0,
                    :,
                    :,
                ].flatten()

        if self.analog_bias:
            matrix[:, -1] = self.bias

        return matrix

    def forward(self, x: Tensor) -> Tensor:
        """AnalogConv2d forward operation.

        See AnalogConvGrad.forward for details.
        """
        return AnalogConvGrad.apply(
            x,
            self.weight,
            self.bias,
            self.core,
            self.bias_rows,
            self.stride,
            self.padding,
            self.dilation,
            self.output_padding,
            self.groups,
        )

    def get_core_weights(self):
        """Returns weight and biases with programming errors."""
        matrix = self.get_matrix()
        if not self.depthwise:
            weight = matrix[self.weight_mask].reshape(self.weight.shape)
        else:
            weight = zeros(self.weight.shape)
            Kx, Ky = self.kernel_size
            for i in range(self.out_channels):
                weight[i] = matrix[i, (i * Kx * Ky) : ((i + 1) * Kx * Ky)].reshape(
                    (Kx, Ky),
                )
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

        if self.bias_rows > 1:
            raise NotImplementedError

        # Build new convParams and resync the params object
        convParams = self._build_conv_dict()
        self._synchronize_params()

        # Shape might have changedn need to call form_matrix again.
        self.core = Convolution(convParams, params=self.params)
        self.core.set_matrix(self.form_matrix().detach().numpy())

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

    def _build_conv_dict(self) -> dict[str, str | bool]:
        """Create convParam dict for input to Convolution from layer params."""
        convParams = {}
        convParams["Nic"] = self.in_channels
        convParams["Noc"] = self.out_channels
        convParams["Kx"], convParams["Ky"] = self.kernel_size
        convParams["stride"] = self.stride[0]
        convParams["bias_row"] = self.analog_bias

        if isinstance(self.padding, str) and self.padding == "same":
            convParams["sameConv"] = True
        else:
            convParams["sameConv"] = False
            convParams["px_0"] = self.padding[0]
            convParams["px_1"] = self.padding[0]
            convParams["py_0"] = self.padding[1]
            convParams["py_1"] = self.padding[1]

        return convParams

    def _synchronize_params(self) -> None:
        """Synchronize the params object from the internal parameters."""
        # Don't modify the following params:
        # x_par, y_par, weight_reoder, and conv_matmul are pure inputs
        # Nwindows is derived in Convolution

        if isinstance(self.params, CrossSimParameters):
            self.params.simulation.convolution.is_conv_core = True
            self.params.simulation.convolution.Kx = self.kernel_size[0]
            self.params.simulation.convolution.Ky = self.kernel_size[1]
            # We do not currently support different X, Y strides
            self.params.simulation.convolution.stride = self.stride[0]
            self.params.simulation.convolution.Noc = self.out_channels
            self.params.simulation.convolution.Nic = self.out_channels
            self.params.simulation.convolution.bias_row = self.analog_bias

        elif isinstance(self.params, list):
            for p in self.params:
                p.simulation.convolution.is_conv_core = True
                p.simulation.convolution.Kx = self.kernel_size[0]
                p.simulation.convolution.Ky = self.kernel_size[1]
                p.simulation.convolution.stride = self.stride[0]
                p.simulation.convolution.Noc = self.out_channels
                p.simulation.convolution.Nic = self.out_channels
                p.simulation.convolution.bias_row = self.analog_bias


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
        core: Convolution,
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
            padding: Conv2D.padding parameter
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

        # Mimicking the baseline torch behavior, if the input is 4D (batch >1)
        # the result is 4D, even if the batch size (leading dimension) is 1.
        # If the input is 3D (no batch) the result will be 3D.

        # Prepare a list of individual convolution ops, convolution already
        # uses matul so just implement a series of conv ops and stack
        if x.ndim == 3:
            return AnalogConvGrad._apply_convolution(x, core, bias, analog_bias)
        else:
            # Possibly some faster approach using reshape sorcery
            # Simple stacking probably fast enough for now
            # TODO: Not actually fast enough, do the sorcery
            return stack(
                [
                    AnalogConvGrad._apply_convolution(x[i], core, bias, analog_bias)
                    for i in range(x.shape[0])
                ],
            )

    @staticmethod
    def _apply_convolution(
        x: Tensor,
        core: Convolution,
        bias: Tensor | None,
        analog_bias: bool,
    ) -> Tensor:
        out = Tensor(core.apply_convolution(x.detach()))

        if bias is not None and not analog_bias:
            return out + bias.expand(*reversed(out.shape)).permute(
                *arange(out.ndim - 1, -1, -1),
            )
        else:
            return out

    @staticmethod
    def backward(ctx, grad_output):
        """Ideal backward implementation of a Conv2D layer.

        Uses ideal (unquantized and unnoised) weights (and biases with analog
        bias) for the gradients. Based on internal torch convolution_backward.
        """
        if ctx.padding == "same":
            raise NotImplementedError(
                "AnalogConv2d.backward does not support 'same' padding.",
            )

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
