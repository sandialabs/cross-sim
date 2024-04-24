#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""CrossSim version of Torch.nn.Linear.

AnalogLinear provides a CrossSim-based forward using analog MVM. Backward is
implemented using an ideal torch backward (unquantized). The layer supports a
split analog bias.
"""

from __future__ import annotations

from .layer import AnalogLayer

from simulator import AnalogCore, CrossSimParameters
from torch import Tensor, cat, ones, kron, from_dlpack
from torch.nn import Linear, Parameter
from torch.autograd import Function


class AnalogLinear(Linear, AnalogLayer):
    """CrossSim implementation of torch.nn.Linear."""

    def __init__(
        self,
        params: CrossSimParameters,
        # Base Linear layer arguments
        in_features: int,
        out_features: int,
        bias: bool = True,
        # Additional arguments for AnalogLinear specifically
        bias_rows: int = 0,
    ) -> None:
        """ """
        # TODO: Do we need to handle device and dtype
        # Device based on GPU vs not, dtype
        super().__init__(in_features, out_features, bias)

        self.params = params.copy()
        self.bias_rows = bias_rows

        # Easier to track analog bias than digital bias
        # Avoids conditionals on bias and digital_bias
        self.analog_bias = bias and bias_rows > 0

        self.weight_mask = (slice(0, out_features, 1), slice(0, in_features, 1))
        self.bias_mask = (
            slice(None, None, 1),
            slice(in_features, in_features + bias_rows, 1),
        )

        self.core = AnalogCore(self.form_matrix().detach(), self.params)

    def form_matrix(self) -> Tensor:
        """ """
        if not self.analog_bias:
            return self.weight
        else:
            bias_expanded = (
                (self.bias / self.bias_rows)
                .reshape((self.out_features, 1))
                .repeat(1, self.bias_rows)
            )
            return cat((self.weight, bias_expanded), dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Linear forward operation.

        See AnalogLinearGrad.forward for details.
        """
        return AnalogLinearGrad.apply(
            x,
            self.weight,
            self.bias,
            self.core,
            self.bias_rows,
        )

    def get_core_weights(self) -> tuple[Tensor, Tensor | None]:
        """Returns weight and biases with programming errors."""
        matrix = self.get_matrix()
        weight = matrix[self.weight_mask]
        if self.analog_bias:
            # Summing along dim1 implicitly handles the reshape
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
        # Also means we can't just the matrix from the old core, need to call
        # form_matrix again.
        self.analog_bias = self.bias is not None and self.bias_rows > 0
        self.core = AnalogCore(self.form_matrix().detach(), self.params)

    @classmethod
    def from_torch(
        cls,
        layer: Linear,
        params: CrossSimParameters | list[CrossSimParameters],
        bias_rows: int = 0,
    ) -> AnalogLinear:
        """Build AnalogLinear from a Linear layer.

        Args:
            layer: torch.nn.Linear layer to be converted
            params:
                CrossSimParameters object or list of CrossSimParameters for
                the AnalogLinear layer. If a list, the length must match the
                number of arrays used within AnalogCore.
            bias_rows:
                Integer indicating the number of rows used for the bias.
                0 means bias is handled digitally.

        Returns:
            AnalogLinear layer with the same weights and properties as input
            Linear layer.
        """
        analog_layer = cls(
            params,
            layer.in_features,
            layer.out_features,
            layer.bias is not None,
            bias_rows,
        )
        analog_layer.weight = layer.weight
        analog_layer.bias = layer.bias

        return analog_layer

    @classmethod
    def to_torch(cls, layer: AnalogLinear, physical_weights: bool = False) -> Linear:
        """Creates a torch Linear layer from an AnalogLinear.

        Args:
            layer: AnalogLinear layer to be converted
            physical_weights:
                Bool indicating whether the torch layer should have ideal or
                weights with programming error applied.
        """
        torch_layer = Linear(
            layer.in_features,
            layer.out_features,
            layer.bias is not None,
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


class AnalogLinearGrad(Function):
    """Gradient implementation for CrossSim Linear layer.

    Forward direction is implemented using CrossSim. Backward is ideal.
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        core: AnalogCore,
        bias_rows: int,
    ) -> Tensor:
        """CrossSim-based Linear forward.

        Foward takes both the layer core object as well as the true weights
        and bias to save for the gradient computation. This uses a matmul
        representation for batched forward operations. All leading dimensions
        are collapsed into a single dimension so inputs to the core are
        always 1D or 2D.

        Args:
            ctx: torch gradient context
            x:
                N-D torch tensor of the layer input. Trailing dimension must
                match in_features.
            weight:
                2D torch tensor of the layer weight
                shape = (out_features, in_features)
            bias: 1D torch tensor of the layer bias of size out_features
            core: AnalogCore object for the operation
            bias_rows:
                Integer number of rows used for the bias (0 meaning digital
                bias)

        Returns:
            N-D torch tensor result. Trailing dimension is out_features.
        """
        ctx.save_for_backward(x, weight, bias)

        # Get x into a 1D or 2D shape
        if x.ndim < 3:
            # 1D and 2D inputs are just a simple matvec/matmat
            x_ = x
        else:
            # torch.nn.Linear can take any leading dimension, for simplicity
            # collapse the leading dimentions into one, compute as if 2D and
            # then reshape back
            x_ = x.flatten(0, x.ndim - 2)

        # Perform the operation using CrossSim
        if not (bias is not None and bias_rows):
            if x.ndim == 1:
                out = from_dlpack(core.dot(x_.detach()))
            else:
                out = from_dlpack(core.dot(x_.detach().T).T)
            if bias is not None:
                out += bias
        else:
            if x.ndim == 1:
                x_aug = cat((x_, ones(bias_rows)))
                out = from_dlpack(core.dot(x_aug.detach()))
            else:
                x_aug = cat((x_.T, ones((bias_rows, x_.shape[0]))))
                out = from_dlpack(core.dot(x_aug.detach()).T)

        # For inputs larget than 2D reshape back to the expected shape
        if x.ndim < 3:
            return out
        else:
            return out.view(*x.shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        """Ideal backward implementation of a Linear layer.

        Uses ideal (unquantized and unnoised) weights (and biases with analog
        bias) for the gradients. Derived from torch gradient tutorial.
        """
        (x, weight, bias) = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_output.matmul(weight)

        if ctx.needs_input_grad[1]:
            if x.ndim == 1:
                grad_weight = kron(grad_output, x).reshape(weight.shape)
            elif x.ndim == 2:
                grad_weight = grad_output.t().matmul(x)
            else:
                x_ = x_ = x.flatten(0, x.ndim - 2)
                grad_output_ = grad_output.flatten(0, grad_output.ndim - 2)
                grad_weight = grad_output_.t().matmul(x_)

        if bias is not None and ctx.needs_input_grad[2]:
            if x.ndim == 1:
                grad_bias = grad_output
            else:
                grad_bias = grad_output.sum(0)

        return grad_x, grad_weight, grad_bias, None, None
