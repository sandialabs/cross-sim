#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""CrossSim version of Torch.nn.Linear.

AnalogLinear provides a CrossSim-based forward using analog MVM. Backward is
implemented using an ideal torch backward (unquantized).
"""

from __future__ import annotations

from .layer import AnalogLayer

from simulator import CrossSimParameters
from simulator.algorithms.dnn.analog_linear import AnalogLinear as LinearCore
from torch import Tensor, kron, from_dlpack
from torch.nn import Linear, Parameter
from torch.autograd import Function


class AnalogLinear(Linear, AnalogLayer):
    """CrossSim implementation of torch.nn.Linear.

    See AnalogLayer for description of CrossSim-specific documentation.
    See torch.nn.Linear for layer functionality documentation.
    """

    def __init__(
        self,
        params: CrossSimParameters | list[CrossSimParameters],
        # Base Linear layer arguments
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        # Additional arguments for AnalogLinear specifically
        bias_rows: int = 0,
    ) -> None:
        """Initializes AnalogLinear and underlying torch.nn.Linear layer.

        Args:
            params:
                CrossSimParameters object or list of CrossSimParameters (for layers
                requiring multiple arrays) for the AnalogLinear layer. If a list, the
                length must match the number of arrays used within AnalogCore.
            in_features: See torch.nn.Linear in_features argument.
            out_features: See torch.nn.Linear out_features argument.
            bias: See torch.nn.Linear bias argument.
            device: See torch.nn.Linear device argument.
            dtype: See torch.nn.Linear dtype argument.
            bias_rows:
                Integer indicating the number of rows to use to implement the bias
                within the array. 0 implies a digital bias. Ignored if bias is false.
        """
        device_ = AnalogLayer._set_device(
            device,
            params[0] if isinstance(params, list) else params,
        )

        super().__init__(in_features, out_features, bias, device_, dtype)

        if isinstance(params, CrossSimParameters):
            self.params = params.copy()
        elif isinstance(params, list):
            self.params = params[0].copy()

        self.bias_rows = bias_rows

        # Easier to track analog bias than digital bias
        # Avoids conditionals on bias and digital_bias
        self.analog_bias = bias and bias_rows > 0

        self.weight_mask = (slice(0, out_features, 1), slice(0, in_features, 1))
        self.bias_mask = (
            slice(None, None, 1),
            slice(in_features, in_features + bias_rows, 1),
        )

        self.core = LinearCore(params, in_features, out_features, bias_rows)
        self.core.set_matrix(self.form_matrix().detach())

    def form_matrix(self) -> Tensor:
        """Builds 2D weight matrix for programming into the array.

        Returns:
            2D Torch Tensor of the matrix.

        """
        weight_ = self.weight.detach().cpu()
        if self.analog_bias:
            return from_dlpack(self.core.form_matrix(weight_, self.bias.detach().cpu()))
        else:
            return from_dlpack(self.core.form_matrix(weight_))

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
        """Gets the weight and bias tensors with errors applied.

        Returns:
            Tuple of Torch Tensors, 2D for weights, 1D or None for bias
        """
        # TODO: Slightly hackish, needed for test passage, remove once we agree on
        # return device for get_matrix
        matrix = self.get_matrix()
        weight = matrix[self.weight_mask].to(self.weight.device)
        if self.analog_bias:
            # Summing along dim1 implicitly handles the reshape
            bias = matrix[self.bias_mask].sum(1).to(self.bias.device)
        else:
            bias = self.bias

        return (weight, bias)

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
        # Also means we can't just the matrix from the old core, need to call
        # form_matrix again.
        self.analog_bias = self.bias is not None and self.bias_rows > 0
        self.core = LinearCore(
            self.params,
            self.in_features,
            self.out_features,
            self.bias_rows,
        )
        self.core.set_matrix(self.form_matrix().detach())

    @classmethod
    def from_torch(
        cls,
        layer: Linear,
        params: CrossSimParameters | list[CrossSimParameters],
        bias_rows: int = 0,
    ) -> AnalogLinear:
        """Build AnalogLinear from a Linear layer.

        Creates a new AnalogLinear layer with the same attributes as the original torch
        layer.

        Args:
            layer: torch.nn.Linear layer to copy.
            params:
                CrossSimParameters object or list of CrossSimParameters (for layers
                requiring multiple arrays) for the AnalogLinear layer. If a list, the
                length must match the number of arrays used within AnalogCore.
            bias_rows:
                Integer indicating the number of analog rows to use for the bias.
                0 indicates a digital bias. Ignored if layer does not have a bias.

        Returns:
            AnalogLinear layer with the same weights and properties as input
            Linear layer.

        Raises:
            ValueError: Device mismatch between CrossSimParameters and layer
        """
        device = AnalogLayer._set_device(
            layer.weight.device,
            params[0] if isinstance(params, list) else params,
        )

        analog_layer = cls(
            params,
            layer.in_features,
            layer.out_features,
            layer.bias is not None,
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
        layer: AnalogLinear,
        physical_weights: bool = False,
        device=None,
        dtype=None,
    ) -> Linear:
        """Creates a torch Linear layer from an AnalogLinear.

        Args:
            layer: AnalogLinear layer to copy.
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
            errors) as the AnalogLinear layer.
        """
        if not device:
            if layer.params.simulation.useGPU:
                device = "cuda:{}".format(layer.params.simulation.gpu_id)
            else:
                device = "cpu"

        if not dtype:
            dtype = AnalogLayer._numpy_to_torch_dtype_dict[layer.core.dtype]

        torch_layer = Linear(
            layer.in_features,
            layer.out_features,
            layer.bias is not None,
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
        core: LinearCore,
        bias_rows: int,
    ) -> Tensor:
        """CrossSim-based Linear forward.

        Forward takes both the layer core object as well as the true weights
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

        analog_bias = bias is not None and bool(bias_rows)

        # Get x into a 1D or 2D shape
        if x.ndim < 3:
            # 1D and 2D inputs are just a simple matvec/matmat
            x_ = x
        else:
            # torch.nn.Linear can take any leading dimension, for simplicity
            # collapse the leading dimensions into one, compute as if 2D and
            # then reshape back
            x_ = x.flatten(0, x.ndim - 2)

        # Perform the operation using CrossSim
        out = from_dlpack(core.apply(x_.detach()))

        if bias is not None and not analog_bias:
            out += bias

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
