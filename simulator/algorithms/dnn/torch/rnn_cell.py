#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""CrossSim version of Torch.nn.RNNCell.

AnalogRNNCell provides a CrossSim-based forward and backward using analog MVM.
"""

from __future__ import annotations

from .layer import AnalogLayer

from simulator import CrossSimParameters
from simulator.algorithms.dnn.analog_rnn_cell import AnalogRNNCell as RNNCellCore
from torch import Tensor, from_dlpack, zeros
from torch.nn import RNNCell, Parameter
from torch.autograd import Function
import torch


class AnalogRNNCell(RNNCell, AnalogLayer):
    """CrossSim implementation of torch.nn.RNNCell.

    See AnalogLayer for description of CrossSim-specific documentation.
    See torch.nn.RNNCell for layer functionality documentation.
    """

    def __init__(
        self,
        params: CrossSimParameters | list[CrossSimParameters],
        # Base RNNCell layer arguments
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device=None,
        dtype=None,
        # Additional arguments for AnalogRNNCell specifically
        bias_ih_rows: int = 0,
        bias_hh_rows: int = 0,
    ) -> None:
        """Initializes AnalogRNNCell and underlying torch.nn.RNNCell layer.

        Args:
            params:
                CrossSimParameters object or list of CrossSimParameters (for
                layers requiring multiple arrays) for the AnalogRNNCell layer.
                If a list, the length must match the number of arrays used
                within AnalogCore.
            input_size: See torch.nn.RNNCell input_size argument.
            hidden_size: See torch.nn.RNNCell hidden_size argument.
            bias: See torch.nn.RNNCell bias argument.
            nonlinearity: See torch.nn.RNNCell nonlinearity argument.
            device: See torch.nn.RNNCell device argument.
            dtype: See torch.nn.RNNCell dtype argument.
            bias_ih_rows:
                Integer indicating the number of rows to use to implement the
                bias_ih within the array. 0 implies a digital bias. Ignored if
                bias is false.
            bias_hh_rows:
                Integer indicating the number of rows to use to implement the
                bias_hh within the array. 0 implies a digital bias. Ignored if
                bias is false.
        """
        device_ = AnalogLayer._set_device(
            device,
            params[0] if isinstance(params, list) else params,
        )
        self._array_weight_variables = ["weight_ih", "weight_hh"]
        self._array_bias_variables = ["bias_ih", "bias_hh"]

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            nonlinearity=nonlinearity,
            device=device_,
            dtype=dtype,
        )

        if isinstance(params, CrossSimParameters):
            self.params = params.copy()
        elif isinstance(params, list):
            self.params = params[0].copy()

        self.bias_ih_rows = bias_ih_rows
        self.bias_hh_rows = bias_hh_rows

        self.core = RNNCellCore(
            params,
            input_size,
            hidden_size,
            nonlinearity,
            bias_ih_rows,
            bias_hh_rows,
        )
        self.core.set_matrix(self.form_matrix().detach())

    def form_matrix(self) -> Tensor:
        """Builds 2D weight matrix for programming into the array.

        Returns:
            2D Torch Tensor of the matrix.

        """
        weight_ih_ = self.weight_ih.detach().cpu()
        weight_hh_ = self.weight_hh.detach().cpu()

        if self.analog_bias_ih:
            bias_ih_ = self.bias_ih.detach().cpu()
        else:
            bias_ih_ = None

        if self.analog_bias_hh:
            bias_hh_ = self.bias_hh.detach().cpu()
        else:
            bias_hh_ = None

        return from_dlpack(
            self.core.form_matrix(weight_ih_, weight_hh_, bias_ih_, bias_hh_)
        )

    def forward(self, M_input: Tensor, hx: Tensor | None = None) -> Tensor:
        """RNNCell forward operation.

        See AnalogRNNCellGrad.forward for details.
        """
        # Handle input dimensions (Equivalent to torch.nn.RNNCell)
        if M_input.dim() not in (1, 2):
            raise ValueError(
                f"RNNCell: Expected input to be 1D or 2D, got {M_input.dim()}D instead"
            )
        if hx is not None and hx.dim() not in (1, 2):
            raise ValueError(
                f"RNNCell: Expected hidden to be 1D or 2D, got {hx.dim()}D instead"
            )

        is_batched = M_input.dim() == 2
        if not is_batched:
            M_input = M_input.unsqueeze(0)

        if hx is None:
            hx = zeros(
                M_input.size(0),
                self.hidden_size,
                dtype=M_input.dtype,
                device=M_input.device,
            )
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        output = AnalogRNNCellGrad.apply(
            M_input,
            hx,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih if self.bias else None,
            self.bias_hh if self.bias else None,
            self.core,
            self.bias_ih_rows,
            self.bias_hh_rows,
            self.hidden_size,
            self.nonlinearity,
            self.bias,
            self.training,
        )

        if not is_batched:
            output = output.squeeze(0)

        return output

    def reinitialize(self) -> None:
        """Rebuilds the layer's internal core object.

        Allows parameters to be updated within a layer without rebuilding the
        layer. This will resample all initialization-time errors
        (e.g. programming error)  even if the models were not be changed.
        Alternatively,  reinitialize can be used to directly resample
        initialization-time errors.
        """
        self._initialized = False

        self.core = RNNCellCore(
            self.params,
            self.input_size,
            self.hidden_size,
            self.nonlinearity,
            self.bias_ih_rows,
            self.bias_hh_rows,
        )
        self.core.set_matrix(self.form_matrix().detach())

    @classmethod
    def from_torch(
        cls,
        layer: RNNCell,
        params: CrossSimParameters | list[CrossSimParameters],
        bias_ih_rows: int = 0,
        bias_hh_rows: int = 0,
    ) -> AnalogRNNCell:
        """Build AnalogRNNCell from an RNNCell layer.

        Args:
            layer: torch.nn.RNNCell layer to copy.
            params:
                CrossSimParameters object or list of CrossSimParameters.
            bias_ih_rows:
                Integer indicating the number of rows to use to implement the
                bias_ih within the array. 0 implies a digital bias. Ignored if
                bias is false.
            bias_hh_rows:
                Integer indicating the number of rows to use to implement the
                bias_hh within the array. 0 implies a digital bias. Ignored if
                bias is false.

        Returns:
            AnalogRNNCell layer with the same weights and properties as input
            RNNCell layer.
        """
        device = AnalogLayer._set_device(
            layer.weight_ih.device,
            params[0] if isinstance(params, list) else params,
        )

        analog_layer = cls(
            params,
            layer.input_size,
            layer.hidden_size,
            layer.bias,
            layer.nonlinearity,
            device,
            layer.weight_ih.dtype,
            bias_ih_rows,
            bias_hh_rows,
        )

        # Copy weights and biases
        analog_layer.weight_ih = layer.weight_ih
        analog_layer.weight_hh = layer.weight_hh
        analog_layer.bias_ih = layer.bias_ih
        analog_layer.bias_hh = layer.bias_hh

        return analog_layer

    @classmethod
    def to_torch(  # noqa:C901
        cls,
        layer: AnalogRNNCell,
        physical_weights: bool = False,
        device=None,
        dtype=None,
    ) -> RNNCell:
        """Creates a torch RNNCell layer from an AnalogRNNCell.

        Args:
            layer: AnalogRNNCell layer to copy.
            physical_weights:
                Bool indicating whether the torch layer should have ideal
                weights or weights with programming error applied.
            device:
                The device where the layer will be placed.
            dtype:
                The dtype of the layer weights.

        Returns:
            torch.nn.RNNCell with the same properties and weights as the
            AnalogRNNCell layer.
        """
        if not device:
            if layer.params.simulation.useGPU:
                device = "cuda:{}".format(layer.params.simulation.gpu_id)
            else:
                device = "cpu"

        if not dtype:
            dtype = AnalogLayer._numpy_to_torch_dtype_dict[layer.core.dtype]

        torch_layer = RNNCell(
            layer.input_size,
            layer.hidden_size,
            bias=layer.bias,
            nonlinearity=layer.nonlinearity,
            device=device,
            dtype=dtype,
        )

        if physical_weights:
            # Get weights with errors from core
            w_ih, w_hh, b_ih, b_hh = layer.core.get_core_weights()
            w_ih = from_dlpack(w_ih)
            w_hh = from_dlpack(w_hh)
            if layer.bias is not None:
                if layer.analog_bias_ih:
                    b_ih = from_dlpack(b_ih) if b_ih is not None else None
                else:
                    b_ih = layer.bias_ih.clone().detach()

                if layer.analog_bias_hh:
                    b_hh = from_dlpack(b_hh) if b_hh is not None else None
                else:
                    b_hh = layer.bias_hh.clone().detach()
        else:
            # Get ideal weights
            w_ih = layer.weight_ih.clone().detach()
            w_hh = layer.weight_hh.clone().detach()
            if layer.bias:
                b_ih = layer.bias_ih.clone().detach()
                b_hh = layer.bias_hh.clone().detach()
            else:
                b_ih = None
                b_hh = None

        # Set weights
        torch_layer.weight_ih = Parameter(w_ih)
        torch_layer.weight_hh = Parameter(w_hh)

        # Set biases if they exist
        if b_ih is not None:
            torch_layer.bias_ih = Parameter(b_ih)
        if b_hh is not None:
            torch_layer.bias_hh = Parameter(b_hh)

        return torch_layer

    @property
    def analog_bias_ih(self):
        """Bool inidicating if the layer uses an analog value for bias_ih."""
        return self.bias and self.bias_ih_rows > 0

    @property
    def analog_bias_hh(self):
        """Bool inidicating if the layer uses an analog value for bias_hh."""
        return self.bias and self.bias_hh_rows > 0


class AnalogRNNCellGrad(Function):
    """Gradient implementation for CrossSim RNNCell layer.

    Forward direction is implemented using CrossSim. Backward is ideal.
    """

    @staticmethod
    def forward(
        ctx,
        M_input: Tensor,
        hx: Tensor,
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Tensor | None,
        bias_hh: Tensor | None,
        core: RNNCellCore,
        bias_ih_rows: int,
        bias_hh_rows: int,
        hidden_size: int,
        nonlinearity: str,
        has_bias: bool,
        training: bool,
    ) -> Tensor:
        """CrossSim-based RNNCell forward.

        Args:
            ctx: torch gradient context
            M_input: Input tensor of shape (batch, input_size)
            hx: Hidden state of shape (batch, hidden_size)
            weight_ih: Input-to-hidden weights
            weight_hh: Hidden-to-hidden weights
            bias_ih: Input-to-hidden bias (or None)
            bias_hh: Hidden-to-hidden bias (or None)
            core: RNNCellCore object for the operation
            bias_ih_rows: Integer number of rows used for the bias_ih
            bias_hh_rows: Integer number of rows used for the bias_hh
            hidden_size: Size of hidden state
            nonlinearity: 'tanh' or 'relu'
            has_bias: Whether the layer has bias
            training: Whether in training mode

        Returns:
            New hidden state of shape (batch, hidden_size)
        """
        analog_bias_ih = bias_ih is not None and bool(bias_ih_rows)
        analog_bias_hh = bias_hh is not None and bool(bias_hh_rows)

        # Save tensors for backward pass
        if training:
            w_ih, w_hh, b_ih, b_hh = core.get_core_weights()
            w_ih = from_dlpack(w_ih)
            w_hh = from_dlpack(w_hh)
            if analog_bias_ih:
                b_ih = from_dlpack(b_ih)
            else:
                b_ih = bias_ih

            if analog_bias_hh:
                b_hh = from_dlpack(b_hh)
            else:
                b_hh = bias_hh

            ctx.save_for_backward(M_input, hx, w_ih, w_hh, b_ih, b_hh)
            ctx.hidden_size = hidden_size
            ctx.nonlinearity = nonlinearity
            ctx.has_bias = has_bias

        # Perform analog operation
        pre_activation = from_dlpack(core.apply(M_input.detach(), hx.detach()))

        # Apply digital bias if needed
        if bias_ih is not None and not analog_bias_ih:
            pre_activation += bias_ih
        if bias_hh is not None and not analog_bias_hh:
            pre_activation += bias_hh

        # Apply activation function
        if nonlinearity == "tanh":
            output = torch.tanh(pre_activation)
        elif nonlinearity == "relu":
            output = torch.relu(pre_activation)
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

        return output

    @staticmethod
    def backward(ctx, grad_output):  # noqa:C901
        """Backward implementation of an RNNCell layer using BPTT.

        This implements the backward pass for a single RNN cell using the
        backpropagation through time algorithm. For a single timestep,
        this is equivalent to standard backpropagation.

        RNN Cell equations:
        h_t = activation(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)

        Where:
        - x_t is the input at time t
        - h_t is the hidden state at time t
        - h_{t-1} is the previous hidden state
        - W_ih, W_hh are weight matrices
        - b_ih, b_hh are bias vectors
        - activation is tanh or relu
        """
        # Retrieve saved tensors and context
        saved_tensors = ctx.saved_tensors
        M_input = saved_tensors[0]  # x_t
        hx = saved_tensors[1]  # h_{t-1}

        # Extract individual weight tensors
        weight_ih = saved_tensors[2]  # W_ih (hidden_size, input_size)
        weight_hh = saved_tensors[3]  # W_hh (hidden_size, hidden_size)
        if ctx.has_bias:
            bias_ih = saved_tensors[4]  # b_ih (hidden_size)
            bias_hh = saved_tensors[5]  # b_hh (hidden_size)
        else:
            bias_ih = bias_hh = None

        nonlinearity = ctx.nonlinearity
        has_bias = ctx.has_bias

        # Initialize gradients
        grad_input = None
        grad_hx = None
        grad_weight_ih = None
        grad_weight_hh = None
        grad_bias_ih = None
        grad_bias_hh = None

        # FORWARD PASS RECONSTRUCTION
        # Pre-activation
        pre_activation = torch.mm(M_input, weight_ih.t()) + torch.mm(hx, weight_hh.t())

        if has_bias:
            pre_activation = pre_activation + bias_ih + bias_hh

        # Apply activation function and compute gradients
        if nonlinearity == "tanh":
            # For tanh: d_tanh/dx = 1 - tanh^2(x)
            # Since output = tanh(pre_activation), we can compute:
            output_reconstructed = torch.tanh(pre_activation)
            grad_pre_activation = grad_output * (1 - output_reconstructed.pow(2))
        elif nonlinearity == "relu":
            # For ReLU: d_relu/dx = 1 if x > 0, else 0
            grad_pre_activation = grad_output * (pre_activation > 0).float()
        else:
            raise ValueError(f"Unsupported activation nonlinearity: {nonlinearity}")

        # GRADIENTS W.R.T. INPUTS
        if ctx.needs_input_grad[0]:  # grad_input (x_t)
            grad_input = torch.mm(grad_pre_activation, weight_ih)

        if ctx.needs_input_grad[1]:  # grad_hx (h_{t-1})
            grad_hx = torch.mm(grad_pre_activation, weight_hh)

        # GRADIENTS W.R.T. WEIGHTS
        if ctx.needs_input_grad[2]:  # grad_weight_ih
            if M_input.dim() == 1:
                grad_weight_ih = torch.outer(grad_pre_activation.squeeze(0), M_input)
            else:
                grad_weight_ih = torch.mm(grad_pre_activation.t(), M_input)

        if ctx.needs_input_grad[3]:  # grad_weight_hh
            if hx.dim() == 1:
                grad_weight_hh = torch.outer(grad_pre_activation.squeeze(0), hx)
            else:
                grad_weight_hh = torch.mm(grad_pre_activation.t(), hx)

        # GRADIENTS W.R.T. BIASES
        if has_bias:
            if ctx.needs_input_grad[4]:  # grad_bias_ih
                grad_bias_ih = grad_pre_activation.sum(dim=0)

            if ctx.needs_input_grad[5]:  # grad_bias_hh
                grad_bias_hh = grad_pre_activation.sum(dim=0)

        # Handle single sample case
        if M_input.dim() == 1:
            if grad_input is not None:
                grad_input = grad_input.squeeze(0)
            if grad_hx is not None:
                grad_hx = grad_hx.squeeze(0)

        # Return gradients in the same order as forward inputs
        return (
            grad_input,  # gradient w.r.t input
            grad_hx,  # gradient w.r.t hidden state
            grad_weight_ih,  # gradient w.r.t weight_ih
            grad_weight_hh,  # gradient w.r.t weight_hh
            grad_bias_ih,  # gradient w.r.t bias_ih
            grad_bias_hh,  # gradient w.r.t bias_hh
            None,  # core (no gradient needed)
            None,  # bias_rows_ih (no gradient needed)
            None,  # bias_rows_hh (no gradient needed)
            None,  # hidden_size (no gradient needed)
            None,  # nonlinearity (no gradient needed)
            None,  # has_bias (no gradient needed)
            None,  # training (no gradient needed)
        )
