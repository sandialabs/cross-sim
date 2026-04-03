#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""CrossSim version of Torch.nn.LSTMCell.

AnalogLSTMCell provides a CrossSim-based forward and backward using analog MVM.
"""

from __future__ import annotations

from .layer import AnalogLayer

from simulator import CrossSimParameters
from simulator.algorithms.dnn.analog_lstm_cell import AnalogLSTMCell as LSTMCellCore
from torch import Tensor, from_dlpack, zeros
from torch.nn import LSTMCell, Parameter
from torch.autograd import Function
import torch


class AnalogLSTMCell(LSTMCell, AnalogLayer):
    """CrossSim implementation of torch.nn.LSTMCell.

    See AnalogLayer for description of CrossSim-specific documentation.
    See torch.nn.LSTMCell for layer functionality documentation.
    """

    def __init__(
        self,
        params: CrossSimParameters | list[CrossSimParameters],
        # Base LSTMCell layer arguments
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
        # Additional arguments for AnalogLSTMCell specifically
        bias_ih_rows: int = 0,
        bias_hh_rows: int = 0,
    ) -> None:
        """Initializes AnalogLSTMCell and underlying torch.nn.LSTMCell layer.

        Args:
            params:
                CrossSimParameters object or list of CrossSimParameters (for
                layers requiring multiple arrays) for the AnalogRNNCell layer.
                If a list, the length must match the number of arrays used
                within AnalogCore.
            input_size: See torch.nn.LSTMCell input_size argument.
            hidden_size: See torch.nn.LSTMCell hidden_size argument.
            bias: See torch.nn.LSTMCell bias argument.
            device: See torch.nn.LSTMCell device argument.
            dtype: See torch.nn.LSTMCell dtype argument.
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
            device=device_,
            dtype=dtype,
        )

        if isinstance(params, CrossSimParameters):
            self.params = params.copy()
        elif isinstance(params, list):
            self.params = params[0].copy()

        self.bias_ih_rows = bias_ih_rows
        self.bias_hh_rows = bias_hh_rows

        self.core = LSTMCellCore(
            params,
            input_size,
            hidden_size,
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

    def forward(
        self, M_input: Tensor, hx: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        """LSTMCell forward operation.

        See AnalogLSTMCellGrad.forward for details.
        """
        # Handle input dimensions (Equivalent to torch.nn.LSTMCell)
        if M_input.dim() not in (1, 2):
            raise ValueError(
                f"LSTMCell: Expected input to be 1D or 2D, got {M_input.dim()}D instead"
            )
        if hx is not None:
            for idx, value in enumerate(hx):
                if value.dim() not in (1, 2):
                    raise ValueError(
                        f"LSTMCell: Expected hx[{idx}] to be 1D or 2D, got "
                        f"{value.dim()}D instead"
                    )

        is_batched = M_input.dim() == 2
        if not is_batched:
            M_input = M_input.unsqueeze(0)

        if hx is None:
            zeros_ = zeros(
                M_input.size(0),
                self.hidden_size,
                dtype=M_input.dtype,
                device=M_input.device,
            )
            hx = (zeros_, zeros_)
        else:
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx

        h_new, c_new = AnalogLSTMCellGrad.apply(
            M_input,
            hx[0],
            hx[1],
            self.weight_ih,
            self.weight_hh,
            self.bias_ih if self.bias else None,
            self.bias_hh if self.bias else None,
            self.core,
            self.bias_ih_rows,
            self.bias_hh_rows,
            self.hidden_size,
            self.bias,
            self.training,
        )

        if not is_batched:
            h_new = h_new.squeeze(0)
            c_new = c_new.squeeze(0)

        return h_new, c_new

    def reinitialize(self) -> None:
        """Rebuilds the layer's internal core object.

        Allows parameters to be updated within a layer without rebuilding the
        layer. This will resample all initialization-time errors
        (e.g. programming error)  even if the models were not be changed.
        Alternatively,  reinitialize can be used to directly resample
        initialization-time errors.
        """
        self._initialized = False

        self.core = LSTMCellCore(
            self.params,
            self.input_size,
            self.hidden_size,
            self.bias_ih_rows,
            self.bias_hh_rows,
        )
        self.core.set_matrix(self.form_matrix().detach())

    @classmethod
    def from_torch(
        cls,
        layer: LSTMCell,
        params: CrossSimParameters | list[CrossSimParameters],
        bias_ih_rows: int = 0,
        bias_hh_rows: int = 0,
    ) -> AnalogLSTMCell:
        """Build AnalogLSTMCell from an LSTMCell layer.

        Args:
            layer: torch.nn.LSTMCell layer to copy.
            params:
                CrossSimParameters object or list of CrossSimParameters.
            bias_ih_rows:
                Integer or list of integers indicating the number of rows to use
                to implement the bias_ih within the array. If a list the list
                lenggh must match num_layers, one per sublayer. 0 implies a
                digital bias. Ignored if bias is false.
            bias_hh_rows:
                Integer or list of integers indicating the number of rows to use
                to implement the bias_hh within the array. If a list the list
                lenggh must match num_layers, one per sublayer. 0 implies a
                digital bias. Ignored if bias is false.

        Returns:
            AnalogLSTMCell layer with the same weights and properties as input
            LSTMCell layer.
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
        layer: AnalogLSTMCell,
        physical_weights: bool = False,
        device=None,
        dtype=None,
    ) -> LSTMCell:
        """Creates a torch LSTMCell layer from an AnalogLSTMCell.

        Args:
            layer: AnalogLSTMCell layer to copy.
            physical_weights:
                Bool indicating whether the torch layer should have ideal
                weights or weights with programming error applied.
            device:
                The device where the layer will be placed.
            dtype:
                The dtype of the layer weights.

        Returns:
            torch.nn.LSTMCell with the same properties and weights as the
            AnalogLSTMCell layer.
        """
        if not device:
            if layer.params.simulation.useGPU:
                device = "cuda:{}".format(layer.params.simulation.gpu_id)
            else:
                device = "cpu"

        if not dtype:
            dtype = AnalogLayer._numpy_to_torch_dtype_dict[layer.core.dtype]

        torch_layer = LSTMCell(
            layer.input_size,
            layer.hidden_size,
            bias=layer.bias,
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


class AnalogLSTMCellGrad(Function):
    """Gradient implementation for CrossSim LSTMCell layer.

    Forward direction is implemented using CrossSim. Backward is ideal.
    """

    @staticmethod
    def forward(  # noqa:C901
        ctx,
        M_input: Tensor,
        h_prev: Tensor,
        c_prev: Tensor,
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Tensor | None,
        bias_hh: Tensor | None,
        core: LSTMCellCore,
        bias_ih_rows: int,
        bias_hh_rows: int,
        hidden_size: int,
        has_bias: bool,
        training: bool,
    ) -> tuple[Tensor, Tensor]:
        """CrossSim-based LSTMCell forward.

        Args:
            ctx: torch gradient context
            M_input: Input tensor of shape (batch, input_size)
            h_prev: Previous hidden state of shape (batch, hidden_size)
            c_prev: Previous cell state of shape (batch, hidden_size)
            weight_ih: Input-to-hidden weights
            weight_hh: Hidden-to-hidden weights
            bias_ih: Input-to-hidden bias (or None)
            bias_hh: Hidden-to-hidden bias (or None)
            core: LSTMCellCore object for the operation
            bias_ih_rows: Integer number of rows used for the bias_ih
            bias_hh_rows: Integer number of rows used for the bias_hh
            hidden_size: Size of hidden state
            has_bias: Whether the layer has bias
            training: Whether in training mode

        Returns:
            Tuple of (new_hidden_state, new_cell_state)
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

            ctx.save_for_backward(M_input, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh)
            ctx.hidden_size = hidden_size
            ctx.has_bias = has_bias

        # Perform analog operation
        gates = from_dlpack(
            core.apply(M_input.detach(), h_prev.detach(), c_prev.detach())
        )

        # Perform digital operations
        # Apply digital bias if needed
        if bias_ih is not None and not analog_bias_ih:
            gates += bias_ih
        if bias_hh is not None and not analog_bias_hh:
            gates += bias_hh

        # Apply activations to gates
        i_gate = torch.sigmoid(gates[..., :hidden_size])
        f_gate = torch.sigmoid(gates[..., hidden_size : 2 * hidden_size])
        g_gate = torch.tanh(gates[..., 2 * hidden_size : 3 * hidden_size])
        o_gate = torch.sigmoid(gates[..., 3 * hidden_size :])

        # LSTM computations
        c_new = f_gate * c_prev + i_gate * g_gate
        h_new = o_gate * torch.tanh(c_new)

        return h_new, c_new

    @staticmethod
    def backward(ctx, grad_h, grad_c):  # noqa:C901
        """Backward implementation of an LSTMCell layer using BPTT.

        This implements the backward pass for a single LSTM cell using the
        backpropagation through time algorithm. For a single timestep,
        this is equivalent to standard backpropagation.

        LSTM Cell equations:
        i_t = sigmoid(W_ii @ x_t + b_ii + W_hi @ h_{t-1} + b_hi)  # input gate
        f_t = sigmoid(W_if @ x_t + b_if + W_hf @ h_{t-1} + b_hf)  # forget gate
        g_t = tanh(W_ig @ x_t + b_ig + W_hg @ h_{t-1} + b_hg)     # cell gate
        o_t = sigmoid(W_io @ x_t + b_io + W_ho @ h_{t-1} + b_ho)  # output gate
        c_t = f_t * c_{t-1} + i_t * g_t                           # cell state
        h_t = o_t * tanh(c_t)                                     # hidden state

        Where:
        - x_t is the input at time t
        - h_t is the hidden state at time t
        - c_t is the cell state at time t
        - h_{t-1}, c_{t-1} are the previous hidden and cell states
        - i_t, f_t, g_t, o_t are the input, forget, cell, and output gates
        - W_ii, W_if, W_ig, W_io are input-to-hidden weight matrices for
            each gate
        - W_hi, W_hf, W_hg, W_ho are hidden-to-hidden weight matrices for
            each gate
        - b_ii, b_if, b_ig, b_io, b_hi, b_hf, b_hg, b_ho are bias vectors
        """
        # Retrieve saved tensors and context
        saved_tensors = ctx.saved_tensors
        M_input = saved_tensors[0]  # x_t
        h_prev = saved_tensors[1]  # h_{t-1}
        c_prev = saved_tensors[2]  # c_{t-1}

        # Extract individual weight tensors
        weight_ih = saved_tensors[3]  # W_ih (4*hidden_size, input_size)
        weight_hh = saved_tensors[4]  # W_hh (4*hidden_size, hidden_size)
        if ctx.has_bias:
            bias_ih = saved_tensors[5]  # b_ih (4*hidden_size)
            bias_hh = saved_tensors[6]  # b_hh (4*hidden_size)
        else:
            bias_ih = bias_hh = None

        hidden_size = ctx.hidden_size
        has_bias = ctx.has_bias

        # Initialize gradients
        grad_input = None
        grad_h_prev = None
        grad_c_prev = None
        grad_weight_ih = None
        grad_weight_hh = None
        grad_bias_ih = None
        grad_bias_hh = None

        # Extract individual weight matrices for each gate
        # Input-to-hidden weights for each gate
        W_ii = weight_ih[:hidden_size, :]  # input gate
        W_if = weight_ih[hidden_size : 2 * hidden_size, :]  # forget gate
        W_ig = weight_ih[2 * hidden_size : 3 * hidden_size, :]  # cell gate
        W_io = weight_ih[3 * hidden_size :, :]  # output gate

        # Hidden-to-hidden weights for each gate
        W_hi = weight_hh[:hidden_size, :]  # input gate
        W_hf = weight_hh[hidden_size : 2 * hidden_size, :]  # forget gate
        W_hg = weight_hh[2 * hidden_size : 3 * hidden_size, :]  # cell gate
        W_ho = weight_hh[3 * hidden_size :, :]  # output gate

        # Extract bias vectors if they exist
        if has_bias:
            b_ii = bias_ih[:hidden_size]  # input input bias
            b_if = bias_ih[hidden_size : 2 * hidden_size]  # forget input bias
            b_ig = bias_ih[2 * hidden_size : 3 * hidden_size]  # cell input bias
            b_io = bias_ih[3 * hidden_size :]  # output input bias

            b_hi = bias_hh[:hidden_size]  # input hidden bias
            b_hf = bias_hh[hidden_size : 2 * hidden_size]  # forget hidden bias
            b_hg = bias_hh[2 * hidden_size : 3 * hidden_size]  # cell hidden bias
            b_ho = bias_hh[3 * hidden_size :]  # output hidden bias
        else:
            b_ii = b_if = b_ig = b_io = b_hi = b_hf = b_hg = b_ho = 0

        # FORWARD PASS RECONSTRUCTION
        # Gate pre-activations
        input_ih = torch.mm(M_input, W_ii.t()) + b_ii
        input_hi = torch.mm(h_prev, W_hi.t()) + b_hi
        input_pre = input_ih + input_hi

        forget_ih = torch.mm(M_input, W_if.t()) + b_if
        forget_hi = torch.mm(h_prev, W_hf.t()) + b_hf
        forget_pre = forget_ih + forget_hi

        cell_ih = torch.mm(M_input, W_ig.t()) + b_ig
        cell_hi = torch.mm(h_prev, W_hg.t()) + b_hg
        cell_pre = cell_ih + cell_hi

        output_ih = torch.mm(M_input, W_io.t()) + b_io
        output_hi = torch.mm(h_prev, W_ho.t()) + b_ho
        output_pre = output_ih + output_hi

        # Gate activations
        i_t = torch.sigmoid(input_pre)
        f_t = torch.sigmoid(forget_pre)
        g_t = torch.tanh(cell_pre)
        o_t = torch.sigmoid(output_pre)

        # Cell and hidden state computation
        c_t = f_t * c_prev + i_t * g_t
        o_t * torch.tanh(c_t)

        # BACKWARD PASS
        # Total gradient w.r.t. cell state c_t
        tanh_c_t = torch.tanh(c_t)
        grad_c_t = grad_c + grad_h * o_t * (1 - tanh_c_t.pow(2))

        # Gradient w.r.t. output gate
        grad_o_t = grad_h * tanh_c_t

        # Gradient w.r.t. cell state equation
        grad_f_t = grad_c_t * c_prev
        grad_i_t = grad_c_t * g_t
        grad_g_t = grad_c_t * i_t
        grad_c_prev_direct = grad_c_t * f_t

        # Gradient w.r.t. gate pre-activations
        grad_input_pre = grad_i_t * i_t * (1 - i_t)
        grad_forget_pre = grad_f_t * f_t * (1 - f_t)
        grad_output_pre = grad_o_t * o_t * (1 - o_t)
        grad_cell_pre = grad_g_t * (1 - g_t.pow(2))

        # Gradient w.r.t. gate pre-activation components
        grad_input_ih = grad_input_pre
        grad_input_hi = grad_input_pre
        grad_forget_ih = grad_forget_pre
        grad_forget_hi = grad_forget_pre
        grad_cell_ih = grad_cell_pre
        grad_cell_hi = grad_cell_pre
        grad_output_ih = grad_output_pre
        grad_output_hi = grad_output_pre

        # GRADIENTS W.R.T. INPUTS
        if ctx.needs_input_grad[0]:  # grad_input (x_t)
            grad_input = (
                torch.mm(grad_input_ih, W_ii)
                + torch.mm(grad_forget_ih, W_if)
                + torch.mm(grad_cell_ih, W_ig)
                + torch.mm(grad_output_ih, W_io)
            )

        if ctx.needs_input_grad[1]:  # grad_h_prev (h_{t-1})
            grad_h_prev = (
                torch.mm(grad_input_hi, W_hi)
                + torch.mm(grad_forget_hi, W_hf)
                + torch.mm(grad_cell_hi, W_hg)
                + torch.mm(grad_output_hi, W_ho)
            )

        if ctx.needs_input_grad[2]:  # grad_c_prev (c_{t-1})
            grad_c_prev = grad_c_prev_direct

        # GRADIENTS W.R.T. WEIGHTS
        if ctx.needs_input_grad[3]:  # grad_weight_ih
            if M_input.dim() == 1:
                grad_W_ii = torch.outer(grad_input_ih.squeeze(0), M_input)
                grad_W_if = torch.outer(grad_forget_ih.squeeze(0), M_input)
                grad_W_ig = torch.outer(grad_cell_ih.squeeze(0), M_input)
                grad_W_io = torch.outer(grad_output_ih.squeeze(0), M_input)
            else:
                grad_W_ii = torch.mm(grad_input_ih.t(), M_input)
                grad_W_if = torch.mm(grad_forget_ih.t(), M_input)
                grad_W_ig = torch.mm(grad_cell_ih.t(), M_input)
                grad_W_io = torch.mm(grad_output_ih.t(), M_input)

            grad_weight_ih = torch.cat(
                [grad_W_ii, grad_W_if, grad_W_ig, grad_W_io], dim=0
            )

        if ctx.needs_input_grad[4]:  # grad_weight_hh
            if h_prev.dim() == 1:
                grad_W_hi = torch.outer(grad_input_hi.squeeze(0), h_prev)
                grad_W_hf = torch.outer(grad_forget_hi.squeeze(0), h_prev)
                grad_W_hg = torch.outer(grad_cell_hi.squeeze(0), h_prev)
                grad_W_ho = torch.outer(grad_output_hi.squeeze(0), h_prev)
            else:
                grad_W_hi = torch.mm(grad_input_hi.t(), h_prev)
                grad_W_hf = torch.mm(grad_forget_hi.t(), h_prev)
                grad_W_hg = torch.mm(grad_cell_hi.t(), h_prev)
                grad_W_ho = torch.mm(grad_output_hi.t(), h_prev)

            grad_weight_hh = torch.cat(
                [grad_W_hi, grad_W_hf, grad_W_hg, grad_W_ho], dim=0
            )

        # GRADIENTS W.R.T. BIASES
        if has_bias:
            if ctx.needs_input_grad[5]:  # grad_bias_ih
                grad_b_ii = grad_input_ih.sum(dim=0)
                grad_b_if = grad_forget_ih.sum(dim=0)
                grad_b_ig = grad_cell_ih.sum(dim=0)
                grad_b_io = grad_output_ih.sum(dim=0)
                grad_bias_ih = torch.cat(
                    [grad_b_ii, grad_b_if, grad_b_ig, grad_b_io], dim=0
                )

            if ctx.needs_input_grad[6]:  # grad_bias_hh
                grad_b_hi = grad_input_hi.sum(dim=0)
                grad_b_hf = grad_forget_hi.sum(dim=0)
                grad_b_hg = grad_cell_hi.sum(dim=0)
                grad_b_ho = grad_output_hi.sum(dim=0)
                grad_bias_hh = torch.cat(
                    [grad_b_hi, grad_b_hf, grad_b_hg, grad_b_ho], dim=0
                )

        # Handle single sample case
        if M_input.dim() == 1:
            if grad_input is not None:
                grad_input = grad_input.squeeze(0)
            if grad_h_prev is not None:
                grad_h_prev = grad_h_prev.squeeze(0)
            if grad_c_prev is not None:
                grad_c_prev = grad_c_prev.squeeze(0)

        # Return gradients in the same order as forward inputs
        return (
            grad_input,  # gradient w.r.t input
            grad_h_prev,  # gradient w.r.t previous hidden state
            grad_c_prev,  # gradient w.r.t previous cell state
            grad_weight_ih,  # gradient w.r.t weight_ih
            grad_weight_hh,  # gradient w.r.t weight_hh
            grad_bias_ih,  # gradient w.r.t bias_ih
            grad_bias_hh,  # gradient w.r.t bias_hh
            None,  # core (no gradient needed)
            None,  # bias_rows_ih (no gradient needed)
            None,  # bias_rows_hh (no gradient needed)
            None,  # hidden_size (no gradient needed)
            None,  # has_bias (no gradient needed)
            None,  # training (no gradient needed)
        )
