#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""CrossSim version of Torch.nn.GRUCell.

AnalogGRUCell provides a CrossSim-based forward and backward using analog MVM.
"""

from __future__ import annotations

from .layer import AnalogLayer

from simulator import CrossSimParameters
from simulator.algorithms.dnn.analog_gru_cell import AnalogGRUCell as GRUCellCore
from torch import Tensor, from_dlpack, zeros
from torch.nn import GRUCell, Parameter
from torch.autograd import Function
import torch


class AnalogGRUCell(GRUCell, AnalogLayer):
    """CrossSim implementation of torch.nn.GRUCell.

    See AnalogLayer for description of CrossSim-specific documentation.
    See torch.nn.GRUCell for layer functionality documentation.
    """

    def __init__(
        self,
        params: CrossSimParameters | list[CrossSimParameters],
        # Base GRUCell layer arguments
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
        # Additional arguments for AnalogGRUCell specifically
        bias_ih_rows: int = 0,
        bias_hh_rows: int = 0,
    ) -> None:
        """Initializes AnalogGRUCell and underlying torch.nn.GRUCell layer.

        Args:
            params:
                CrossSimParameters object or list of CrossSimParameters (for
                layers requiring multiple arrays) for the AnalogRNNCell layer.
                If a list, the length must match the number of arrays used
                within AnalogCore.
            input_size: See torch.nn.GRUCell input_size argument.
            hidden_size: See torch.nn.GRUCell hidden_size argument.
            bias: See torch.nn.GRUCell bias argument.
            device: See torch.nn.GRUCell device argument.
            dtype: See torch.nn.GRUCell dtype argument.
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

        self.core = GRUCellCore(
            params,
            input_size,
            hidden_size,
            bias_ih_rows,
            bias_hh_rows,
        )

        # Program weight matrices
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
        """GRUCell forward operation.

        See AnalogGRUCellGrad.forward for details.
        """
        # Handle input dimensions (Equivalent to torch.nn.GRUCell)
        if M_input.dim() not in (1, 2):
            raise ValueError(
                f"GRUCell: Expected input to be 1D or 2D, got {M_input.dim()}D instead"
            )
        if hx is not None and hx.dim() not in (1, 2):
            raise ValueError(
                f"GRUCell: Expected hidden to be 1D or 2D, got {hx.dim()}D instead"
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

        output = AnalogGRUCellGrad.apply(
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

        self.core = GRUCellCore(
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
        layer: GRUCell,
        params: CrossSimParameters | list[CrossSimParameters],
        bias_ih_rows: int = 0,
        bias_hh_rows: int = 0,
    ) -> AnalogGRUCell:
        """Build AnalogGRUCell from a GRUCell layer.

        Args:
            layer: torch.nn.GRUCell layer to copy.
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
            AnalogGRUCell layer with the same weights and properties as input
            GRUCell layer.
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
        layer: AnalogGRUCell,
        physical_weights: bool = False,
        device=None,
        dtype=None,
    ) -> GRUCell:
        """Creates a torch GRUCell layer from an AnalogGRUCell.

        Args:
            layer: AnalogGRUCell layer to copy.
            physical_weights:
                Bool indicating whether the torch layer should have ideal
                weights or weights with programming error applied.
            device:
                The device where the layer will be placed.
            dtype:
                The dtype of the layer weights.

        Returns:
            torch.nn.GRUCell with the same properties and weights as the
            AnalogGRUCell layer.
        """
        if not device:
            if layer.params.simulation.useGPU:
                device = "cuda:{}".format(layer.params.simulation.gpu_id)
            else:
                device = "cpu"

        if not dtype:
            dtype = AnalogLayer._numpy_to_torch_dtype_dict[layer.core.dtype]

        torch_layer = GRUCell(
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
        """Bool inidicating if the layer uses an analog value for bias_hh."""
        return self.bias and self.bias_ih_rows > 0

    @property
    def analog_bias_hh(self):
        """Bool inidicating if the layer uses an analog value for bias_hh."""
        return self.bias and self.bias_hh_rows > 0


class AnalogGRUCellGrad(Function):
    """Gradient implementation for CrossSim GRUCell layer.

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
        core: GRUCellCore,
        bias_ih_rows: int,
        bias_hh_rows: int,
        hidden_size: int,
        has_bias: bool,
        training: bool,
    ) -> Tensor:
        """CrossSim-based GRUCell forward.

        Args:
            ctx: torch gradient context
            M_input: Input tensor of shape (batch, input_size)
            hx: Hidden state of shape (batch, hidden_size)
            weight_ih: Input-to-hidden weights
            weight_hh: Hidden-to-hidden weights
            bias_ih: Input-to-hidden bias (or None)
            bias_hh: Hidden-to-hidden bias (or None)
            core: GRUCellCore object for the operation
            bias_ih_rows: Integer number of rows used for the bias_ih
            bias_hh_rows: Integer number of rows used for the bias_hh
            hidden_size: Size of hidden state
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
            ctx.has_bias = has_bias

        # Perform analog operation
        gates = from_dlpack(core.apply(M_input.detach(), hx.detach()))

        # Apply digital bias if needed
        if bias_ih is not None and not analog_bias_ih:
            b_ir = bias_ih[:hidden_size]  # reset input bias
            b_iz = bias_ih[hidden_size : 2 * hidden_size]  # update input bias
            b_in = bias_ih[2 * hidden_size :]  # new input bias

            gates[:, :hidden_size] += b_ir  # reset gate
            gates[:, hidden_size : 2 * hidden_size] += b_iz  # update gate
            gates[:, 2 * hidden_size : 3 * hidden_size] += b_in  # new input component

        if bias_hh is not None and not analog_bias_hh:
            b_hr = bias_hh[:hidden_size]  # reset hidden bias
            b_hz = bias_hh[hidden_size : 2 * hidden_size]  # update hidden bias
            b_hn = bias_hh[2 * hidden_size :]  # new hidden bias

            gates[:, :hidden_size] += b_hr  # reset gate
            gates[:, hidden_size : 2 * hidden_size] += b_hz  # update gate
            gates[:, 3 * hidden_size : 4 * hidden_size] += b_hn

        reset_linear = gates[:, :hidden_size]  # W_ir*xt + W_hr*ht-1
        update_linear = gates[:, hidden_size : 2 * hidden_size]  # W_iz*xt + W_hz*ht-1
        new_input = gates[:, 2 * hidden_size : 3 * hidden_size]  # W_in*xt
        new_hidden = gates[:, 3 * hidden_size : 4 * hidden_size]  # W_hn*ht-1

        # Extract individual gates
        r_gate = torch.sigmoid(reset_linear)
        z_gate = torch.sigmoid(update_linear)

        new_combined = new_input + r_gate * new_hidden

        n_gate = torch.tanh(new_combined)

        # GRU computations
        h_new = (1 - z_gate) * n_gate + z_gate * hx

        if M_input.ndim == 1:  # no_batch
            h_new = h_new.reshape(hidden_size)

        return h_new

    @staticmethod
    def backward(ctx, grad_output):  # noqa:C901
        """Backward implementation of a GRUCell layer using BPTT.

        This implements the backward pass for a single GRU cell using the
        backpropagation through time algorithm. For a single timestep,
        this is equivalent to standard backpropagation.

        GRU Cell equations:
        r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)  # reset gate
        z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)  # update gate
        n_t = tanh(W_in @ x_t + b_in + W_hn @ h_{t-1} + b_hn)    # new gate
        h_t = (1 - z_t) * n_t + z_t * h_{t-1}                    # output

        Where:
        - x_t is the input at time t
        - h_t is the hidden state at time t
        - h_{t-1} is the previous hidden state
        - r_t, z_t, n_t are the reset, update, and new gates respectively
        - W_ir, W_iz, W_in are input-to-hidden weight matrices for each gate
        - W_hr, W_hz, W_hn are hidden-to-hidden weight matrices for each gate
        - b_ir, b_iz, b_in, b_hr, b_hz, b_hn are bias vectors
        """
        # Retrieve saved tensors and context
        saved_tensors = ctx.saved_tensors
        M_input = saved_tensors[0]  # x_t
        hx = saved_tensors[1]  # h_{t-1}

        # Extract individual weight tensors
        weight_ih = saved_tensors[2]  # W_ih (3*hidden_size, input_size)
        weight_hh = saved_tensors[3]  # W_hh (3*hidden_size, hidden_size)
        if ctx.has_bias:
            bias_ih = saved_tensors[4]  # b_ih (3*hidden_size)
            bias_hh = saved_tensors[5]  # b_hh (3*hidden_size)
        else:
            bias_ih = bias_hh = None

        hidden_size = ctx.hidden_size
        has_bias = ctx.has_bias

        # Initialize gradients
        grad_input = None
        grad_hx = None
        grad_weight_ih = None
        grad_weight_hh = None
        grad_bias_ih = None
        grad_bias_hh = None

        # Extract individual weight matrices for each gate
        # Input-to-hidden weights for each gate
        W_ir = weight_ih[:hidden_size, :]  # reset gate
        W_iz = weight_ih[hidden_size : 2 * hidden_size, :]  # update gate
        W_in = weight_ih[2 * hidden_size :, :]  # new gate

        # Hidden-to-hidden weights for each gate
        W_hr = weight_hh[:hidden_size, :]  # reset gate
        W_hz = weight_hh[hidden_size : 2 * hidden_size, :]  # update gate
        W_hn = weight_hh[2 * hidden_size :, :]  # new gate

        # Extract bias vectors if they exist
        if has_bias:
            b_ir = bias_ih[:hidden_size]  # reset input bias
            b_iz = bias_ih[hidden_size : 2 * hidden_size]  # update input bias
            b_in = bias_ih[2 * hidden_size :]  # new input bias

            b_hr = bias_hh[:hidden_size]  # reset hidden bias
            b_hz = bias_hh[hidden_size : 2 * hidden_size]  # update hidden bias
            b_hn = bias_hh[2 * hidden_size :]  # new hidden bias
        else:
            b_ir = b_iz = b_in = b_hr = b_hz = b_hn = 0

        # FORWARD PASS RECONSTRUCTION
        # Gate pre-activations
        reset_ih = torch.mm(M_input, W_ir.t()) + b_ir
        reset_hi = torch.mm(hx, W_hr.t()) + b_hr
        reset_pre = reset_ih + reset_hi

        update_ih = torch.mm(M_input, W_iz.t()) + b_iz
        update_hi = torch.mm(hx, W_hz.t()) + b_hz
        update_pre = update_ih + update_hi

        new_ih = torch.mm(M_input, W_in.t()) + b_in
        new_hi = torch.mm(hx, W_hn.t()) + b_hn
        new_pre = new_ih + new_hi

        # Gate activations
        r_t = torch.sigmoid(reset_pre)
        z_t = torch.sigmoid(update_pre)
        n_t = torch.tanh(new_pre)

        # Output computation
        (1 - z_t) * n_t + z_t * hx

        # BACKWARD PASS
        # Total gradient w.r.t. output
        #   equation: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
        grad_z_t = grad_output * (hx - n_t)
        grad_n_t = grad_output * (1 - z_t)
        grad_hx_direct = grad_output * z_t

        # Gradient w.r.t. gate pre-activations
        grad_reset_pre = (
            grad_n_t * r_t * (1 - r_t)
        )  # No direct connection, but through new gate
        grad_update_pre = grad_z_t * z_t * (1 - z_t)
        grad_new_pre = grad_n_t * (1 - n_t.pow(2))

        # FORWARD PASS RECONSTRUCTION
        reset_ih = torch.mm(M_input, W_ir.t()) + b_ir
        reset_hi = torch.mm(hx, W_hr.t()) + b_hr
        reset_pre = reset_ih + reset_hi

        update_ih = torch.mm(M_input, W_iz.t()) + b_iz
        update_hi = torch.mm(hx, W_hz.t()) + b_hz
        update_pre = update_ih + update_hi

        new_ih = torch.mm(M_input, W_in.t()) + b_in
        new_hi = torch.mm(hx, W_hn.t()) + b_hn

        # Gate activations
        r_t = torch.sigmoid(reset_pre)
        z_t = torch.sigmoid(update_pre)

        # New gate computation with reset gate
        new_pre = new_ih + r_t * new_hi
        n_t = torch.tanh(new_pre)

        # Output computation
        (1 - z_t) * n_t + z_t * hx

        # BACKWARD PASS
        # Gradient w.r.t. output equation
        grad_z_t = grad_output * (hx - n_t)
        grad_n_t = grad_output * (1 - z_t)
        grad_hx_direct = grad_output * z_t

        # Gradient w.r.t. new gate pre-activation
        grad_new_pre = grad_n_t * (1 - n_t.pow(2))

        # Gradient w.r.t. new gate components
        grad_new_ih = grad_new_pre
        grad_r_t = grad_new_pre * new_hi
        grad_new_hi = grad_new_pre * r_t

        # Gradient w.r.t. update gate
        grad_update_pre = grad_z_t * z_t * (1 - z_t)

        # Gradient w.r.t. reset gate
        grad_reset_pre = grad_r_t * r_t * (1 - r_t)

        # Gradient w.r.t. gate pre-activation components
        grad_reset_ih = grad_reset_pre
        grad_reset_hi = grad_reset_pre
        grad_update_ih = grad_update_pre
        grad_update_hi = grad_update_pre

        # GRADIENTS W.R.T. INPUTS
        if ctx.needs_input_grad[0]:  # grad_input (x_t)
            grad_input = (
                torch.mm(grad_reset_ih, W_ir)
                + torch.mm(grad_update_ih, W_iz)
                + torch.mm(grad_new_ih, W_in)
            )

        if ctx.needs_input_grad[1]:  # grad_hx (h_{t-1})
            grad_hx = (
                grad_hx_direct
                + torch.mm(grad_reset_hi, W_hr)
                + torch.mm(grad_update_hi, W_hz)
                + torch.mm(grad_new_hi, W_hn)
            )

        # GRADIENTS W.R.T. WEIGHTS
        if ctx.needs_input_grad[2]:  # grad_weight_ih
            if M_input.dim() == 1:
                grad_W_ir = torch.outer(grad_reset_ih.squeeze(0), M_input)
                grad_W_iz = torch.outer(grad_update_ih.squeeze(0), M_input)
                grad_W_in = torch.outer(grad_new_ih.squeeze(0), M_input)
            else:
                grad_W_ir = torch.mm(grad_reset_ih.t(), M_input)
                grad_W_iz = torch.mm(grad_update_ih.t(), M_input)
                grad_W_in = torch.mm(grad_new_ih.t(), M_input)

            grad_weight_ih = torch.cat([grad_W_ir, grad_W_iz, grad_W_in], dim=0)

        if ctx.needs_input_grad[3]:  # grad_weight_hh
            if hx.dim() == 1:
                grad_W_hr = torch.outer(grad_reset_hi.squeeze(0), hx)
                grad_W_hz = torch.outer(grad_update_hi.squeeze(0), hx)
                grad_W_hn = torch.outer(grad_new_hi.squeeze(0), hx)
            else:
                grad_W_hr = torch.mm(grad_reset_hi.t(), hx)
                grad_W_hz = torch.mm(grad_update_hi.t(), hx)
                grad_W_hn = torch.mm(grad_new_hi.t(), hx)

            grad_weight_hh = torch.cat([grad_W_hr, grad_W_hz, grad_W_hn], dim=0)

        # GRADIENTS W.R.T. BIASES
        if has_bias:
            if ctx.needs_input_grad[4]:  # grad_bias_ih
                grad_b_ir = grad_reset_ih.sum(dim=0)
                grad_b_iz = grad_update_ih.sum(dim=0)
                grad_b_in = grad_new_ih.sum(dim=0)
                grad_bias_ih = torch.cat([grad_b_ir, grad_b_iz, grad_b_in], dim=0)

            if ctx.needs_input_grad[5]:  # grad_bias_hh
                grad_b_hr = grad_reset_hi.sum(dim=0)
                grad_b_hz = grad_update_hi.sum(dim=0)
                grad_b_hn = grad_new_hi.sum(dim=0)
                grad_bias_hh = torch.cat([grad_b_hr, grad_b_hz, grad_b_hn], dim=0)

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
            None,  # has_bias (no gradient needed)
            None,  # training (no gradient needed)
        )
