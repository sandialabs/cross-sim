#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""CrossSim version of Torch.nn.LSTM.

AnalogLSTM provides a CrossSim-based forward using analog MVM.
"""

from __future__ import annotations

from .layer import AnalogLayer
from .lstm_cell import AnalogLSTMCellGrad

from simulator import CrossSimParameters
from simulator.algorithms.dnn.analog_lstm import AnalogLSTM as LSTMCore
from torch import Tensor, from_dlpack, zeros
from torch.nn import LSTM, Parameter
import torch


class AnalogLSTM(LSTM, AnalogLayer):
    """CrossSim implementation of torch.nn.LSTM.

    See AnalogLayer for description of CrossSim-specific documentation.
    See torch.nn.LSTM for layer functionality documentation.
    """

    def __init__(
        self,
        params: CrossSimParameters | list[CrossSimParameters],
        # Base LSTM layer arguments
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device=None,
        dtype=None,
        # Additional arguments for AnalogLSTM specifically
        bias_ih_rows: int | list[int] = 0,
        bias_hh_rows: int | list[int] = 0,
    ) -> None:
        """Initializes AnalogLSTM and underlying torch.nn.LSTM layer.

        Args:
            params:
                CrossSimParameters object or list of CrossSimParameters (for
                layers requiring multiple arrays) for the AnalogRNNCell layer.
                If a list, the length must match the number of arrays used
                within AnalogCore.
            input_size: See torch.nn.LSTM input_size argument.
            hidden_size: See torch.nn.LSTM hidden_size argument.
            num_layers: See torch.nn.LSTM num_layers argument.
            bias: See torch.nn.LSTM bias argument.
            batch_first: See torch.nn.LSTM batch_first argument.
            dropout: See torch.nn.LSTM dropout argument.
            bidirectional: See torch.nn.LSTM bidirectional argument.
            proj_size: See torch.nn.LSTM proj_size argument.
            device: See torch.nn.LSTM device argument.
            dtype: See torch.nn.LSTM dtype argument.
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
        """
        if bidirectional:
            raise NotImplementedError("Bidirectional AnalogLSTM is not supported yet.")

        if proj_size != 0:
            raise NotImplementedError("Projection size is not supported in AnalogLSTM")

        device_ = AnalogLayer._set_device(
            device,
            params[0] if isinstance(params, list) else params,
        )

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
            device=device_,
            dtype=dtype,
        )

        self._array_weight_variables = [
            x for i in range(num_layers) for x in (f"weight_ih_l{i}", f"weight_hh_l{i}")
        ]

        self._array_bias_variables = [
            x for i in range(num_layers) for x in (f"bias_ih_l{i}", f"bias_hh_l{i}")
        ]
        # Use the fact that not all biases need to appear. This means we don't
        # need a special __setattr__ hook for reinitialize
        self._array_bias_variables += ["bias_ih", "bias_hh"]

        if isinstance(params, CrossSimParameters):
            self.params = params.copy()
        elif isinstance(params, list):
            self.params = params[0].copy()

        if isinstance(bias_ih_rows, list):
            self.bias_ih_rows = bias_ih_rows
        else:
            self.bias_ih_rows = [bias_ih_rows] * num_layers

        if isinstance(bias_hh_rows, list):
            self.bias_hh_rows = bias_hh_rows
        else:
            self.bias_hh_rows = [bias_hh_rows] * num_layers

        self.core = LSTMCore(
            params,
            input_size,
            hidden_size,
            num_layers,
            bias_ih_rows,
            bias_hh_rows,
        )

        # form_matrix returns a list so we don't need to detach
        self.core.set_matrix(self.form_matrix())

    def form_matrix(self) -> list[Tensor]:
        """Build a list of 2D weight matrices for programming into arrays."""
        weight_ih_ = [
            getattr(self, f"weight_ih_l{i}").detach().cpu()
            for i in range(self.num_layers)
        ]
        weight_hh_ = [
            getattr(self, f"weight_hh_l{i}").detach().cpu()
            for i in range(self.num_layers)
        ]
        bias_ih_ = [
            getattr(self, f"bias_ih_l{i}").detach().cpu()
            if getattr(self, f"analog_bias_ih_l{i}")
            else None
            for i in range(self.num_layers)
        ]
        bias_hh_ = [
            getattr(self, f"bias_hh_l{i}").detach().cpu()
            if getattr(self, f"analog_bias_hh_l{i}")
            else None
            for i in range(self.num_layers)
        ]
        return [
            from_dlpack(i)
            for i in self.core.form_matrix(weight_ih_, weight_hh_, bias_ih_, bias_hh_)
        ]

    def forward(  # noqa:C901
        self, M_input: Tensor, hx: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """LSTM forward operation using per-cell analog computation.

        Processes the input sequence by iterating over timesteps and layers,
        delegating each (timestep, layer) step to AnalogLSTMCellGrad. This
        allows PyTorch's autograd to automatically chain the per-cell backward
        passes, implementing BPTT without a separate monolithic backward.

        The analog matrix-vector multiplication is handled by each cell's core.
        All other operations (sequencing, layer stacking, dropout, gate
        activations, LSTM recurrence) are performed digitally in PyTorch.

        Args:
            M_input: Input tensor of shape (seq_len, batch, input_size) or
                (batch, seq_len, input_size) if batch_first=True.
                Also accepts 2D input (seq_len, input_size) for unbatched.
            hx: Tuple of (h0, c0) initial states, each of shape
                (num_layers, batch, hidden_size). Defaults to zeros if not
                provided.

        Returns:
            Tuple of (output, (h_n, c_n)) where:
                output: Hidden states from the last layer at each timestep,
                    shape (seq_len, batch, hidden_size) or
                    (batch, seq_len, hidden_size) if batch_first=True.
                h_n: Final hidden state for each layer,
                    shape (num_layers, batch, hidden_size).
                c_n: Final cell state for each layer,
                    shape (num_layers, batch, hidden_size).
        """
        # Handle input dimensions
        if M_input.dim() not in (2, 3):
            raise ValueError(
                f"LSTM: Expected input to be 2D or 3D, got {M_input.dim()}D instead"
            )

        is_batched = M_input.dim() == 3
        batch_dim = 0 if self.batch_first else 1

        if not is_batched:
            M_input = M_input.unsqueeze(batch_dim)

        max_batch_size = M_input.size(0) if self.batch_first else M_input.size(1)

        if hx is None:
            h0 = zeros(
                self.num_layers,
                max_batch_size,
                self.hidden_size,
                dtype=M_input.dtype,
                device=M_input.device,
            )
            c0 = zeros(
                self.num_layers,
                max_batch_size,
                self.hidden_size,
                dtype=M_input.dtype,
                device=M_input.device,
            )
        else:
            h0, c0 = hx
            if is_batched:
                if h0.dim() != 3 or c0.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, h0 and c0 should also be 3-D "
                        f"but got ({h0.dim()}-D, {c0.dim()}-D) tensors"
                    )
            else:
                if h0.dim() != 2 or c0.dim() != 2:
                    raise RuntimeError(
                        f"For unbatched 2-D input, h0 and c0 should also be 2-D "
                        f"but got ({h0.dim()}-D, {c0.dim()}-D) tensors"
                    )
                h0 = h0.unsqueeze(1)
                c0 = c0.unsqueeze(1)

        # Convert to (seq_len, batch_size, input_size)
        if self.batch_first:
            batch_size, seq_len, _ = M_input.shape
            M_input = M_input.permute(1, 0, 2)
        else:
            seq_len, batch_size, _ = M_input.shape

        if h0.shape != (self.num_layers, batch_size, self.hidden_size):
            raise ValueError(
                f"Expected h0 shape ({self.num_layers}, {batch_size}, "
                f"{self.hidden_size}), got {h0.shape}"
            )
        if c0.shape != (self.num_layers, batch_size, self.hidden_size):
            raise ValueError(
                f"Expected c0 shape ({self.num_layers}, {batch_size}, "
                f"{self.hidden_size}), got {c0.shape}"
            )

        # Current hidden and cell states for all layers
        h_current = [h0[layer_idx] for layer_idx in range(self.num_layers)]
        c_current = [c0[layer_idx] for layer_idx in range(self.num_layers)]

        outputs = []

        # Process each timestep
        for t in range(seq_len):
            layer_input = M_input[t]

            for layer_idx in range(self.num_layers):
                # Grab the per-layer weights
                weight_ih = getattr(self, f"weight_ih_l{layer_idx}")
                weight_hh = getattr(self, f"weight_hh_l{layer_idx}")
                bias_ih = getattr(self, f"bias_ih_l{layer_idx}", None)
                bias_hh = getattr(self, f"bias_hh_l{layer_idx}", None)

                # Use the cell-level autograd function.
                h_current[layer_idx], c_current[layer_idx] = AnalogLSTMCellGrad.apply(
                    layer_input,
                    h_current[layer_idx],
                    c_current[layer_idx],
                    weight_ih,
                    weight_hh,
                    bias_ih,
                    bias_hh,
                    self.core[layer_idx],
                    self.bias_ih_rows[layer_idx],
                    self.bias_hh_rows[layer_idx],
                    self.hidden_size,
                    self.bias,
                    self.training,
                )

                # Dropout between layers (not after last layer)
                if layer_idx < self.num_layers - 1:
                    if self.training and self.dropout > 0.0:
                        layer_input = torch.nn.functional.dropout(
                            h_current[layer_idx],
                            self.dropout,
                            training=True,
                            inplace=False,
                        )
                    else:
                        layer_input = h_current[layer_idx]

            outputs.append(h_current[-1])

        # Stack outputs: (seq_len, batch_size, hidden_size)
        output = torch.stack(outputs, dim=0)

        # Stack final states: (num_layers, batch_size, hidden_size)
        h_n = torch.stack(h_current, dim=0)
        c_n = torch.stack(c_current, dim=0)

        # Convert back to batch_first if needed
        if self.batch_first:
            output = output.transpose(0, 1)

        # Handle unbatched case
        if not is_batched:
            output = torch.squeeze(output, dim=batch_dim)
            h_n = torch.squeeze(h_n, dim=1)
            c_n = torch.squeeze(c_n, dim=1)

        return output, (h_n, c_n)

    def reinitialize(self) -> None:
        """Rebuilds the layer's internal core object."""
        self._initialized = False

        self.core = LSTMCore(
            self.params,
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.bias_ih_rows,
            self.bias_hh_rows,
        )
        self.core.set_matrix(self.form_matrix())

    def synchronize(self) -> None:
        """Updates the analog weight representation with weight parameters."""
        self.core.set_matrix(self.form_matrix())

    @classmethod
    def from_torch(
        cls,
        layer: LSTM,
        params: CrossSimParameters | list[CrossSimParameters],
        bias_ih_rows: int | list[int] = 0,
        bias_hh_rows: int | list[int] = 0,
    ) -> AnalogLSTM:
        """Build AnalogLSTM from an LSTM layer.

        Args:
            layer: torch.nn.LSTM layer to copy.
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
            AnalogLSTM layer with the same weights and properties as input
            LSTM layer.
        """
        if layer.bidirectional:
            raise ValueError("Bidirectional LSTM is not supported")
        if layer.proj_size != 0:
            raise ValueError("Projection size is not supported in AnalogLSTM")

        device = AnalogLayer._set_device(
            layer.weight_ih_l0.device,
            params[0] if isinstance(params, list) else params,
        )

        analog_layer = cls(
            params,
            layer.input_size,
            layer.hidden_size,
            layer.num_layers,
            layer.bias,
            layer.batch_first,
            layer.dropout,
            layer.bidirectional,
            layer.proj_size,
            device,
            layer.weight_ih_l0.dtype,
            bias_ih_rows,
            bias_hh_rows,
        )

        # Copy weights and biases for all layers
        for layer_idx in range(layer.num_layers):
            setattr(
                analog_layer,
                f"weight_ih_l{layer_idx}",
                getattr(layer, f"weight_ih_l{layer_idx}"),
            )
            setattr(
                analog_layer,
                f"weight_hh_l{layer_idx}",
                getattr(layer, f"weight_hh_l{layer_idx}"),
            )
            if layer.bias:
                setattr(
                    analog_layer,
                    f"bias_ih_l{layer_idx}",
                    getattr(layer, f"bias_ih_l{layer_idx}"),
                )
                setattr(
                    analog_layer,
                    f"bias_hh_l{layer_idx}",
                    getattr(layer, f"bias_hh_l{layer_idx}"),
                )

        return analog_layer

    @classmethod
    def to_torch(  # noqa:C901
        cls,
        layer: AnalogLSTM,
        physical_weights: bool = False,
        device=None,
        dtype=None,
    ) -> LSTM:
        """Creates a torch LSTM layer from an AnalogLSTM.

        Args:
            layer: AnalogLSTM layer to copy.
            physical_weights:
                Bool indicating whether the torch layer should have ideal
                weights or weights with programming error applied.
            device:
                The device where the layer will be placed.
            dtype:
                The dtype of the layer weights.

        Returns:
            torch.nn.LSTM with the same properties and weights as the AnalogLSTM
            layer.
        """
        if not device:
            if layer.params.simulation.useGPU:
                device = "cuda:{}".format(layer.params.simulation.gpu_id)
            else:
                device = "cpu"

        if not dtype:
            dtype = AnalogLayer._numpy_to_torch_dtype_dict[layer.core.dtype]

        torch_layer = LSTM(
            layer.input_size,
            layer.hidden_size,
            layer.num_layers,
            bias=layer.bias,
            batch_first=layer.batch_first,
            dropout=layer.dropout,
            bidirectional=layer.bidirectional,
            proj_size=layer.proj_size,
            device=device,
            dtype=dtype,
        )

        if physical_weights:
            pweights = layer.core.get_core_weights()

        for layer_idx in range(layer.num_layers):
            if physical_weights:
                w_ih = from_dlpack(pweights[layer_idx * 2])
                w_hh = from_dlpack(pweights[layer_idx * 2 + 1])
                b_ih = pweights[layer.num_layers * 2 + layer_idx * 2]
                b_hh = pweights[layer.num_layers * 2 + layer_idx * 2 + 1]
                if layer.bias is not None:
                    if getattr(layer, f"analog_bias_ih_l{layer_idx}"):
                        b_ih = from_dlpack(b_ih) if b_ih is not None else None
                    else:
                        b_ih = getattr(f"bias_ih_l{layer_idx}").clone().detach()
                    if getattr(layer, f"analog_bias_hh_l{layer_idx}"):
                        b_hh = from_dlpack(b_hh) if b_hh is not None else None
                    else:
                        b_hh = getattr(f"bias_hh_l{layer_idx}").clone().detach()

            else:
                # Get ideal weights
                w_ih = getattr(layer, f"weight_ih_l{layer_idx}").clone().detach()
                w_hh = getattr(layer, f"weight_hh_l{layer_idx}").clone().detach()
                if layer.bias:
                    b_ih = getattr(layer, f"bias_ih_l{layer_idx}").clone().detach()
                    b_hh = getattr(layer, f"bias_hh_l{layer_idx}").clone().detach()
                else:
                    b_ih = None
                    b_hh = None

            # Set weights
            setattr(torch_layer, f"weight_ih_l{layer_idx}", Parameter(w_ih))
            setattr(torch_layer, f"weight_hh_l{layer_idx}", Parameter(w_hh))

            # Set biases if they exist
            if b_ih is not None:
                setattr(torch_layer, f"bias_ih_l{layer_idx}", Parameter(b_ih))
            if b_hh is not None:
                setattr(torch_layer, f"bias_hh_l{layer_idx}", Parameter(b_hh))

        return torch_layer

        return torch_layer

    def __setattr__(self, name, value) -> None:
        """Converts bias_row sets into list accesses."""
        # Converts integer refernces into lists and specifically named values
        # update their list entry and then forward that up.
        # We still want to make sure the torch __setattr__ is first so we don't
        # change anything here, just modify the value and pass it up.
        if (
            name.endswith("_rows")
            and name.removesuffix("_rows") in self._array_bias_variables
        ):
            try:
                var, layer = name.split("_l", 1)
            except ValueError:
                # Doesn't contain the _l pattern means we're dealing with the
                # non sublayer specific value so convert to list and forward up
                if isinstance(value, list):
                    if len(value) != self.num_layers:
                        raise ValueError(
                            f"Lists for {name} must equal number of layers. "
                            f"Expected: {self.num_layers}, got {len(value)}."
                        ) from None
                    value_ = value
                else:
                    value_ = [value] * self.num_layers

                super().__setattr__(name, value_)
                return

            layer = int(layer.removesuffix("_rows"))
            value_ = getattr(self, f"{var}_rows")
            value_[layer] = value
            super().__setattr__(name, value_)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        """Converts accesses by name to list accesses."""
        # Not what we're looking for, forward to above.
        # Needed because torch also does some screwy stuff with __getattr__
        if not (name.startswith("analog_") or name.endswith("_rows")):
            return super().__getattr__(name)

        # Both patterns have a split on _l[int] except names without a
        # sublayer specifier.  This is already a member so just do
        # getattr (will also handle errors).
        try:
            var, layer = name.split("_l", 1)
        except ValueError:
            return getattr(self, name)

        if name.removeprefix("analog_") in self._array_bias_variables:
            layer = int(layer)
        elif name.removesuffix("_rows") in self._array_bias_variables:
            layer = int(layer.removesuffix("_rows"))
        else:
            raise AttributeError(f"AnalogRNN has no attribute {name}")

        return getattr(self, var)[layer]

    @property
    def analog_bias_ih(self):
        """List of sublayer analog_bias_ih properties for each sublayer."""
        return [bool(self.bias and r > 0) for r in self.bias_ih_rows]

    @property
    def analog_bias_hh(self):
        """List of sublayer analog_bias_hh properties for each sublayer."""
        return [bool(self.bias and r > 0) for r in self.bias_hh_rows]
