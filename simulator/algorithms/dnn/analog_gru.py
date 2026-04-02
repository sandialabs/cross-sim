#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""CrossSim implementation of GRU (Gated Recurrent Unit) layers."""

from __future__ import annotations

from simulator import CrossSimParameters
from simulator.backend import ComputeBackend
from .analog_layer import AnalogLayer
from .analog_gru_cell import AnalogGRUCell

import numpy.typing as npt

xp = ComputeBackend()


class AnalogGRU(AnalogLayer):
    """CrossSim implementation for GRU (Gated Recurrent Unit) layers."""

    def __init__(
        self,
        params: CrossSimParameters | list[CrossSimParameters],
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias_ih_rows: int | list[int],
        bias_hh_rows: int | list[int],
    ) -> None:
        """Initializes a GRU layer wrapper around multiple AnalogGRUCell layers.

        Args:
            params:
                CrossSimParameters object or list of CrossSimParameters (for
                layers requiring multiple arrays) for the AnalogGRU layer. If a
                list, the length must match the number of arrays used within
                AnalogCore.
            input_size:
                Number of input features for the first layer.
            hidden_size:
                Number of hidden features for all layers.
            num_layers:
                Number of recurrent layers.
            bias_ih_rows:
                Integer indicating the number of rows to use to implement the
                bias_ih within the array. 0 implies a digital bias.
            bias_hh_rows:
                Integer indicating the number of rows to use to implement the
                bias_hh within the array. 0 implies a digital bias.
            bias:
                If False, then the layer does not use bias weights.
        """
        if not isinstance(params, list):
            self.params = params
        else:
            self.params = params[0].copy()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if isinstance(bias_ih_rows, list) and len(bias_ih_rows) != num_layers:
            raise ValueError(
                f"bias_ih_rows of length {len(bias_ih_rows)} must be an int or "
                f"match num_layers {num_layers}"
            )
        if isinstance(bias_hh_rows, list) and len(bias_hh_rows) != num_layers:
            raise ValueError(
                f"bias_ih_rows of length {len(bias_hh_rows)} must be an int or "
                f"match num_layers {num_layers}"
            )
        self.bias_ih_rows = bias_ih_rows
        self.bias_hh_rows = bias_hh_rows

        # Create AnalogGRUCell instances for each layer
        self.cells = []
        for layer_idx in range(num_layers):
            # For the first layer, input size is the actual input size
            # For subsequent layers, input size is the hidden size of the
            # previous layer
            cell = AnalogGRUCell(
                params,
                input_size if layer_idx == 0 else hidden_size,
                hidden_size,
                bias_ih_rows
                if isinstance(bias_ih_rows, int)
                else bias_ih_rows[layer_idx],
                bias_hh_rows
                if isinstance(bias_hh_rows, int)
                else bias_hh_rows[layer_idx],
            )
            self.cells.append(cell)

    def form_matrix(
        self,
        weight_ih: npt.ArrayLike,
        weight_hh: npt.ArrayLike,
        bias_ih: npt.ArrayLike | None = None,
        bias_hh: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
        """Build a list of 2D weight matrices for programming into arrays.

        Args:
            weight_ih:
                2D numpy NDArray (or similar) with the input-to-hidden weight
                matrix.
            weight_hh:
                2D numpy NDArray (or similar) with the hidden-to-hidden weight
                matrix.
            bias_ih:
                1D numpy NDArray (or similar) with the input-to-hidden bias
                vector. Ignored if the matrix does not use an analog bias.
            bias_hh:
                1D numpy NDArray (or similar) with the hidden-to-hidden bias
                vector. Ignored if the matrix does not use an analog bias.

        Returns:
            2D numpy NDArray of the matrix for the specified layer.
        """
        mats = [None] * self.num_layers
        for i in range(self.num_layers):
            mats[i] = self.cells[i].form_matrix(
                weight_ih[i],
                weight_hh[i],
                bias_ih[i] if bias_ih is not None else None,
                bias_hh[i] if bias_hh is not None else None,
            )
        return mats

    def get_core_weights(self) -> tuple[npt.NDArray | None, ...]:
        """Gets the weight and bias matrices with errors applied.

        This function only returns weight and bias values which can be derived
        from the stored array values. If the layer uses a digital bias the
        returned bias will be None.

        Returns:
            Tuple of numpy arrays ordered as weights first then bias with ih and
            hh interleaved within both weights and bias. For instance for a 2
            layer RNN w_ih0, w_hh0, w_ih1, w_hh1, b_ih0, b_hh0, b_ih1, b_hh1.
            This is consistent with the order of torch named parameters.
        """
        weights = [None] * 2 * self.num_layers
        biases = [None] * 2 * self.num_layers
        for layer in range(self.num_layers):
            w_ih, w_hh, b_ih, b_hh = self.cells[layer].get_core_weights()
            weights[layer * 2] = w_ih
            weights[layer * 2 + 1] = w_hh
            biases[layer * 2] = b_ih
            biases[layer * 2 + 1] = b_hh

        return (*weights, *biases)

    def apply(
        self,
        layer_idx: int,
        M_input: npt.ArrayLike,
        hx: npt.ArrayLike,
    ) -> npt.NDArray:
        """Apply analog computation for a specific layer..

        This method only handles the analog matrix-vector multiplication.
        All digital processing (loops, activations, GRU logic) is handled
        in the PyTorch forward method.

        Args:
            layer_idx: Index of the layer to apply
            M_input: Input tensor
            hx: Previous hidden state

        Returns:
            Raw gate outputs before activations
        """
        if layer_idx >= self.num_layers:
            raise ValueError(
                f"Layer index {layer_idx} is out of range for {self.num_layers} layers"
            )

        return self.cells[layer_idx].apply(M_input, hx)

    def get_matrix(self) -> list[npt.NDArray]:
        """Returns the programmed 2D analog array for a specific layer.

        Args:
            layer_idx: Index of the layer to get matrix for.

        Returns:
            Numpy array of the 2D array with non-idealities applied.
        """
        return [self.cells[layers].get_matrix() for layers in range(self.num_layers)]

    def set_matrix(self, matrix: list[npt.ArrayLike], verbose: bool = False) -> None:
        """Programs a matrix into a specific layer's internal AnalogCore.

        Args:
            layer_idx: Index of the layer to set matrix for.
            matrix: Numpy ndarray to be programmed into the array.
            verbose: Boolean flag to enable verbose print statements.
        """
        for layers in range(self.num_layers):
            self.cells[layers].set_matrix(matrix[layers], verbose)

    # Properties that aggregate information from all cells
    @property
    def shape(self) -> list[tuple[int, int]]:
        """Shape of 2D matrices representing each layer.

        Returns:
            List of shapes for each layer's matrix.
        """
        return [cell.shape for cell in self.cells]

    @property
    def max(self) -> list[float]:
        """Internal AnalogCore's defined maximum matrix values for each layer.

        Returns:
            List of maximum values for each layer.
        """
        return [cell.max for cell in self.cells]

    @property
    def min(self) -> list[float]:
        """Internal AnalogCore's defined minimum matrix values for each layer.

        Returns:
            List of minimum values for each layer.
        """
        return [cell.min for cell in self.cells]

    @property
    def Ncores(self) -> list[int]:
        """Number of partitions in each internal AnalogCore.

        Returns:
            List of number of cores for each layer.
        """
        return [cell.Ncores for cell in self.cells]

    @property
    def dtype(self) -> npt.DTypeLike:
        """Datatype of the matrices stored in the internal AnalogCores.

        Assumes all layers use the same dtype.
        """
        return self.cells[0].dtype if self.cells else None

    def __getitem__(self, layer_idx: int) -> AnalogGRUCell:
        """Get a specific GRU cell by layer index.

        Args:
            layer_idx: Index of the layer to retrieve.

        Returns:
            AnalogGRUCell for the specified layer.
        """
        if layer_idx >= self.num_layers:
            raise ValueError(
                f"Layer index {layer_idx} is out of range for {self.num_layers} layers"
            )

        return self.cells[layer_idx]

    def __len__(self) -> int:
        """Get the number of GRU layers.

        Returns:
            Number of layers in the GRU.
        """
        return self.num_layers

    # For now hard coding the relevant prefixes because we know all of them
    # Likely needs to be changed for Keras support
    _variable_names = {"weight_ih", "weight_hh", "bias_ih", "bias_hh"}

    def __getattr__(self, name):
        """Forwards mask attributes from the cells to the RNN."""
        # Matching variables of the format [var]_l[i]_mask
        try:
            var, remainder = name.split("_l", 1)
        except ValueError:
            raise AttributeError(f"AnalogRNN has no attributes {name}") from None

        if var not in AnalogGRU._variable_names:
            raise AttributeError(f"AnalogRNN has no attributes {name}")

        if not remainder.endswith("_mask"):
            raise AttributeError(f"AnalogRNN has no attributes {name}")

        layer = int(remainder.removesuffix("_mask"))

        if not layer < self.num_layers:
            raise AttributeError(f"AnalogRNN has no attributes {name}")

        return getattr(self.cells[layer], f"{var}_mask")
