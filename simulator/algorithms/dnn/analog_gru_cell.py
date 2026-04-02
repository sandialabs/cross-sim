#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""CrossSim implementation of GRU (Gated Recurrent Unit) cell layers."""

from __future__ import annotations

import numpy as np
from simulator import AnalogCore, CrossSimParameters
from simulator.backend import ComputeBackend
from .analog_layer import AnalogLayer

import numpy.typing as npt

xp = ComputeBackend()


class AnalogGRUCell(AnalogLayer):
    """CrossSim implementation for GRUCell (Gated Recurrent Unit Cell) layers.

    The matrix structure follows the diagram:
    Rows 0 to h-1:     [W_ir, W_hr] for reset gate
    Rows h to 2h-1:    [W_iz, W_hz] for update gate
    Rows 2h to 3h-1:   [W_in, 0   ] for new gate input component
    Rows 3h to 4h-1:   [0,    W_hn] for new gate hidden component

    """

    def __init__(
        self,
        params: CrossSimParameters | list[CrossSimParameters],
        input_size: int,
        hidden_size: int,
        bias_ih_rows: int,
        bias_hh_rows: int,
    ) -> None:
        """Initializes a GRUCell layer wrapper around AnalogCore.

        Args:
            params:
                CrossSimParameters object or list of CrossSimParameters (for
                layers requiring multiple arrays) for the AnalogGRUCell layer.
                If a list, the length must match the number of arrays used
                within AnalogCore.
            input_size:
                Number of input features, see torch.nn.GRUCell input_size for
                detailed documentation.
            hidden_size:
                Number of hidden features, see torch.nn.GRUCell hidden_size for
                detailed documentation.
            bias_ih_rows:
                Integer indicating the number of rows to use to implement
                bias_ih within the array. 0 implies a digital bias.
            bias_hh_rows:
                Integer indicating the number of rows to use to implement
                bias_hh within the array. 0 implies a digital bias.
        """
        if not isinstance(params, list):
            self.params = params
        else:
            self.params = params[0].copy()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_ih_rows = bias_ih_rows
        self.bias_hh_rows = bias_hh_rows

        # Initialize xbar_inputs for input profiling
        self.xbar_inputs = None

        total_input_size = input_size + hidden_size
        total_cols = total_input_size + bias_ih_rows + bias_hh_rows

        # Since the bottom quarter of weight_ih is all zeros don't include it
        # in the mask. By contrast weight_hh covers all 4 including the all
        # zero section
        self.weight_ih_mask = (slice(0, 3 * hidden_size, 1), slice(0, input_size, 1))
        self.weight_hh_mask = (
            slice(0, 4 * hidden_size, 1),
            slice(input_size, total_input_size, 1),
        )
        self.bias_ih_mask = (
            slice(0, 3 * hidden_size, 1),
            slice(total_input_size, total_input_size + bias_ih_rows, 1),
        )
        self.bias_hh_mask = (
            slice(0, 4 * hidden_size, 1),
            slice(
                total_input_size + bias_ih_rows,
                total_input_size + bias_ih_rows + bias_hh_rows,
                1,
            ),
        )
        # Output Size: 4 * hidden_size (reset, update, new_input, new_hidden)
        self.core = AnalogCore(
            np.zeros((4 * hidden_size, total_cols)),
            params,
            empty_matrix=True,
        )

    def form_matrix(
        self,
        weight_ih: npt.ArrayLike,
        weight_hh: npt.ArrayLike,
        bias_ih: npt.ArrayLike | None = None,
        bias_hh: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
        """Build a 2D weight matrix for the GRU layer.

        The matrix structure follows the diagram:
        Rows 0 to h-1:     [W_ir, W_hr] for reset gate
        Rows h to 2h-1:    [W_iz, W_hz] for update gate
        Rows 2h to 3h-1:   [W_in, 0   ] for new gate input component
        Rows 3h to 4h-1:   [0,    W_hn] for new gate hidden component

        Input vector structure: [xt, ht-1] + bias_columns

        Args:
            weight_ih:
                2D numpy NDArray with input-to-hidden weights.
                Shape: (3*hidden_size, input_size)
                Contains [W_ir; W_iz; W_in] stacked vertically.
            weight_hh:
                2D numpy NDArray with hidden-to-hidden weights.
                Shape: (3*hidden_size, hidden_size)
                Contains [W_hr; W_hz; W_hn] stacked vertically.
            bias_ih:
                1D numpy NDArray with input-to-hidden bias.
                Shape: (3*hidden_size). Ignored if not using analog bias.
            bias_hh:
                1D numpy NDArray with hidden-to-hidden bias.
                Shape: (3*hidden_size). Ignored if not using analog bias.

        Returns:
            2D numpy NDArray of the structured matrix.
        """
        w_ih = np.asarray(weight_ih)
        w_hh = np.asarray(weight_hh)

        if bias_ih is not None:
            b_ih = np.asarray(bias_ih)
        if bias_hh is not None:
            b_hh = np.asarray(bias_hh)

        if w_ih.shape != (3 * self.hidden_size, self.input_size):
            raise ValueError(
                f"Expected weight_ih shape ({3 * self.hidden_size}, "
                f"{self.input_size}), got {w_ih.shape}"
            )

        if w_hh.shape != (3 * self.hidden_size, self.hidden_size):
            raise ValueError(
                f"Expected weight_hh shape ({3 * self.hidden_size}, "
                f"{self.hidden_size}),  got {w_hh.shape}"
            )

        # Since all versions of GRU are non-contigious we can't just
        # concatenate. Always need to manually build the matrix
        total_cols = (
            self.input_size + self.hidden_size + self.bias_ih_rows + self.bias_hh_rows
        )
        matrix = np.zeros((4 * self.hidden_size, total_cols), dtype=w_ih.dtype)

        # w_ih is just the 3 hidden size blocks
        matrix[: 3 * self.hidden_size, : self.input_size] = w_ih

        # w_hh has 2 chunks which much be manually split
        matrix[
            : 2 * self.hidden_size, self.input_size : self.input_size + self.hidden_size
        ] = w_hh[: 2 * self.hidden_size, :]
        matrix[
            3 * self.hidden_size : 4 * self.hidden_size,
            self.input_size : self.input_size + self.hidden_size,
        ] = w_hh[2 * self.hidden_size :, :]

        if bias_ih is not None and self.bias_ih_rows:
            bias_start = self.input_size + self.hidden_size
            matrix[
                : 3 * self.hidden_size, bias_start : bias_start + self.bias_ih_rows
            ] = np.repeat(
                b_ih / self.bias_ih_rows,
                self.bias_ih_rows,
                axis=0,
            ).reshape(3 * self.hidden_size, self.bias_ih_rows)

        # bias_hh needs to be handled in 2 chunks as with weights
        if bias_hh is not None and self.bias_hh_rows:
            b_cols = np.repeat(
                b_hh / self.bias_hh_rows,
                self.bias_hh_rows,
                axis=0,
            ).reshape(-1, self.bias_hh_rows)
            bias_start = self.input_size + self.hidden_size + self.bias_ih_rows
            matrix[
                : 2 * self.hidden_size, bias_start : bias_start + self.bias_hh_rows
            ] = b_cols[: 2 * self.hidden_size, :]
            matrix[
                3 * self.hidden_size :, bias_start : bias_start + self.bias_hh_rows
            ] = b_cols[2 * self.hidden_size :, :]

        return matrix

    def get_core_weights(
        self,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray | None, npt.NDArray | None]:
        """Gets the weight and bias matrices with errors applied.

        Returns:
            Tuple of numpy arrays: (weight_ih, weight_hh, bias_ih, bias_hh).
            Bias arrays are None if digital bias is used.

            The weight matrices are reconstructed from the structured matrix:
            - weight_ih contains [W_ir; W_iz; W_in] stacked vertically
            - weight_hh contains [W_hr; W_hz; W_hn] stacked vertically
        """
        matrix = self.get_matrix()

        # Just extract everything as usual using the masks
        # Then we'll reconstruct the hh values.
        weight_ih = matrix[self.weight_ih_mask]
        weight_hh = matrix[self.weight_hh_mask]
        bias_ih = None

        if self.bias_ih_rows > 0:
            bias_ih = matrix[self.bias_ih_mask].sum(1)
        if self.bias_hh_rows > 0:
            b_hh = xp.zeros((3 * self.hidden_size), matrix.dtype)
            bias_hh = matrix[self.bias_hh_mask].sum(1)
            b_hh[: 2 * self.hidden_size] = bias_hh[: 2 * self.hidden_size]
            b_hh[2 * self.hidden_size :] = bias_hh[3 * self.hidden_size :]
        else:
            b_hh = None

        w_hh = xp.zeros((3 * self.hidden_size, self.hidden_size), matrix.dtype)
        w_hh[: 2 * self.hidden_size, :] = weight_hh[: 2 * self.hidden_size, :]
        w_hh[2 * self.hidden_size :, :] = weight_hh[3 * self.hidden_size :, :]

        return (weight_ih, w_hh, bias_ih, b_hh)

    def apply(self, M_input: npt.ArrayLike, hx: npt.ArrayLike) -> npt.NDArray:
        """Apply a single GRU step using analog computation.

        This implementation performs a single MVM operation to compute all gate
        components.

        The matrix structure matches the diagram:
        - Rows 0 to h-1:   [W_ir, W_hr] for reset gate
        - Rows h to 2h-1:  [W_iz, W_hz] for update gate
        - Rows 2h to 3h-1: [W_in, 0   ] for new gate input component
        - Rows 3h to 4h-1: [0,    W_hn] for new gate hidden component

        Args:
            M_input:
                1D or 2D numpy NDArray of shape (input_size) or
                (batch_size, input_size).
            hx:
                1D or 2D numpy NDArray of shape (hidden_size) or
                (batch_size, hidden_size) for previous hidden state.

        Returns:
            Raw pre-activation output of the GRU cell
        """
        M_input = xp.asarray(M_input)
        hx = xp.asarray(hx)

        # Handle unbatched inputs
        if M_input.ndim == 1:
            M_input = M_input.reshape(1, -1)
            hx = hx.reshape(1, -1)

        batch_size = M_input.shape[0]

        if M_input.shape[1] != self.input_size:
            raise ValueError(
                f"Expected input size {self.input_size}, got {M_input.shape[1]}"
            )

        if hx.shape != (batch_size, self.hidden_size):
            raise ValueError(
                f"Expected hidden shape ({batch_size}, {self.hidden_size}), "
                f"got {hx.shape}"
            )

        # Prepare input vector for MVM: [xt, ht-1]
        combined_input = xp.concatenate([M_input, hx], axis=1)

        # Add bias columns if using analog bias
        if self.bias_ih_rows or self.bias_hh_rows:
            bias_cols = xp.ones(
                (batch_size, self.bias_ih_rows + self.bias_hh_rows), dtype=xp.float32
            )
            combined_input = xp.concatenate([combined_input, bias_cols], axis=1)

        # Profile core inputs
        if self.params.simulation.analytics.profile_xbar_inputs:
            if self.xbar_inputs is None:
                self.xbar_inputs = combined_input.copy()
            else:
                self.xbar_inputs = xp.concatenate(
                    [self.xbar_inputs, combined_input], axis=0
                )

        # SINGLE MATRIX-VECTOR MULTIPLICATION
        gates = self.core.matmul(combined_input.T).T

        return gates
