#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""CrossSim implementation of RNN (Recurrent Neural Network) cell layers."""

from __future__ import annotations

import numpy as np
from simulator import AnalogCore, CrossSimParameters
from simulator.backend import ComputeBackend
from .analog_layer import AnalogLayer

import numpy.typing as npt

xp = ComputeBackend()


class AnalogRNNCell(AnalogLayer):
    """CrossSim implementation for Recurrent Neural Network Cell layers."""

    def __init__(
        self,
        params: CrossSimParameters | list[CrossSimParameters],
        input_size: int,
        hidden_size: int,
        nonlinearity: str,
        bias_ih_rows: int,
        bias_hh_rows: int,
    ) -> None:
        """Initializes an RNNCell layer wrapper around AnalogCore.

        Args:
            params:
                CrossSimParameters object or list of CrossSimParameters (for
                layers requiring multiple arrays) for the AnalogRNNCell layer.
                If a list, the length must match the number of arrays used
                within AnalogCore.
            input_size:
                Number of input features, see torch.nn.RNNCell input_size for
                detailed documentation.
            hidden_size:
                Number of hidden features, see torch.nn.RNNCell hidden_size for
                detailed documentation.
            nonlinearity:
                The non-linearity to use. Can be either 'tanh' or 'relu'.
                Defaults to 'tanh'.
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

        # For RNNCell, we need to handle input-to-hidden and hidden-to-hidden
        # weights. The matrix will contain both weight_ih and weight_hh
        # concatenated
        total_input_size = input_size + hidden_size

        total_cols = total_input_size + self.bias_ih_rows + self.bias_hh_rows

        # Define masks for accessing different parts of the matrix
        self.weight_ih_mask = (slice(0, hidden_size, 1), slice(0, input_size, 1))
        self.weight_hh_mask = (
            slice(0, hidden_size, 1),
            slice(input_size, total_input_size, 1),
        )

        self.bias_ih_mask = (
            slice(None, None, 1),
            slice(total_input_size, total_input_size + self.bias_ih_rows, 1),
        )

        self.bias_hh_mask = (
            slice(None, None, 1),
            slice(
                total_input_size + self.bias_ih_rows,
                total_input_size + self.bias_ih_rows + self.bias_hh_rows,
                1,
            ),
        )

        self.core = AnalogCore(
            np.zeros((hidden_size, total_cols)),
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
        """Build a 2D weight matrix of the for programming into an array.

        Args:
            weight_ih:
                2D numpy NDArray (or similar) with the input-to-hidden weight
                matrix. Must have shape (hidden_size, input_size).
            weight_hh:
                2D numpy NDArray (or similar) with the hidden-to-hidden weight
                matrix. Must have shape (hidden_size, hidden_size).
            bias_ih:
                1D numpy NDArray (or similar) with the input-to-hidden bias
                vector. Ignored if the matrix does not use an analog bias. Must
                have shape (hidden_size).
            bias_hh:
                1D numpy NDArray (or similar) with the hidden-to-hidden bias
                vector. Ignored if the matrix does not use an analog bias.
                Must have shape (hidden_size).

        Returns:
            2D numpy NDArray of the matrix.
        """
        w_ih = np.asarray(weight_ih)
        w_hh = np.asarray(weight_hh)

        if bias_ih is not None:
            b_ih = np.asarray(bias_ih)
        if bias_hh is not None:
            b_hh = np.asarray(bias_hh)

        if w_ih.shape != (self.hidden_size, self.input_size):
            raise ValueError(
                f"Expected weight_ih shape ({self.hidden_size}, {self.input_size}), "
                f"got {w_ih.shape}"
            )

        if w_hh.shape != (self.hidden_size, self.hidden_size):
            raise ValueError(
                f"Expected weight_hh shape ({self.hidden_size}, {self.hidden_size}), "
                f"got {w_hh.shape}"
            )

        # Concatenate input-to-hidden and hidden-to-hidden weights
        w_combined = np.concatenate([w_ih, w_hh], axis=1)

        # If no bias rows, return combined weights
        if not (self.bias_ih_rows or self.bias_hh_rows):
            return w_combined

        # Create matrix with space for bias
        total_cols = (
            self.input_size + self.hidden_size + self.bias_ih_rows + self.bias_hh_rows
        )
        matrix = np.zeros((self.hidden_size, total_cols), dtype=w_ih.dtype)

        # Set weight portions
        matrix[:, : self.input_size + self.hidden_size] = w_combined

        # Set bias portions if provided
        if bias_ih is not None and self.bias_ih_rows:
            bias_start = self.input_size + self.hidden_size
            matrix[:, bias_start : bias_start + self.bias_ih_rows] = np.repeat(
                b_ih / self.bias_ih_rows,
                self.bias_ih_rows,
                axis=0,
            ).reshape(self.hidden_size, self.bias_ih_rows)

        if bias_hh is not None and self.bias_hh_rows:
            bias_start = self.input_size + self.hidden_size + self.bias_ih_rows
            matrix[:, bias_start : bias_start + self.bias_hh_rows] = np.repeat(
                b_hh / self.bias_hh_rows,
                self.bias_hh_rows,
                axis=0,
            ).reshape(self.hidden_size, self.bias_hh_rows)

        return matrix

    def get_core_weights(
        self,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray | None, npt.NDArray | None]:
        """Gets the weight and bias matrices with errors applied.

        Returns:
            Tuple of numpy arrays: (weight_ih, weight_hh, bias_ih, bias_hh).
            Bias arrays are None if digital bias is used.
        """
        matrix = self.get_matrix()

        weight_ih = matrix[self.weight_ih_mask]
        weight_hh = matrix[self.weight_hh_mask]
        bias_ih = None
        bias_hh = None

        if self.bias_ih_rows > 0:
            bias_ih = matrix[self.bias_ih_mask].sum(1)
        if self.bias_hh_rows > 0:
            bias_hh = matrix[self.bias_hh_mask].sum(1)

        return (weight_ih, weight_hh, bias_ih, bias_hh)

    def apply(self, M_input: npt.ArrayLike, hx: npt.ArrayLike) -> npt.NDArray:
        """Apply a single RNN step using analog computation.

        Args:
            M_input:
                1D or 2D numpy NDArray of shape (input_size) or
                (batch_size, input_size).
            hx:
                1D or 2D numpy NDArray of shape (hidden_size) or
                (batch_size, hidden_size) for previous hidden state.

        Returns:
            Raw pre-activation output as numpy NDArray of shape
            (batch_size, hidden_size).
            Note: Activation function is NOT applied, this should be done by the
            caller.
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
                f"Expected M_input size {self.input_size}, got {M_input.shape[1]}"
            )

        if hx.shape != (batch_size, self.hidden_size):
            raise ValueError(
                f"Expected hidden shape ({batch_size}, {self.hidden_size}), "
                f"got {hx.shape}"
            )

        # Concatenate M_input and hidden state: [M_input, hx]
        combined_input = xp.concatenate([M_input, hx], axis=1)

        # Add bias columns if using analog bias: [M_input, hx, bias_ones]
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

        # Apply analog matrix-vector multiplication
        # Core matrix shape: (hidden_size,
        #   input_size + hidden_size + 2 * bias_rows)
        # Combined_input shape: (batch_size,
        #   input_size + hidden_size + 2 * bias_rows)
        # Core @ combined_input.T shape:(hidden_size, batch_size)
        # We then transpose to get (batch_size, hidden_size)
        output = self.core.matmul(combined_input.T).T

        return output
