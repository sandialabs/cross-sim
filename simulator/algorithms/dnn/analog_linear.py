#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""CrossSim implementation of linear (dense) neural network layers."""

from __future__ import annotations

import numpy as np
from simulator import AnalogCore, CrossSimParameters
from simulator.backend import ComputeBackend
from .analog_layer import AnalogLayer

import numpy.typing as npt

xp = ComputeBackend()


class AnalogLinear(AnalogLayer):
    """CrossSim implementation of for linear (dense) layers."""

    def __init__(
        self,
        params: CrossSimParameters | list[CrossSimParameters],
        in_features: int,
        out_features: int,
        bias_rows: int,
    ) -> None:
        """Initializes a Linear layer wrapper around AnalogCore.

        Args:
            params:
                CrossSimParameters object or list of CrossSimParameters (for layers
                requiring multiple arrays) for the AnalogLinear layer. If a list, the
                length must match the number of arrays used within AnalogCore.
            in_features:
                Number of input features, see torch.nn.Linear in_features for detailed
                documentation.
            out_features:
                Number of output features, see torch.nn.Linear in_features for detailed
                documentation.
            bias_rows:
                Integer indicating the number of rows to use to implement the bias
                within the array. 0 implies a digital bias.
        """
        if not isinstance(params, list):
            self.params = params
        else:
            self.params = params[0].copy()

        self.in_features = in_features
        self.out_features = out_features
        self.bias_rows = bias_rows

        # Initialize the counter for input profiling.
        self.last_input = 0

        self.weight_mask = (slice(0, out_features, 1), slice(0, in_features, 1))
        self.bias_mask = (
            slice(None, None, 1),
            slice(in_features, in_features + bias_rows, 1),
        )

        self.core = AnalogCore(
            np.zeros((out_features, in_features + bias_rows)),
            params,
            empty_matrix=True,
        )

    def form_matrix(
        self,
        weight: npt.ArrayLike,
        bias: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
        """Build a 2D weight matrix for the linear layer for programming into an array.

        Args:
            weight:
                2D numpy NDArray (or similar) with the weight matrix of the neural
                network. Must have shape (out_features, in_features).
            bias:
                1D numpy NDArray (or similar) with the bias vector of the neural
                network. Ignored if the matrix does not use an analog_bias. Also
                ignored if an analog bias is specified but no bias is provided
                (most likely if the layer does not have a bias). Must have shape
                (out_features).

        Returns:
            2D numpy NDArray of the matrix.
        """
        w = np.asarray(weight)
        if bias is not None:
            b = np.asarray(bias)

        if w.shape != (self.out_features, self.in_features):
            raise ValueError(
                "Expected weight shape",
                (self.out_features, self.in_features),
                "got",
                w.shape,
            )

        # slicing with -0 doesn't work, so special case this
        if not self.bias_rows:
            return w

        matrix = np.zeros(self.shape, dtype=w.dtype)
        matrix[:, : -self.bias_rows] = w
        if bias is not None and self.bias_rows:
            matrix[:, -self.bias_rows :] = np.repeat(
                b / self.bias_rows,
                self.bias_rows,
                axis=0,
            ).reshape(self.shape[0], self.bias_rows)

        return matrix

    def get_core_weights(self) -> tuple[npt.NDArray, npt.NDArray | None]:
        """Gets the weight and bias matrices with errors applied.

        This function only returns weight and bias values which can be derived from
        the stored array values. If the layer uses a digital bias the returned bias
        will be None.

        Returns:
            Tuple of numpy arrays, 2D for weights, 1D or None for bias.
        """
        matrix = self.get_matrix()
        weight = matrix[self.weight_mask]
        if self.bias_rows > 0:
            bias = matrix[self.bias_mask].sum(1)
            return (weight, bias)
        else:
            return (weight, None)

    def apply(self, M_input: npt.ArrayLike) -> npt.NDArray:
        """Analog MVM based forward operation for a Linear layer.

        This function assumes that there is at-most 1 leading dimension (a single batch
        dimension). Layer implementations are responsible for reshaping inputs and
        outputs for this function as needed by the neural network framework.

        Args:
            M_input:
                1D or 2D Numpy NDArray (or similar input for the forward operation.

        Returns:
            1D or 2D Numpy NDArray result of the operation. The number of dimensions in
            the return value will match the number of dimensions of the input.
        """
        M_input = xp.asarray(M_input)

        no_batch = False
        if M_input.ndim == 1:
            no_batch = True
            M_input = M_input.reshape(1, *M_input.shape)

        Nbatch, Ninputs = M_input.shape

        if Ninputs != self.in_features:
            raise ValueError(f"Expected {self.in_features} inputs, got {Ninputs}")

        # Profile core inputs
        if self.params.simulation.analytics.profile_xbar_inputs:
            if self.last_input == 0:
                self.xbar_inputs = xp.zeros(
                    (self.params.simulation.analytics.ntest, Ninputs),
                )
            self.xbar_inputs[
                self.last_input : (self.last_input + Nbatch),
                ...,
            ] = M_input.copy()
            self.last_input += Nbatch

        if self.bias_rows:
            M_input_all = xp.ones((Nbatch, self.core.shape[1]), dtype=xp.float32)
            M_input_all[:, : -self.bias_rows] = M_input
            M_input = M_input_all

        # To keep AnalogLinear consistent with AnalogConvolution we adopt the convention
        # that the first dimension of the matrix is the number of outputs. This means
        # all neural network operations will use the MVM rather than VMM code path.
        # We use a double transpose on the input and output to remedy the
        # inconsistency. Notably, this should be fast in simulation due to how
        # numpy/cupy implement transpose and in hardware this just corresponds to
        # how the hardware indexes the data array rather than a real transpose
        # operation that needs to be implemented in software.
        M_out = self.core.matmul(M_input.T).T

        if no_batch:
            M_out = M_out.reshape(*M_out.shape[1:])
        return M_out
