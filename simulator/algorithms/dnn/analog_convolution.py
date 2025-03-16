#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""CrossSim implementation of convolutional neural network layers."""

from __future__ import annotations

from simulator import AnalogCore, CrossSimParameters
from simulator.backend import ComputeBackend
from .analog_layer import AnalogLayer

from math import prod

# Need numpy specifically for matrix formation as grouped convs use a build method
# incompatible with cupy
import numpy as np

import numpy.typing as npt

xp = ComputeBackend()


class AnalogConvolution(AnalogLayer):
    """CrossSim base class for N-dimensional convolutional layers.

    Base class contains all functionality for n-dimensional convolutions except the
    convolution operation. Implemented classes should implement at least one of
    `apply_convolution_matvec` and `apply_convolution_matmul` which differ based on
    whether convolutions are implemented with 1D MVMs or ND Matmuls.
    """

    def __init__(
        self,
        params: CrossSimParameters | list[CrossSimParameters],
        Nic: int,
        Noc: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        dilation: tuple[int, ...],
        groups: int,
        bias_rows: int,
    ) -> None:
        """Initializes a base convolutional layer wrapper around AnalogCore.

        Args:
            params:
                CrossSimParameters object or list of CrossSimParameters (for layers
                requiring multiple arrays) for the AnalogLinear layer. If a list, the
                length must match the number of arrays used within AnalogCore.
            Nic:
                Number of input channels, see torch.nn.Conv[N]D in_channels for
                detailed documentation.
            Noc:
                Number of output channels, see torch.nn.Conv[N]D out_channels for
                detailed documentation.
            kernel_size:
                N-element tuple of the kernel size along each convolution dimension. See
                torch.nn.Conv[N]D kernel_size for detailed documentation.
            stride:
                N-element tuple of the strides along each convolution dimension. See
                torch.nn.Conv[N]D stride for detailed documentation.
            dilation:
                N-element tuple of the dilation rate along each convolution dimension.
                See torch.nn.Conv[N]D dilation for detailed documentation. Only
                dilation rates of 1 on all dimensions are supported.
            groups:
                Number of groups used for the convolution. See torch.nn.Conv[N]D groups
                for detailed documentation.
            bias_rows:
                Integer indicating the number of rows to use to implement the bias
                within the array. 0 implies a digital bias.
        """
        # Keep a copy of the convolution parameters
        if not isinstance(params, list):
            self.params = params
        else:
            self.params = params[0].copy()

        # Set the convolution function we'll be using
        if self.params.simulation.fast_matmul:
            self.apply = self.apply_convolution_matmul
        else:
            self.apply = self.apply_convolution_matvec

        self.Nic = Nic
        self.Noc = Noc
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias_rows = bias_rows

        # Initialize the counter for input profiling. The data container will be
        # initialized in apply() where the # sliding windows is known
        self.last_input = 0

        self._synchronize_params()

        self.weight_mask = (
            slice(0, Noc, 1),
            slice(0, prod(kernel_size) * Nic, 1),
        )
        self.bias_mask = (
            slice(None, None, 1),
            slice(self.weight_mask[1].stop, self.weight_mask[1].stop + bias_rows, 1),
        )

        self.core = AnalogCore(
            np.zeros((Noc, prod(kernel_size) * Nic + bias_rows)),
            params,
            empty_matrix=True,
        )

    def set_matrix(self, matrix: npt.ArrayLike, verbose=False) -> None:
        """Programs a matrix into the layer's internal AnalogCore.

        AnalogConvolution needs a special set_matrix function to handle parallel matrix
        expansion to accelerate simulation when using matvec-based convolutions.

        See AnalogCore.set_matrix for details.

        Args:
            matrix: Numpy ndarray to be programmed into the array.
            verbose: Boolean flag to enable verbose print statements.
        """
        self.core.set_matrix(matrix, verbose=verbose)

        # Expand the cores if x_par or y_par > 1
        Ncopy = (
            self.params.simulation.convolution.x_par
            * self.params.simulation.convolution.y_par
        )
        if Ncopy > 1 and self.params.simulation.disable_fast_matmul:
            for j in range(self.core.num_cores_row):
                for k in range(self.core.num_cores_col):
                    self.core.cores[j][k].expand_matrix(Ncopy)

    def get_core_weights(self) -> tuple[npt.NDArray, npt.NDArray | None]:
        """Gets the weight and bias matrices with errors applied.

        This function only returns weight and bias values which can be derived from
        the stored array values. If the layer uses a digital bias the returned bias
        will be None.

        Returns:
            Tuple of numpy arrays, 2D for weights, 1D or None for bias.
        """
        matrix = self.get_matrix()
        if self.groups == 1:
            weight = matrix[self.weight_mask].reshape(
                (self.Noc, self.Nic, *self.kernel_size),
            )
        else:
            weight = xp.zeros(
                (self.Noc, self.Nic // self.groups, *self.kernel_size),
                dtype=matrix.dtype,
            )
            weights_per_out = prod(weight.shape[1:])
            outs_per_group = self.Noc // self.groups
            for i in range(self.Noc):
                group = i // outs_per_group
                weight[i] = matrix[
                    i,
                    group * weights_per_out : (group + 1) * weights_per_out,
                ].reshape(weight.shape[1:])

        if self.bias_rows > 0:
            bias = matrix[self.bias_mask].sum(1)
            return (weight, bias)
        else:
            return (weight, None)

    def form_matrix(
        self,
        weight: npt.ArrayLike,
        bias: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
        """Build a 2D weight matrix for the conv layer for programming into an array.

        Args:
            weight:
                [3,4,5]D numpy NDArray (or similar) with the weight matrix of the neural
                network for (1,2,3)D convolutions. Must have shape
                (Noc, Nic, kernel_size) where kernel_size has as many dimensions as the
                convolutional layer.
            bias:
                1D numpy NDArray (or similar) with the bias vector of the neural
                network. Ignored if the matrix does not use an analog_bias. Also
                ignored if an analog bias is specified but no bias is provided
                (most likely if the layer does not have a bias). Must have shape
                (Noc).

        Returns:
            2D numpy NDArray of the matrix.

        """
        w = np.asarray(weight)
        if bias is not None:
            b = np.asarray(bias)

        if w.shape != (self.Noc, self.Nic // self.groups, *self.kernel_size):
            raise ValueError(
                "Expected weight shape",
                (self.Noc, self.Nic // self.groups, *self.kernel_size),
                "got",
                w.shape,
            )

        # This reshape puts one output channel per row
        w_ = w.reshape(self.Noc, -1)

        # slicing with -0 doesn't work, so special case this
        if self.groups == 1 and not self.bias_rows:
            return w_

        # Otherwise (grouped conv or bias rows), we need to build the matrix
        # Specifically form on a CPU for compatibility with fill
        matrix = np.zeros(self.shape, dtype=w.dtype)

        if self.groups == 1:
            matrix[:, : -self.bias_rows] = w_
        else:
            submat_size = self.Nic // self.groups * prod(self.kernel_size)
            for i in range(self.Noc):
                offset = i // (self.Noc // self.groups)
                matrix[i, offset * submat_size : (offset + 1) * submat_size] = w_[i]

        if bias is not None and self.bias_rows:
            matrix[:, -self.bias_rows :] = np.repeat(
                b / self.bias_rows,
                self.bias_rows,
                axis=0,
            ).reshape(self.shape[0], self.bias_rows)

        return matrix

    def _synchronize_params(self) -> None:
        """Synchronize the params object from the internal parameters."""
        # Don't modify the following params:
        # x_par, y_par are pure inputs
        # Nwindows is derived by the caller

        if isinstance(self.params, CrossSimParameters):
            self.params.simulation.convolution.is_conv_core = True

        elif isinstance(self.params, list):
            for p in self.params:
                p.simulation.convolution.is_conv_core = True

    def _set_num_windows(self, Nwindows: int) -> None:
        """Set the derived parameter Nwindows on this object.

        This sets Nwindows for a layer's AnalogCore, and all subcores.
        """
        # Set Nwindows on the params for this object and its AnalogCore
        if isinstance(self.params, CrossSimParameters):
            self.params.simulation.convolution.Nwindows = Nwindows
            self.core.params.simulation.convolution.Nwindows = Nwindows
        elif isinstance(self.params, list):
            for p in self.params:
                p.simulation.convolution.Nwindows = Nwindows
            for p in self.core.params:
                p.simulation.convolution.Nwindows = Nwindows
        # Set Nwindows on the lower-level cores
        for c in range(self.core.num_cores_col):
            for r in range(self.core.num_cores_row):
                self.core.cores[r][c].params.simulation.convolution.Nwindows = Nwindows

    def __setitem__(self, key, value):
        """Forward setitem on the layer to the internal AnalogCore.

        Used primarily for PyTorch layer synchronization by setting values with masks.
        """
        # When setting the weights directly the input may not be reshaped yet
        # So reshape here. Should be safe as bias is 1D.
        if value.ndim > 2:
            value_ = value.reshape((value.shape[0], -1))
        else:
            value_ = value
        self.core.__setitem__(key, value_)

        # Then need to expand the cores if x_par or y_par > 1
        Ncopy = (
            self.params.simulation.convolution.x_par
            * self.params.simulation.convolution.y_par
        )
        if Ncopy > 1 and self.params.simulation.disable_fast_matmul:
            for j in range(self.core.num_cores_row):
                for k in range(self.core.num_cores_col):
                    self.core.cores[j][k].expand_matrix(Ncopy)


class AnalogConvolution1D(AnalogConvolution):
    """CrossSim implementation of for 1D convolutional layers."""

    def __init__(
        self,
        params: CrossSimParameters | list[CrossSimParameters],
        Nic: int,
        Noc: int,
        kernel_size: tuple[int,],
        stride: tuple[int,],
        dilation: tuple[int,],
        groups: int,
        bias_rows: int,
    ) -> None:
        """Initializes a 1D convolutional layer wrapper around AnalogCore.

        See AnalogConvolution for argument details. Note that kernel_size, stride, and
        dilation (values >1 not currently supported) must be 1 entry tuples.
        """
        super().__init__(
            params,
            Nic,
            Noc,
            kernel_size,
            stride,
            dilation,
            groups,
            bias_rows,
        )

    def apply(self, M_input: npt.ArrayLike) -> npt.NDArray:
        """Analog MVM based forward operation for a 1D convolutional layer.

        Will call either apply_convolution_matvec or apply_convolution_matmul based on
        params.simulation.fast_matmul. Matvec will always be used if the underlying
        simulation is not compatible with matmuls. See fast_matmul for more details.

        Args:
            M_input:
                A 2D or 3D of size (Nx, Nic) or (Nbatch, Nic, Nx). The
                trailing dimension must match self.Nic.

        Returns:
            A 2D or 3D of size (Noc, Nx_out) or (Nbatch, Noc, Nx_out). The
            number of dimensions of the returned matrix will match the
            number of dimensions of the input matrix.
        """
        return super().apply(M_input)

    def apply_convolution_matvec(self, M_input: npt.ArrayLike) -> npt.NDArray:
        """Applies a convolution operation using 1D MVMs.

        Uses the sliding window method. Each MVM returns returns the outputs
        for all output channels for a single window. The results are then
        re-constructed into an output matrix that follows the same format as
        the input matrix.

        See AnalogConvolution1D.apply for argument documentation.
        """
        M_input = xp.asarray(M_input)

        no_batch = False
        if M_input.ndim < 2:
            raise ValueError(f"Expected 2d or 3d input, got {M_input.ndim}d input")
        elif M_input.ndim == 2:
            no_batch = True
            M_input = M_input.reshape(1, *M_input.shape)

        Nbatch, Nic, Nx = M_input.shape

        if Nic != self.Nic:
            raise ValueError(f"Expected {self.Nic} channels, got {Nic}")

        (Kx,) = self.kernel_size
        Noc = self.Noc
        Nrows = self.core.shape[1]
        (strideX,) = self.stride
        x_par = self.params.simulation.convolution.x_par
        NrowsX = Kx * Nic  # number of rows per sliding window MVM

        # Number of sliding windows
        Nx_out = (M_input.shape[2] - Kx) // strideX + 1

        # Initialize data container and params for input and ADC profiling
        if self.last_input == 0:
            if self.params.simulation.analytics.profile_xbar_inputs:
                self.xbar_inputs = xp.zeros(
                    (self.params.simulation.analytics.ntest, Nic, Nx),
                )
            if self.params.simulation.analytics.profile_adc_inputs:
                self._set_num_windows(Nx_out)

        # Profile core inputs
        if self.params.simulation.analytics.profile_xbar_inputs:
            self.xbar_inputs[
                self.last_input : (self.last_input + Nbatch),
                ...,
            ] = M_input.copy()
        self.last_input += Nbatch

        M_outs = [None] * Nbatch
        for b in range(Nbatch):
            M_input_ = M_input[b]
            # Allocate memory for the output
            M_out = xp.empty((Noc, Nx_out), dtype=M_input_.dtype)

            for i in range(0, Nx_out, x_par):
                x_block = x_par if (Nx_out - i) >= x_par else Nx_out - i
                x_start = i * strideX
                if Kx == 1:
                    if not self.bias_rows:
                        Min_large = xp.zeros(
                            int(Nrows * x_par),
                            dtype=M_input_.dtype,
                        )
                        Min_block = (
                            M_input_[
                                :,
                                x_start : (x_start + strideX * x_block) : strideX,
                            ]
                            .transpose((1, 0))
                            .flatten()
                        )
                        Min_large[: len(Min_block)] = Min_block
                    else:
                        Min_large = xp.ones(
                            int(Nrows * x_par),
                            dtype=M_input_.dtype,
                        )
                        v_start, v_end = 0, NrowsX
                        for xxp in range(x_block):
                            Min_ij = M_input_[:, x_start]
                            Min_large[v_start:v_end] = Min_ij
                            v_start += Nrows
                            v_end += Nrows
                        x_start += strideX

                else:
                    Min_ij = xp.zeros(
                        (Nic * x_par, Kx),
                        dtype=M_input_.dtype,
                    )
                    x_end = x_start + Kx
                    v_start, v_end = 0, Nic

                    for xxp in range(x_block):
                        Min_ij[v_start:v_end, :] = M_input_[
                            :,
                            x_start:x_end,
                        ]
                        v_start += Nic
                        v_end += Nic
                        x_start += strideX
                        x_end += strideX

                    if self.bias_rows:
                        Min_large = xp.ones(
                            (x_block, Nrows),
                            dtype=M_input_.dtype,
                        )
                        Min_ij = Min_ij.reshape((x_block, NrowsX))
                        Min_large[:, : -self.bias_rows] = Min_ij
                    else:
                        Min_large = Min_ij

                M_out_p = self.core.mat_multivec(Min_large)
                # The line below is pure diabolical sorcery
                M_out[:, i : (i + x_block)] = M_out_p.reshape(
                    (Noc, x_par),
                    order="F",
                ).transpose((0, 1))[:, :x_block]
            M_outs[b] = M_out

        if no_batch:
            return xp.stack(M_outs).reshape((Noc, Nx_out))
        else:
            return xp.stack(M_outs).reshape((Nbatch, Noc, Nx_out))

    def apply_convolution_matmul(self, M_input: npt.ArrayLike) -> npt.NDArray:
        """Applies a convolution operation using 1D matrix multiplications.

        Uses matrix multiplication to compute all sliding windows and all
        images in batch in one shot. Generally significantly faster than
        matvec is used whenever the parameters are compatible with matmul.

        See AnalogConvolution1D.apply for argument documentation.
        """
        M_input = xp.asarray(M_input)

        no_batch = False
        if M_input.ndim < 2:
            raise ValueError(f"Expected 2d or 3d input, got {M_input.ndim}d input")
        elif M_input.ndim == 2:
            no_batch = True
            M_input = M_input.reshape(1, *M_input.shape)

        Nbatch, Nic, Nx = M_input.shape

        if Nic != self.Nic:
            raise ValueError(f"Expected {self.Nic} channels, got {Nic}")

        (Kx,) = self.kernel_size
        Noc = self.Noc
        (stride,) = self.stride
        NrowsX = Kx * Nic  # number of rows per sliding window MVM

        # Number of sliding windows
        Nx_out = (M_input.shape[2] - Kx) // stride + 1

        # Initialize data container and params for input and ADC profiling
        if self.last_input == 0:
            if self.params.simulation.analytics.profile_xbar_inputs:
                self.xbar_inputs = xp.zeros(
                    (self.params.simulation.analytics.ntest, Nic, Nx),
                )
            if self.params.simulation.analytics.profile_adc_inputs:
                self._set_num_windows(Nx_out)

        # Profile core inputs
        if self.params.simulation.analytics.profile_xbar_inputs:
            self.xbar_inputs[
                self.last_input : (self.last_input + Nbatch),
                ...,
            ] = M_input.copy()
        self.last_input += Nbatch

        if Kx == 1:
            M_input = M_input[:, :, ::stride]
            M_input = M_input.reshape((Nbatch, NrowsX, Nx_out))
        else:
            # Diabolical sorcery #2.5
            M_input = xp.lib.stride_tricks.as_strided(
                M_input,
                (M_input.shape[0], M_input.shape[1], Nx_out, Kx),
                (
                    M_input.strides[0],
                    M_input.strides[1],
                    M_input.strides[2] * stride,
                    M_input.strides[2],
                ),
            )
            M_input = M_input.transpose((0, 1, 3, 2))
            M_input = M_input.reshape((Nbatch, NrowsX, Nx_out))

        if self.bias_rows:
            M_input_all = xp.ones(
                (Nbatch, self.core.shape[1], Nx_out),
                dtype=xp.float32,
            )
            M_input_all[:, : -self.bias_rows, :] = M_input
            M_input = M_input_all

        M_out = self.core.matmat(M_input)

        if no_batch:
            return M_out.reshape((Noc, Nx_out))
        else:
            return M_out.reshape((Nbatch, Noc, Nx_out))


class AnalogConvolution2D(AnalogConvolution):
    """CrossSim implementation of for 2D convolutional layers."""

    def __init__(
        self,
        params: CrossSimParameters | list[CrossSimParameters],
        Nic: int,
        Noc: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
        bias_rows: int,
    ) -> None:
        """Initializes a 2D convolutional layer wrapper around AnalogCore.

        See AnalogConvolution for argument details. Note that kernel_size, stride, and
        dilation (values >1 not currently supported) must be 2 entry tuples.
        """
        super().__init__(
            params,
            Nic,
            Noc,
            kernel_size,
            stride,
            dilation,
            groups,
            bias_rows,
        )

    def apply(self, M_input: npt.ArrayLike) -> npt.NDArray:
        """Analog MVM based forward operation for a 2D convolutional layer.

        Will call either apply_convolution_matvec or apply_convolution_matmul based on
        params.simulation.fast_matmul. Matvec will always be used if the underlying
        simulation is not compatible with matmuls. See fast_matmul for more details.

        Args:
            M_input:
                A 3D or 4D of size (Nic, Nx, Ny) or (Nbatch, Nic, Nx, Ny). The
                trailing dimension must match self.Nic.

        Returns:
            A 3D or 4D of size (Noc, Nx_out, Ny_out) or
            (Nbatch, Noc, Nx_out, Ny_out). The number of dimensions of the
            returned matrix will match the number of dimensions of the input
            matrix.
        """
        return super().apply(M_input)

    def apply_convolution_matvec(self, M_input: npt.ArrayLike) -> npt.NDArray:
        """Applies a convolution operation using 2D MVMs.

        Uses the sliding window method. Each MVM returns returns the outputs
        for all output channels for a single window. The results are then
        re-constructed into an output matrix that follows the same format as
        the input matrix.

        See AnalogConvolution2D.apply for argument documentation.
        """
        M_input = xp.asarray(M_input)

        no_batch = False
        if M_input.ndim < 3:
            raise ValueError(f"Expected 3d or 4d input, got {M_input.ndim}d input")
        elif M_input.ndim == 3:
            no_batch = True
            M_input = M_input.reshape(1, *M_input.shape)

        Nbatch, Nic, Nx, Ny = M_input.shape

        if Nic != self.Nic:
            raise ValueError(f"Expected {self.Nic} channels, got {Nic}")

        Kx, Ky = self.kernel_size
        Noc = self.Noc
        Nrows = self.core.shape[1]
        strideX, strideY = self.stride
        x_par = self.params.simulation.convolution.x_par
        y_par = self.params.simulation.convolution.y_par
        NrowsX = Kx * Ky * Nic  # number of rows per sliding window MVM

        # Number of sliding windows
        Nx_out, Ny_out = (
            (M_input.shape[2] - Kx) // strideX + 1,
            (M_input.shape[3] - Ky) // strideY + 1,
        )

        # Initialize data container and params for input and ADC profiling
        if self.last_input == 0:
            if self.params.simulation.analytics.profile_xbar_inputs:
                self.xbar_inputs = xp.zeros(
                    (self.params.simulation.analytics.ntest, Nic, Nx, Ny),
                )
            if self.params.simulation.analytics.profile_adc_inputs:
                self._set_num_windows(Nx_out * Ny_out)

        # Profile core inputs
        if self.params.simulation.analytics.profile_xbar_inputs:
            self.xbar_inputs[
                self.last_input : (self.last_input + Nbatch),
                ...,
            ] = M_input.copy()
        self.last_input += Nbatch

        M_outs = [None] * Nbatch
        for b in range(Nbatch):
            M_input_ = M_input[b]
            # Allocate memory for the output
            M_out = xp.empty((Noc, Nx_out, Ny_out), dtype=M_input_.dtype)

            for i in range(0, Nx_out, x_par):
                x_block = x_par if (Nx_out - i) >= x_par else Nx_out - i
                for j in range(0, Ny_out, y_par):
                    y_block = y_par if (Ny_out - j) >= y_par else Ny_out - j
                    x_start = i * strideX
                    y_start0 = j * strideY
                    if Kx == 1 and Ky == 1:
                        if not self.bias_rows:
                            Min_large = xp.zeros(
                                (Nrows * x_par * y_par),
                                dtype=M_input_.dtype,
                            )
                            Min_block = (
                                M_input_[
                                    :,
                                    x_start : (x_start + strideX * x_block) : strideX,
                                    y_start0 : (y_start0 + strideY * y_block) : strideY,
                                ]
                                .transpose((1, 2, 0))
                                .flatten()
                            )
                            Min_large[: len(Min_block)] = Min_block
                        else:
                            Min_large = xp.ones(
                                int(Nrows * x_par * y_par),
                                dtype=M_input_.dtype,
                            )
                            v_start, v_end = 0, NrowsX
                            for xxp in range(x_block):
                                y_start = y_start0
                                for yyp in range(y_block):
                                    Min_ij = M_input_[:, x_start, y_start]
                                    Min_large[v_start:v_end] = Min_ij
                                    y_start += strideY
                                    v_start += Nrows
                                    v_end += Nrows
                                x_start += strideX
                    else:
                        Min_ij = xp.zeros(
                            (Nic * x_par * y_par, Kx, Ky),
                            dtype=M_input_.dtype,
                        )
                        x_end = x_start + Kx
                        v_start, v_end = 0, Nic

                        for xxp in range(x_block):
                            y_start = y_start0
                            y_end = y_start + Ky
                            for yyp in range(y_block):
                                Min_ij[v_start:v_end, :, :] = M_input_[
                                    :,
                                    x_start:x_end,
                                    y_start:y_end,
                                ]
                                y_start += strideY
                                y_end += strideY
                                v_start += Nic
                                v_end += Nic
                            x_start += strideX
                            x_end += strideX

                        if self.bias_rows:
                            Min_large = xp.ones(
                                (x_block * y_block, Nrows),
                                dtype=M_input_.dtype,
                            )
                            Min_ij = Min_ij.reshape((x_block * y_block, NrowsX))
                            Min_large[:, : -self.bias_rows] = Min_ij
                        else:
                            Min_large = Min_ij

                    M_out_p = self.core.mat_multivec(Min_large)
                    # The line below is pure diabolical sorcery
                    M_out[:, i : (i + x_block), j : (j + y_block)] = M_out_p.reshape(
                        (Noc, y_par, x_par),
                        order="F",
                    ).transpose((0, 2, 1))[:, :x_block, :y_block]
            M_outs[b] = M_out

        if no_batch:
            return xp.stack(M_outs).reshape((Noc, Nx_out, Ny_out))
        else:
            return xp.stack(M_outs).reshape((Nbatch, Noc, Nx_out, Ny_out))

    def apply_convolution_matmul(self, M_input: npt.ArrayLike) -> npt.NDArray:
        """Applies a convolution operation using 2D matrix multiplications.

        Uses matrix multiplication to compute all sliding windows and all
        images in batch in one shot. Generally significantly faster than
        matvec is used whenever the parameters are compatible with matmul.

        See AnalogConvolution2D.apply for argument documentation.
        """
        M_input = xp.asarray(M_input)

        dim3 = False
        if M_input.ndim < 3:
            raise ValueError(f"Expected 3d or 4d input, got {M_input.ndim}d input")
        elif M_input.ndim == 3:
            dim3 = True
            M_input = M_input.reshape(1, *M_input.shape)

        Nbatch, Nic, Nx, Ny = M_input.shape

        if Nic != self.Nic:
            raise ValueError(f"Expected {self.Nic} channels, got {Nic}")

        Kx, Ky = self.kernel_size
        Noc = self.Noc
        strideX, strideY = self.stride
        NrowsX = Kx * Ky * Nic  # number of rows per sliding window MVM

        # Number of sliding windows
        Nx_out, Ny_out = (
            (M_input.shape[2] - Kx) // strideX + 1,
            (M_input.shape[3] - Ky) // strideY + 1,
        )

        # Initialize data container and params for input and ADC profiling
        if self.last_input == 0:
            if self.params.simulation.analytics.profile_xbar_inputs:
                self.xbar_inputs = xp.zeros(
                    (self.params.simulation.analytics.ntest, Nic, Nx, Ny),
                )
            if self.params.simulation.analytics.profile_adc_inputs:
                self._set_num_windows(Nx_out * Ny_out)

        # Profile core inputs
        if self.params.simulation.analytics.profile_xbar_inputs:
            self.xbar_inputs[
                self.last_input : (self.last_input + Nbatch),
                ...,
            ] = M_input.copy()
        self.last_input += Nbatch

        if Kx == 1 and Ky == 1:
            M_input = M_input[:, :, ::strideX, ::strideY]
            M_input = M_input.reshape((Nbatch, NrowsX, Nx_out * Ny_out))
        else:
            # Diabolical sorcery #2.5
            M_input = xp.lib.stride_tricks.as_strided(
                M_input,
                (M_input.shape[0], M_input.shape[1], Nx_out, Ny_out, Kx, Ky),
                (
                    M_input.strides[0],
                    M_input.strides[1],
                    M_input.strides[2] * strideX,
                    M_input.strides[3] * strideY,
                    M_input.strides[2],
                    M_input.strides[3],
                ),
            )
            M_input = M_input.transpose((0, 1, 4, 5, 2, 3))
            M_input = M_input.reshape((Nbatch, NrowsX, Nx_out * Ny_out))

        if self.bias_rows:
            M_input_all = xp.ones(
                (Nbatch, self.core.shape[1], Nx_out * Ny_out),
                dtype=xp.float32,
            )
            M_input_all[:, : -self.bias_rows, :] = M_input
            M_input = M_input_all

        M_out = self.core.matmat(M_input)

        if dim3:
            return M_out.reshape((Noc, Nx_out, Ny_out))
        else:
            return M_out.reshape((Nbatch, Noc, Nx_out, Ny_out))


class AnalogConvolution3D(AnalogConvolution):
    """CrossSim implementation of for 3D convolutional layers."""

    def __init__(
        self,
        params: CrossSimParameters | list[CrossSimParameters],
        Nic: int,
        Noc: int,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
        dilation: tuple[int, int, int],
        groups: int,
        bias_rows: int,
    ) -> None:
        """Initializes a 3D convolutional layer wrapper around AnalogCore.

        See AnalogConvolution for argument details. Note that kernel_size, stride, and
        dilation (values >1 not currently supported) must be 3 entry tuples.

        Matvec-based operations are not supported for 3D networks.
        """
        super().__init__(
            params,
            Nic,
            Noc,
            kernel_size,
            stride,
            dilation,
            groups,
            bias_rows,
        )

    def apply(self, M_input: npt.ArrayLike) -> npt.NDArray:
        """Analog MVM based forward operation for a 3D convolutional layer.

        Will call either apply_convolution_matvec or apply_convolution_matmul based on
        params.simulation.fast_matmul. Matvec will always be used if the underlying
        simulation is not compatible with matmuls. See fast_matmul for more details.

        Args:
            M_input:
                A 4D or 5D of size (Nic, Nx, Ny, Nz) or (Nbatch, Nic, Nx, Ny, Nz). The
                trailing dimension must match self.Nic.

        Returns:
            A 4D or 5D of size (Noc, Nx_out, Ny_out, Nz_out) or
            (Nbatch, Noc, Nx_out, Ny_out, Nz_out). The number of dimensions of the
            returned matrix will match the number of dimensions of the input
            matrix.

        """
        return super().apply(M_input)

    def apply_convolution_matvec(self, M_input: npt.ArrayLike) -> npt.NDArray:
        """Applies a convolution operation using 3D MVMs.

        Uses the sliding window method. Each MVM returns returns the outputs
        for all output channels for a single window. The results are then
        re-constructed into an output matrix that follows the same format as
        the input matrix.

        See AnalogConvolution2D.apply for argument documentation.
        """
        raise NotImplementedError(
            "Matvec computation for 3D Convolutions is not" " currently implemented.",
        )

    def apply_convolution_matmul(self, M_input: npt.ArrayLike) -> npt.NDArray:
        """Applies a convolution operation using 3D matrix multiplications.

        Uses matrix multiplication to compute all sliding windows and all
        images in batch in one shot. Generally significantly faster than
        matvec is used whenever the parameters are compatible with matmul.

        See AnalogConvolution3D.apply for argument documentation.
        """
        M_input = xp.asarray(M_input)

        no_batch = False
        if M_input.ndim < 4:
            raise ValueError(f"Expected 4 or 5d input, got {M_input.ndim}d input")
        elif M_input.ndim == 4:
            no_batch = True
            M_input = M_input.reshape(1, *M_input.shape)

        Nbatch, Nic, Nx, Ny, Nz = M_input.shape

        if Nic != self.Nic:
            raise ValueError(f"Expected {self.Nic} channels, got {Nic}")

        Kx, Ky, Kz = self.kernel_size
        Noc = self.Noc
        strideX, strideY, strideZ = self.stride
        NrowsX = Kx * Ky * Kz * Nic  # number of rows per sliding window MVM

        # Number of sliding windows
        Nx_out, Ny_out, Nz_out = (
            (M_input.shape[2] - Kx) // strideX + 1,
            (M_input.shape[3] - Ky) // strideY + 1,
            (M_input.shape[4] - Kz) // strideZ + 1,
        )

        # Initialize data container and params for input and ADC profiling
        if self.last_input == 0:
            if self.params.simulation.analytics.profile_xbar_inputs:
                self.xbar_inputs = xp.zeros(
                    (self.params.simulation.analytics.ntest, Nic, Nx, Ny, Nz),
                )
            if self.params.simulation.analytics.profile_adc_inputs:
                self._set_num_windows(Nx_out * Ny_out * Nz_out)

        # Profile core inputs
        if self.params.simulation.analytics.profile_xbar_inputs:
            self.xbar_inputs[
                self.last_input : (self.last_input + Nbatch),
                ...,
            ] = M_input.copy()
        self.last_input += Nbatch

        if Kx == 1 and Ky == 1 and Kz == 1:
            M_input = M_input[:, :, ::strideX, ::strideY, ::strideZ]
            M_input = M_input.reshape((Nbatch, NrowsX, Nx_out * Ny_out * Nz_out))
        else:
            # Diabolical sorcery #2.5
            M_input = xp.lib.stride_tricks.as_strided(
                M_input,
                (
                    M_input.shape[0],
                    M_input.shape[1],
                    Nx_out,
                    Ny_out,
                    Nz_out,
                    Kx,
                    Ky,
                    Kz,
                ),
                (
                    M_input.strides[0],
                    M_input.strides[1],
                    M_input.strides[2] * strideX,
                    M_input.strides[3] * strideY,
                    M_input.strides[4] * strideZ,
                    M_input.strides[2],
                    M_input.strides[3],
                    M_input.strides[4],
                ),
            )
            M_input = M_input.transpose((0, 1, 5, 6, 7, 2, 3, 4))
            M_input = M_input.reshape((Nbatch, NrowsX, Nx_out * Ny_out * Nz_out))

        if self.bias_rows:
            M_input_all = xp.ones(
                (Nbatch, self.core.shape[1], Nx_out * Ny_out * Nz_out),
                dtype=xp.float32,
            )
            M_input_all[:, : -self.bias_rows, :] = M_input
            M_input = M_input_all

        M_out = self.core.matmat(M_input)

        if no_batch:
            return M_out.reshape((Noc, Nx_out, Ny_out, Nz_out))
        else:
            return M_out.reshape((Nbatch, Noc, Nx_out, Ny_out, Nz_out))
