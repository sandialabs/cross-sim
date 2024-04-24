#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from __future__ import annotations

import numpy as np
from ...cores.analog_core import AnalogCore, CrossSimParameters
from ...backend import ComputeBackend
from math import prod

import numpy.typing as npt

xp = ComputeBackend()


class AnalogConvolution:
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
        # Keep a copy of the convolution parameters
        if not isinstance(params, list):
            self.params = params
        else:
            self.params = params[0].copy()

        # Set the convolution function we'll be using
        if (
            self.params.simulation.fast_matmul
            and self.params.simulation.convolution.conv_matmul
        ):
            self.apply_convolution = self.apply_convolution_matmul
        else:
            self.apply_convolution = self.apply_convolution_matvec

        self.Nic = Nic
        self.Noc = Noc
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias_rows = bias_rows

        self._synchronize_params()

        self.core = AnalogCore(
            np.zeros((Noc, prod(kernel_size) * Nic + bias_rows)),
            params,
            empty_matrix=True,
        )

    def set_matrix(self, matrix: npt.ArrayLike, verbose=False) -> None:
        """ """
        self.core.set_matrix(matrix, verbose=verbose)

    def get_matrix(self) -> npt.NDArray:
        """Read the internal matrix held by this core."""
        return self.core.get_matrix()

    def form_matrix(
        self,
        weight: npt.ArrayLike,
        bias: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
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
        matrix = xp.zeros(self.shape, dtype=w.dtype)

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
        # x_par, y_par, weight_reoder, and conv_matmul are pure inputs
        # Nwindows is derived by the caller

        if isinstance(self.params, CrossSimParameters):
            self.params.simulation.convolution.is_conv_core = True
            self.params.simulation.convolution.Kx = self.kernel_size[0]
            self.params.simulation.convolution.Ky = self.kernel_size[1]
            # TODO: downstream changes from different X/Y strides?
            self.params.simulation.convolution.stride = self.stride[0]
            self.params.simulation.convolution.Noc = self.Noc
            self.params.simulation.convolution.Nic = self.Nic
            self.params.simulation.convolution.bias_row = self.bias_rows > 0

        elif isinstance(self.params, list):
            for p in self.params:
                p.simulation.convolution.is_conv_core = True
                p.simulation.convolution.Kx = self.kernel_size[0]
                p.simulation.convolution.Ky = self.kernel_size[1]
                p.simulation.convolution.stride = self.stride[0]
                p.simulation.convolution.Noc = self.Noc
                p.simulation.convolution.Nic = self.Nic
                p.simulation.convolution.bias_row = self.bias_rows > 0

    # A few properties to pass through from AnalogCore so this can be treated
    # like an AnalogCore for manipulation and debugging purposes.
    @property
    def max(self):
        return self.core.max

    @property
    def min(self):
        return self.core.min

    @property
    def shape(self):
        return self.core.shape

    @property
    def Ncores(self):
        return self.core.Ncores

    @property
    def cores(self):
        return self.core.cores

    def __setitem__(self, key, value):
        # When setting the weights directly the input may not be reshaped yet
        # So reshape here. Should be safe as bias is 1D.
        if value.ndim > 2:
            value_ = value.reshape((value.shape[0], -1))
        else:
            value_ = value
        self.core.__setitem__(key, value_)


class AnalogConvolution2D(AnalogConvolution):
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

    def apply_convolution_matvec(self, M_input: npt.ArrayLike) -> npt.NDArray:
        """Applies a convolution operation using 1D MVMs.

        Uses the sliding window method. Each MVM returns returns the outputs
        for all output channels for a single window. The results are then
        re-constructed into an output matrix that follows the same format as
        the input matrix.

        Args:
            M_input:
                A 3D or 4D of size (Nx, Ny, Nic) or (Nbatch, Nx, Ny, Nic). The
                trailing dimension must match self.Nic.

        Returns:
            A 3D or 4D of size (Nx_out, Ny_out, Noc) or
            (Nbatch, Nx_out, Ny_out, Noc). The number of dimensions of the
            returned matrix will match the number of dimensions of the input
            matrix.
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

        Kx = self.kernel_size[0]
        Ky = self.kernel_size[1]
        Noc = self.Noc
        Nrows = self.core.shape[1]
        strideX = self.stride[0]
        strideY = self.stride[1]
        x_par = self.params.simulation.convolution.x_par
        y_par = self.params.simulation.convolution.y_par
        weight_reorder = self.params.simulation.convolution.weight_reorder
        NrowsX = Kx * Ky * Nic  # number of rows per sliding window MVM

        # Number of sliding windows
        Nx_out, Ny_out = (
            (M_input.shape[2] - Kx) // strideX + 1,
            (M_input.shape[3] - Ky) // strideY + 1,
        )

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
                            Min_block = M_input_[
                                :,
                                x_start : (x_start + strideX * x_par) : strideX,
                                y_start0 : (y_start0 + strideY * y_par) : strideY,
                            ]
                            Min_large = Min_block.transpose((1, 2, 0)).flatten()
                        else:
                            Min_large = xp.ones(
                                int(Nrows * x_par * y_par),
                                dtype=M_input_.dtype,
                            )
                            v_start, v_end = 0, NrowsX
                            for xxp in range(x_par):
                                y_start = y_start0
                                for yyp in range(y_par):
                                    Min_ij = M_input_[:, x_start, y_start]
                                    Min_large[v_start:v_end] = Min_ij
                                    y_start += strideY
                                    v_start += Nrows
                                    v_end += Nrows
                                x_start += strideX

                    else:
                        if weight_reorder:
                            x_end = x_start + Kx + strideX * (x_par - 1)
                            y_end = y_start0 + Ky + strideY * (y_par - 1)
                            Min_large = M_input_[:, x_start:x_end, y_start0:y_end]

                        else:
                            Min_ij = xp.zeros(
                                (Nic * x_par * y_par, Kx, Ky),
                                dtype=M_input_.dtype,
                            )
                            x_end = x_start + Kx
                            v_start, v_end = 0, Nic

                            for xxp in range(x_par):
                                y_start = y_start0
                                y_end = y_start + Ky
                                for yyp in range(y_par):
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
                                    (x_par * y_par, Nrows),
                                    dtype=M_input_.dtype,
                                )
                                Min_ij = Min_ij.reshape((x_par * y_par, NrowsX))
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

        if dim3:
            return xp.stack(M_outs).reshape((Noc, Nx_out, Ny_out))
        else:
            return xp.stack(M_outs).reshape((Nbatch, Noc, Nx_out, Ny_out))

    def apply_convolution_matmul(self, M_input: npt.ArrayLike) -> npt.NDArray:
        """Applies a convolution operation using ND matrix multiplications.

        Uses matrix multiplication to compute all sliding windows and all
        images in batch in one shot. Generally significantly faster than
        matvec is used whenever the parameters are compabible with matmul.

        Args:
            M_input:
                A 3D or 4D of size (Nx, Ny, Nic) or (Nbatch, Nx, Ny, Nic). The
                trailing dimension must match self.Nic.

        Returns:
            A 3D or 4D of size (Nx_out, Ny_out, Noc) or
            (Nbatch, Nx_out, Ny_out, Noc). The number of dimensions of the
            returned matrix will match the number of dimensions of the input
        matrix.
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

        Kx = self.kernel_size[0]
        Ky = self.kernel_size[1]
        Noc = self.Noc
        strideX = self.stride[0]
        strideY = self.stride[1]
        NrowsX = Kx * Ky * Nic  # number of rows per sliding window MVM

        # Number of sliding windows
        Nx_out, Ny_out = (
            (M_input.shape[2] - Kx) // strideX + 1,
            (M_input.shape[3] - Ky) // strideY + 1,
        )

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
