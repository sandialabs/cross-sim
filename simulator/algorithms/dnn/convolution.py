#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np
from ...cores.analog_core import AnalogCore
from ...backend import ComputeBackend

xp = ComputeBackend()


class Convolution:
    """Implements a crossbar that performs a convolutional kernel
    ---- INFERENCE ONLY! ----
    Outer product update and VMM are not implemented for convolutions.
    """

    def __init__(self, convParams, **kwargs):
        # Set the params of the Convolution core
        # If the Convolution core contains multiple neural cores, the master params object is a copy of the first
        # subcore params

        if "params" not in kwargs:
            raise ValueError(
                "params must be passed as an argument to initialize Convolution object",
            )
        params = kwargs["params"]

        self.derived_params = False

        #################
        ## Get convolution parameters
        #################

        self.Kx = convParams["Kx"]
        self.Ky = convParams["Ky"]
        self.stride = convParams["stride"]
        self.Nic = convParams["Nic"]
        self.Noc = convParams["Noc"]
        self.bias_row = convParams["bias_row"]

        # These parameters will be used for derivations later
        self.sameConv = convParams["sameConv"]
        if not self.sameConv:
            self.px_0 = convParams["px_0"]
            self.px_1 = convParams["px_1"]
            self.py_0 = convParams["py_0"]
            self.py_1 = convParams["py_1"]

        # If we don't have these values we'll get them from the first input
        if "Nix" in convParams:
            self.Nix = convParams["Nix"]
        if "Niy" in convParams:
            self.Niy = convParams["Niy"]

        #################
        ## Compute derived convolution parameters
        #################

        # Calculate number of rows in array
        if self.bias_row:
            self.Nrows = self.Kx * self.Ky * self.Nic + 1
        else:
            self.Nrows = self.Kx * self.Ky * self.Nic

        #################
        ## Instantiate analog cores
        #################

        # Keep a copy of the convolution parameters
        if type(params) is not list:
            self.params = params
        else:
            self.params = params[0].copy()

        # These parameter are strictly to be accessed by dnn.py
        self.nrow = convParams["Noc"]
        self.ncol = self.Nrows
        self.core = AnalogCore(
            np.zeros((self.nrow, self.ncol)),
            params=params,
        )

        # Map wrapper cores of AnalogCore to this core so that this object can be treated like an AnalogCore
        self.Ncores = self.core.Ncores
        self.cores = self.core.cores
        self.num_cores_row = self.core.num_cores_row
        self.num_cores_col = self.core.num_cores_col

        # With both Nix and Niy in the input params we can compute the derived values now
        if hasattr(self, "Nix") and hasattr(self, "Niy"):
            self._compute_derived_params()

    def _compute_derived_params(
        self,
    ):
        self.derived_params = True
        # Calculate output size
        if self.sameConv:
            (self.Nox, self.Noy) = (self.Nix // self.stride, self.Niy // self.stride)
        else:
            self.Nox = 1 + (self.Nix - self.Kx + self.px_0 + self.px_1) // self.stride
            self.Noy = 1 + (self.Niy - self.Ky + self.py_0 + self.py_1) // self.stride

        # Modify Nwindows here
        # This is dangerous because params can be copied so the params object won't
        # be reflected but all current uses of Nwindows refer to the object directly
        # Still, be very careful with this pattern, its kind of a hack
        self.params.simulation.convolution.Nwindows = self.Nox * self.Noy
        for row_list in self.cores:
            for core in row_list:
                core.params.simulation.convolution.Nwindows = self.Nox * self.Noy

        # If sameConv, calculate padding
        if self.sameConv:
            if (self.Kx % 2 != 0) and (self.Ky % 2 != 0):
                if self.Nix % self.stride == 0:
                    px = max(self.Kx - self.stride, 0)
                else:
                    px = max(self.Kx - (self.Nix % self.stride), 0)
                if self.Niy % self.stride == 0:
                    py = max(self.Ky - self.stride, 0)
                else:
                    py = max(self.Ky - (self.Niy % self.stride), 0)
                self.px_0 = px // 2
                self.px_1 = px - self.px_0
                self.py_0 = py // 2
                self.py_1 = py - self.py_0
            else:
                # Even size filter
                px = (self.Nox - 1) * self.stride + self.Kx - self.Nix
                py = (self.Noy - 1) * self.stride + self.Ky - self.Niy
                self.px_0 = px // 2
                self.px_1 = px - self.px_0  # px_1 is one more than px_0 if px is odd
                self.py_0 = py // 2
                self.py_1 = py - self.py_0

        # Now that Nox and Noy are set, check them against x_par and y_par
        x_par = self.params.simulation.convolution.x_par
        y_par = self.params.simulation.convolution.y_par

        if ((self.Nox % x_par) != 0) or ((self.Noy % y_par) != 0):
            print(
                "Warning: # sliding windows in a block ("
                + str(x_par)
                + ","
                + str(y_par)
                + ") "
                + "not divisible by total # windows ("
                + str(self.Nox)
                + ","
                + str(self.Noy)
                + ")",
            )
        if x_par > self.Nox or y_par > self.Noy:
            raise ValueError(
                "# sliding windows in a block ("
                + str(x_par)
                + ","
                + str(y_par)
                + ") "
                + "cannot be larger than output feature dimensions ("
                + str(self.Nox)
                + ","
                + str(self.Noy)
                + ")",
            )

    def set_matrix(self, matrix, verbose=False):
        """Set the weight matrix across all constituent wrapper cores
        The crossbar arrangement is as follows:
        - Along the rows are ALL the kernel weights for each input channel. Input channel 0 first, then input channel 1, ...
        - Along the columns are the weights for the different output channels.
        """

        # Check number of rows
        # matrix.shape[0] for VMM, matrix.shape[1] for MVM
        if matrix.shape[1] != self.Nrows:
            raise ValueError(
                "The number of rows in the weight matrix is inconsistent with conv core parameters",
            )

        # Check number of columns
        # matrix.shape[1] for VMM, matrix.shape[0] for MVM
        if matrix.shape[0] != self.Noc:
            raise ValueError(
                "The number of columns in the weight matrix is inconsistent with conv core parameters",
            )

        self.core.set_matrix(matrix, verbose=verbose)

    def get_matrix(self):
        """Read the internal matrix held by this core."""
        return self.core.get_matrix()

    @property
    def max(self):
        return self.core.max

    @property
    def min(self):
        return self.core.min

    @property
    def shape(self):
        return self.core.shape

    def __setitem__(self, key, value):
        # When setting the weights directly the input may not be reshaped yet
        # So reshape here. Should be safe as bias is 1D.
        if value.ndim > 2:
            value_ = value.reshape((value.shape[0], -1))
        else:
            value_ = value
        self.core.__setitem__(key, value_)

    def reshape_input(self, M_input):
        """Reshape a vector input to a matrix input with the dimensions specified for this conv layer. The vector must be the
        appropriate length.
        """
        if M_input.shape[-3:] == (self.Nic, self.Nix, self.Niy):
            return M_input

        # If input is a vector assume there is only one channel
        if len(M_input.shape) == 1 and self.Nic != 1:
            raise ValueError("Multiple input channels are needed for conv layer")

        # Try to avoid reshape operation in the channel dimension
        if len(M_input.shape) > 1 and M_input.shape[-3] != self.Nic:
            raise ValueError(
                "Input does not have correct dimensions to be used in conv layer",
            )

        try:
            if M_input.ndim == 3:
                return M_input.reshape(self.Nic, self.Nix, self.Niy)
            elif M_input.ndim == 4:
                return M_input.reshape(M_input.shape[0], self.Nic, self.Nix, self.Niy)
        except:
            raise ValueError(
                "Input does not have correct dimensions to be used in conv layer",
            )

    def apply_convolution(self, M_input):
        # Moderately hacky
        # If we haven't set Nix/Niy yet we need to do it from the first input
        # This assumes the first input is correctly shaped, if it isn't this will break

        # Convert input to xp array
        M_input = xp.array(M_input)

        if M_input.ndim < 4:
            if M_input.ndim == 1:
                M_input = M_input[:, None, None]
            Nic, Nix, Niy = M_input.shape
        else:
            _, Nic, Nix, Niy = M_input.shape

        if not self.derived_params:
            if Nic != self.Nic:
                raise ValueError(
                    "Number of channels in input matrix does not match "
                    "the number of input channels in the  convolutional layer. "
                    "This value will be used to set the input size so no reshape is "
                    "attempted."
                )
            self.Nix, self.Niy = Nix, Niy
            self._compute_derived_params()

        # Attempt to reshape input if incorrect size
        if M_input.shape[-3:] != (self.Nic, self.Nix, self.Niy):
            M_input = self.reshape_input(M_input)

        if Nic != self.Nic or Nix != self.Nix or Niy != self.Niy:
            raise ValueError(
                "The number of channels in the input matrix does not match the number of input "
                + "channels in the convolutional layer.",
            )

        # Choose the compute method based on sliding window and input image batching option
        if (
            self.params.simulation.convolution.conv_matmul
            and self.params.simulation.fast_matmul
        ):
            if M_input.ndim < 4:
                return self.apply_convolution_matmul(M_input)
            else:
                return self.apply_convolution_matmul_batched(M_input)
        else:
            if M_input.ndim < 4:
                return self.apply_convolution_matvec(M_input)
            else:
                return xp.stack(
                    [
                        self.apply_convolution_matvec(M_input[i])
                        for i in range(M_input.shape[0])
                    ]
                )

    def apply_convolution_matvec(self, M_input):
        """Applies a convolution operation to an input matrix M_input. Uses the sliding window method. Each MVM returns returns the outputs ...
        for all output channels for a single window. The results are then re-constructed into an output matrix that follows the same ...
        format as the input matrix.

        M_input: a 3D matrix of size (Nx, Ny, Nic). The third dimension must match the Nic in the conv core's parameters
        M_output: a 3D matrix of size (Nx_out, Ny_out, Noc). Nx_out and Ny_out is a function of the conv core's filter size, stride, and padding
        """
        Nic, Nx, Ny = M_input.shape
        Kx = self.Kx
        Ky = self.Ky
        Noc = self.Noc
        Nrows = self.Nrows
        stride = self.stride
        x_par = self.params.simulation.convolution.x_par
        y_par = self.params.simulation.convolution.y_par
        weight_reorder = self.params.simulation.convolution.weight_reorder
        NrowsX = Kx * Ky * Nic  # number of rows per sliding window MVM

        # Apply zero padding
        M_input = xp.pad(
            M_input,
            ((0, 0), (self.px_0, self.px_1), (self.py_0, self.py_1)),
            "constant",
        )

        # Number of sliding windows
        Nx_out, Ny_out = (
            (M_input.shape[1] - Kx) // stride + 1,
            (M_input.shape[2] - Ky) // stride + 1,
        )

        # Additional zero padding to account for output not being divisible by x_par, y_par
        if Nx_out % x_par != 0 or Ny_out % y_par != 0:
            x_pad = int(np.ceil(Nx_out / x_par) * x_par - Nx_out) * stride
            y_pad = int(np.ceil(Ny_out / y_par) * y_par - Ny_out) * stride
            M_input = xp.pad(M_input, ((0, 0), (0, x_pad), (0, y_pad)), "constant")

        # Allocate memory for the output
        M_out = xp.empty((Noc, Nx_out, Ny_out), dtype=M_input.dtype)

        for i in range(0, Nx_out, x_par):
            x_block = x_par if (Nx_out - i) >= x_par else Nx_out - i
            for j in range(0, Ny_out, y_par):
                y_block = y_par if (Ny_out - j) >= y_par else Ny_out - j
                x_start = i * stride
                y_start0 = j * stride
                if Kx == 1 and Ky == 1:
                    if not self.bias_row:
                        Min_block = M_input[
                            :,
                            x_start : (x_start + stride * x_par) : stride,
                            y_start0 : (y_start0 + stride * y_par) : stride,
                        ]
                        Min_large = Min_block.transpose((1, 2, 0)).flatten()
                    else:
                        Min_large = xp.ones(
                            int(Nrows * x_par * y_par),
                            dtype=M_input.dtype,
                        )
                        v_start, v_end = 0, NrowsX
                        for xxp in range(x_par):
                            y_start = y_start0
                            for yyp in range(y_par):
                                Min_ij = M_input[:, x_start, y_start]
                                Min_large[v_start:v_end] = Min_ij
                                y_start += stride
                                v_start += Nrows
                                v_end += Nrows
                            x_start += stride

                else:
                    if weight_reorder:
                        x_end = x_start + Kx + stride * (x_par - 1)
                        y_end = y_start0 + Ky + stride * (y_par - 1)
                        Min_large = M_input[:, x_start:x_end, y_start0:y_end]

                    else:
                        Min_ij = xp.zeros(
                            (Nic * x_par * y_par, Kx, Ky),
                            dtype=M_input.dtype,
                        )
                        x_end = x_start + Kx
                        v_start, v_end = 0, Nic

                        for xxp in range(x_par):
                            y_start = y_start0
                            y_end = y_start + Ky
                            for yyp in range(y_par):
                                Min_ij[v_start:v_end, :, :] = M_input[
                                    :,
                                    x_start:x_end,
                                    y_start:y_end,
                                ]
                                y_start += stride
                                y_end += stride
                                v_start += Nic
                                v_end += Nic
                            x_start += stride
                            x_end += stride

                        if self.bias_row:
                            Min_large = xp.ones(
                                (x_par * y_par, Nrows),
                                dtype=M_input.dtype,
                            )
                            Min_ij = Min_ij.reshape((x_par * y_par, NrowsX))
                            Min_large[:, :-1] = Min_ij
                        else:
                            Min_large = Min_ij

                M_out_p = self.core.mat_multivec(Min_large)
                # The line below is pure diabolical sorcery
                M_out[:, i : (i + x_block), j : (j + y_block)] = M_out_p.reshape(
                    (Noc, y_par, x_par),
                    order="F",
                ).transpose((0, 2, 1))[:, :x_block, :y_block]

        return M_out

    def apply_convolution_matmul(self, M_input):
        """Applies a convolution operation to an input matrix M_input. Uses matrix multiplication to compute all sliding windows in one shot.
        Each MVM returns returns the outputs for all output channels for a single window.
        The results are then re-constructed into an output matrix that follows the same format as the input matrix.

        M_input: a 3D matrix of size (Nx, Ny, Nic). The third dimension must match the Nic in the conv core's parameters
        M_output: a 3D matrix of size (Nx_out, Ny_out, Noc). Nx_out and Ny_out is a function of the conv core's filter size, stride, and padding
        """
        Nic, Nx, Ny = M_input.shape
        Kx = self.Kx
        Ky = self.Ky
        Noc = self.Noc
        stride = self.stride
        NrowsX = Kx * Ky * Nic  # number of rows per sliding window MVM

        # Apply zero padding
        M_input = xp.pad(
            M_input,
            ((0, 0), (self.px_0, self.px_1), (self.py_0, self.py_1)),
            "constant",
        )

        # Number of sliding windows
        Nx_out, Ny_out = (
            (M_input.shape[1] - Kx) // stride + 1,
            (M_input.shape[2] - Ky) // stride + 1,
        )

        if Kx == 1 and Ky == 1:
            M_input = M_input[:, ::stride, ::stride]
            M_input = M_input.reshape((NrowsX, Nx_out * Ny_out))
        else:
            # Diabolical sorcery #2
            M_input = xp.lib.stride_tricks.as_strided(
                M_input,
                (M_input.shape[0], Nx_out, Ny_out, Kx, Ky),
                (
                    M_input.strides[0],
                    M_input.strides[1] * stride,
                    M_input.strides[2] * stride,
                    M_input.strides[1],
                    M_input.strides[2],
                ),
            )
            M_input = M_input.transpose((0, 3, 4, 1, 2))
            M_input = M_input.reshape((NrowsX, Nx_out * Ny_out))

        if self.bias_row:
            M_input_all = xp.ones((self.Nrows, Nx_out * Ny_out), dtype=xp.float32)
            M_input_all[:-1, :] = M_input
            M_input = M_input_all

        M_out = self.core.matmat(M_input)
        M_out = M_out.reshape((Noc, Nx_out, Ny_out))

        return M_out

    def apply_convolution_matmul_batched(self, M_input):
        """Applies a convolution operation to an input matrix M_input which is 4D.
        Uses matrix multiplication to compute all sliding windows and all images in batch in one shot.
        Each MVM returns returns the outputs for all output channels for a single window.
        The results are then re-constructed into an output matrix that follows the same format as the input matrix.

        M_input: a 4D matrix of size (Nbatch, Nx, Ny, Nic). The fourth dimension must match the Nic in the conv core's parameters
        M_output: a 4D matrix of size (Nbatch, Nx_out, Ny_out, Noc). Nx_out and Ny_out is a function of the conv core's filter size, stride, and padding
        """
        Nbatch, Nic, Nx, Ny = M_input.shape
        Kx = self.Kx
        Ky = self.Ky
        Noc = self.Noc
        stride = self.stride
        NrowsX = Kx * Ky * Nic  # number of rows per sliding window MVM

        # Apply zero padding
        M_input = xp.pad(
            M_input,
            ((0, 0), (0, 0), (self.px_0, self.px_1), (self.py_0, self.py_1)),
            "constant",
        )

        # Number of sliding windows
        Nx_out, Ny_out = (
            (M_input.shape[2] - Kx) // stride + 1,
            (M_input.shape[3] - Ky) // stride + 1,
        )

        if Kx == 1 and Ky == 1:
            M_input = M_input[:, :, ::stride, ::stride]
            M_input = M_input.reshape((Nbatch, NrowsX, Nx_out * Ny_out))
        else:
            # Diabolical sorcery #2.5
            M_input = xp.lib.stride_tricks.as_strided(
                M_input,
                (M_input.shape[0], M_input.shape[1], Nx_out, Ny_out, Kx, Ky),
                (
                    M_input.strides[0],
                    M_input.strides[1],
                    M_input.strides[2] * stride,
                    M_input.strides[3] * stride,
                    M_input.strides[2],
                    M_input.strides[3],
                ),
            )
            M_input = M_input.transpose((0, 1, 4, 5, 2, 3))
            M_input = M_input.reshape((Nbatch, NrowsX, Nx_out * Ny_out))

        if self.bias_row:
            M_input_all = xp.ones(
                (Nbatch, self.Nrows, Nx_out * Ny_out), dtype=xp.float32
            )
            M_input_all[:, :-1, :] = M_input
            M_input = M_input_all

        M_out = self.core.matmat(M_input)
        M_out = M_out.reshape((Nbatch, Noc, Nx_out, Ny_out))

        return M_out
