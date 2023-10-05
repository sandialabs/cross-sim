#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np
from ...cores.analog_core import AnalogCore
from scipy.linalg import dft


class DFT:
    """Implements a crossbar that performs a Discrete Fourier Transform using direct MVM/VMM
    This is a simple example of a class that is built on top of the MVM interface.

    N_ft        : length of Fourier transform
    params      : parameters of simulated analog core
    normalize   : whether to normalize DFT matrix by sqrt(N)
    inverse     : whether to implement inverse DFT matrix (complex conjugate)

    """

    def __init__(self, N_ft, params=None, normalize=False, inverse=False):
        # Set the params of the DFT core
        if params is None:
            raise ValueError(
                "params must be passed as an argument to initialize DFT object",
            )

        # Limits of DFT matrix
        params.core.mapping.weights.max = 1
        params.core.mapping.weights.min = -1
        params.core.mapping.weights.percentile = None

        # Keep a copy of the core parameters
        if type(params) is not list:
            self.params = params.copy()
        else:
            self.params = params[0].copy()

        # Create and set the DFT matrix
        dft_mat = dft(N_ft).astype(np.complex64)
        if normalize:
            dft_mat /= np.sqrt(N_ft)
        if inverse:
            dft_mat = np.matrix.getH(dft_mat).astype(np.complex64)

        # These parameter are strictly to be accessed by dnn.py
        self.core = AnalogCore(dft_mat, params)

        # Map wrapper cores of AnalogCore to this core so that this object can be treated like an AnalogCore
        self.N_ft = N_ft
        self.Ncores = self.core.Ncores
        self.cores = self.core.cores

    def get_matrix(self):
        """Read the internal matrix held by this core."""
        return self.core.get_matrix()

    def dft_1d(self, x):
        """Computes the Discrete Fourier Transform of a 1D input x.
        Uses MVM.
        """
        if len(x) != self.N_ft:
            raise ValueError("Input length does not match DFT length of core.")

        return self.core.matvec(x)

    def dft_2d(self, X):
        """Computes the Discrete Fourier Transform of a 2D square input X.
        Uses both MVM and VMM!
        """
        if X.shape[0] != self.N_ft or X.shape[1] != self.N_ft:
            raise ValueError("Input length does not match DFT length of core.")

        y_imed = self.core @ X
        y_ft = y_imed @ self.core
        return y_ft
