#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from ..idevice import EmptyDevice
from ...backend import ComputeBackend

xp = ComputeBackend()


class RRAMMilo(EmptyDevice):
    """This programming error model is based on the data for the resistive random access memory (ReRAM)
    device in:
    V. Milo, et al. "Optimized programming algorithms for multilevel RRAM in hardware neural networks",
    IEEE International Reliability Physics Symposium (IRPS), 2021.
    https://ieeexplore.ieee.org/document/9405119.

    The programming error model uses a linear fit to the data in Fig. 4 of the paper for
    conductances in the 50-225 uS range.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Max conductance in PCM model
        self.Gmax = 1 / self.device_params.Rmin

        # Convert to microSiemens
        self.Gmax *= 1e6

        self.Gmin = self.Gmax / self.on_off_ratio if self.on_off_ratio != 0 else 0

        #### Check that parameters are within the range of the model
        if self.Gmax > 225 or self.Gmin < 50:
            raise ValueError(
                "When using the RRAMMilo error model, please set "
                + "xbar.device.Rmin so that Gmax is <= 225 uS, and "
                + "xbar.device.Rmax so that Gmin is >= 50 uS.",
            )

    def programming_error(self, input_):
        # Convert xbar normalized conductances to real ReRAM conductance (uS)
        G = (
            self.Gmin
            + (self.Gmax - self.Gmin) * (input_ - self.Gmin_norm) / self.Grange_norm
        )

        # Determine the conductance programming error at these conductance values
        # Fit coefficients: linear fit to the data in Fig. 4 (IGVVA-100 algorithm)
        A, B = -0.009107, 4.782321
        sigma_G = xp.maximum(A * G + B, 0)

        # Convert random error sigmas back to CrossSim normalized weight units and
        # apply this to the random matrix
        sigma_W = sigma_G / (self.Gmax - self.Gmin)

        if sigma_W.any():
            randMat = xp.random.normal(
                scale=self.Grange_norm,
                size=input_.shape,
            ).astype(input_.dtype)
            input_ = input_ + sigma_W * randMat

        return input_

    def drift_error(self, input_, time):
        # No drift model
        raise ValueError(
            "RRAM Milo is not a valid drift model. It can only be used to simulate programming error.",
        )
