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

    SUGGESTED ON/OFF RATIO : 4.5 (50 to 225 uS)
    The programming error model is based on Fig. 4 of the paper.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Max conductance in ReRAM model
        self.Gmax = 225  # microsiemens (uS)

        if self.on_off_ratio == 0:
            self.Gmin = 0
        else:
            self.Gmin = self.Gmax / self.on_off_ratio

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
