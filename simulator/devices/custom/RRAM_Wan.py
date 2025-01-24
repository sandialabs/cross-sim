#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from ..idevice import EmptyDevice
from ...backend import ComputeBackend

xp = ComputeBackend()


class RRAMWan(EmptyDevice):
    """This programming error model is based on the data for the resistive random access memory (ReRAM)
    device in:
    W. Wan, et al. "A compute-in-memory chip based on resistive random-access memory", Nature, 2022.
    https://www.nature.com/articles/s41586-022-04992-8.

    The programming error model uses a cubic fit to Extended Data Fig. 3 of the paper (t = 1s data).

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Max conductance in PCM model
        self.Gmax = 1 / self.device_params.Rmin

        # Convert to microSiemens
        self.Gmax *= 1e6

        self.Gmin = self.Gmax / self.on_off_ratio if self.on_off_ratio != 0 else 0

        #### Check that parameters are within the range of the model
        if self.Gmax > 39.26:
            raise ValueError(
                "When using the RRAMMilo error model, please set "
                + "xbar.device.Rmin so that Gmax is <= 39.26 uS.",
            )

    def programming_error(self, input_):
        # Convert xbar normalized conductances to real ReRAM conductance (uS)
        G = (
            self.Gmin
            + (self.Gmax - self.Gmin) * (input_ - self.Gmin_norm) / self.Grange_norm
        )

        # Determine the conductance programming error at these conductance values
        # Fit coefficients: cubic fit to the data in Extended Data Fig. 3
        A, B, C, D = 1.536e-4, -1.167e-2, 2.374e-1, 1.053
        sigma_G = xp.maximum(A * pow(G, 3) + B * pow(G, 2) + C * G + D, 0)

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
            "RRAM Wan is not a valid drift model. It can only be used to simulate programming error.",
        )
