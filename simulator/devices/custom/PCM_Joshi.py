#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from ..idevice import EmptyDevice
from ..generic_device import NormalError
from ...backend import ComputeBackend

xp = ComputeBackend()


class PCMJoshi(EmptyDevice):
    """This drift + programming error model is based on the data for the phase change memory (PCM) device in:
    V. Joshi, et al. "Accurate deep neural network inference using computational phase-change memory",
    Nature Communications 11, 2473, 2020.
    https://www.nature.com/articles/s41467-020-16108-9.

    SUGGESTED ON/OFF RATIO : 100 (0.25 to 25 uS)

    The programming eror model is based on a quadratic fit to Fig. 3(b) of the paper.

    Drift is not modeled here as there is insufficient information to reproduce the time-dependent accuracy results.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distribution = NormalError(self.Grange_norm)

        # Max conductance in PCM model
        self.Gmax = 25.0  # microSiemens (uS)
        self.Gmin = self.Gmax / self.on_off_ratio if self.on_off_ratio != 0 else 0

        # In Joshi et al, the initial time (programming time) is 27.36s after programming
        # Fit coefficients, quadaratic fit to the data in Fig. 3(b)
        self.A, self.B, self.C = -0.00178767, 0.07585724, 0.28638599

    def programming_error(self, input_):
        """Apply the PCM programming error based on Fig. 3(b)."""
        # Convert xbar normalized conductances to real PCM conductance (uS)
        G = (
            self.Gmin
            + (self.Gmax - self.Gmin) * (input_ - self.Gmin_norm) / self.Grange_norm
        )

        # Determine the conductance programming error at these conductance values, using
        # a quadratic fit to the data in Fig. 3(b)
        sigma_G = xp.maximum(self.A * (G**2) + self.B * G + self.C, 0)

        # Convert back to CrossSim normalized weight units
        sigma_W = sigma_G / (self.Gmax - self.Gmin)
        random_matrix = self.distribution.create_error(input_)
        random_matrix *= sigma_W
        return input_ + sigma_W

    def drift_error(self, input_, time):
        # No drift model
        raise ValueError(
            "PCM Joshi is not a valid drift model. It can only be used to simulate programming error.",
        )
