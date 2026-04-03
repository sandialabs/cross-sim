#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from simulator.devices.idevice import EmptyDevice
from simulator.backend import ComputeBackend

xp = ComputeBackend()


class CubicDevice(EmptyDevice):
    """This is a simple analytical device model whose current is a cubic
    function of voltage.

    It does not represent a physical device but is meant to serve as an example
    of I-V nonlinearity modeling. This function implements the following
    I-V curve:
        I = G0 * (V - 0.25*V^3)
    where the nominal conductance G0 is defined at V=0.
    The modeled I-V nonlinearity will become more pronounced as
    self.device_params.Vread is increased.
    To avoid currents with the wrong polarity, we limit V to the range
    (-2V, +2V).
    """

    def __init__(self, *args, **kwargs):
        """Initializes a cubic nonlinear device."""
        super().__init__(*args, **kwargs)

        self.Gmax = 1 / self.device_params.Rmin
        self.Gmin = self.Gmax / self.on_off_ratio if self.on_off_ratio != 0 else 0

        # Max read voltage, used for I-V nonlinearity simulation
        self.Vread = self.device_params.Vread

        if self.device_params.nonlinear_IV.enable and self.Vread > 2.00001:
            raise ValueError(
                "When simulating I-V nonlinearity with the CubicDevice model, "
                + "please set xbar.device.Vread <= 2 V."
            )

    def nonlinear_current(self, Gmat, Vmat):
        """Returns the current through every cell."""
        # Convert Gmat and Vmat from normalized to real conductance or voltage
        G = (
            self.Gmin
            + (self.Gmax - self.Gmin) * (Gmat - self.Gmin_norm) / self.Grange_norm
        )
        V = Vmat * self.Vread

        # Compute the current for every device
        if Gmat.ndim == Vmat.ndim:
            Imat = G * (V - 0.25 * pow(V, 3))
        else:
            expand_dim = V.ndim - G.ndim
            Imat = G[(..., *(None,) * expand_dim)] * (V - 0.25 * pow(V, 3))

        # Normalize current
        Imat = (Imat / self.Vread) * self.Grange_norm / (
            self.Gmax - self.Gmin
        ) - Vmat * (
            self.Grange_norm * self.Gmin / (self.Gmax - self.Gmin) - self.Gmin_norm
        )
        return Imat

    def nonlinear_current_sum(self, Gmat, Vterm):
        """Returns the current sums accounting for I-V nonlinearity."""
        # Convert Gmat and Vmat from normalized to real conductance or voltage
        G = (
            self.Gmin
            + (self.Gmax - self.Gmin) * (Gmat - self.Gmin_norm) / self.Grange_norm
        )
        V = Vterm * self.Vread

        # Compute the current for every device, and normalize
        Imat = xp.matmul(G, V - 0.25 * pow(V, 3))
        Imat = (Imat / self.Vread) * self.Grange_norm / (self.Gmax - self.Gmin)

        return Imat
