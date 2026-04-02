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


class RRAMMilo(EmptyDevice):
    """This programming error model is based on the data for the resistive
    random access memory (ReRAM) device.

    Described in:
    V. Milo, et al. "Optimized programming algorithms for multilevel RRAM in
    hardware neural networks",
    IEEE International Reliability Physics Symposium (IRPS), 2021.
    https://ieeexplore.ieee.org/document/9405119.

    SUGGESTED ON/OFF RATIO : 4.5 (50 to 225 uS)
    The programming error model is based on Fig. 4 of the paper.

    The I-V nonlinearity model is based on Fig. 1 of the paper, using the I-V
    curve between +/-0.5V for four of the states: G0 = 62.5 uS, 76.2 uS,
    93.9 uS, and 110 uS.
    The conductance is defined as the ratio of current to voltage near V=0.
    The nonlinearity is modeled by expressing the current as a third-order
    polynomial of voltage with no constant term:
        I = G0*V + NL2*V^2 + NL3*V^3
    This fits well to the four measured I-V curves, with a separate set of
    coefficients for each measured G.
    To model the continuous state dependence, we fit NL2 and NL3 as polynomial
    functions of log10(G0)
        NL2 = A0 + A1*log10(G0) + A2*log10(G0)^2 + A3*log10(G0)^3
        NL3 = B0 + B1*log10(G0) + B2*log10(G0)^2
    These A and B coefficients are stored in self.p_NL2 and self.p_NL3.
    These coefficients are fit based on the four datapoints between 62.5 uS and
    110 uS, and an artificial fifth point which assumed that NL2 = 0 and NL3 = 0
    at G = 225 uS.
    This fifth point was chosen based on the observation that the I-V curve
    becomes more linear with increasing conductance. This enables I-V
    nonlinearity simulation over the full conductance range where programming
    error data is also available, but it may not be fully accurate over the
    entire range. To guarantee a fully accurate I-V model, please set
    Gmin >= 62.5 uS and Gmax <= 110 uS.
    """

    def __init__(self, *args, **kwargs):
        """Initialize RRAM Milo Device Model."""
        super().__init__(*args, **kwargs)

        # Max conductance in RRAM model
        self.Gmax = 1 / self.device_params.Rmin
        self.Gmin = self.Gmax / self.on_off_ratio if self.on_off_ratio != 0 else 0

        # Max read voltage, used for I-V nonlinearity simulation
        self.Vread = self.device_params.Vread

        # Fit parameters for conductance dependence of nonlinearity
        # These parameters are only valid between G = 50 uS and 225 uS
        # Second-order nonlinearity
        self.p_NL2 = xp.array([0.0001388, 0.001541, 0.005711, 0.007063])
        # Third-order nonlinearity
        self.p_NL3 = xp.array([8.8710e-05, 6.7919e-04, 1.2968e-03])

        # Check that parameters are within the range of the model
        if self.Gmax > 225.01e-6 or self.Gmin < 49.99e-6:
            raise ValueError(
                "When using the RRAMMilo error model, please set "
                + "xbar.device.Rmin so that Gmax is <= 225 uS, and "
                + "xbar.device.Rmax so that Gmin is >= 50 uS.",
            )
        if self.device_params.nonlinear_IV.enable and self.Vread > 0.5001:
            raise ValueError(
                "When simulating I-V nonlinearity with the RRAMMilo model, "
                + "please set xbar.device.Vread <= 0.5 V."
            )

    def programming_error(self, input_):
        """Apply the RRAM programming eror based on Fig. 4."""
        # Convert xbar normalized conductances to real ReRAM conductance (uS)
        G = (
            self.Gmin
            + (self.Gmax - self.Gmin) * (input_ - self.Gmin_norm) / self.Grange_norm
        )

        # Convert to microSiemens for the error fit
        G *= 1.00e6

        # Determine the conductance programming error at these conductance
        # values. Fit coefficients: linear fit to the data in Fig. 4
        # (IGVVA-100 algorithm)
        A, B = -0.009107, 4.782321
        sigma_G = xp.maximum(A * G + B, 0)

        # Convert back to Siemens
        sigma_G *= 1.00e-6

        # Convert random error sigmas back to CrossSim normalized weight units
        # and apply this to the random matrix
        sigma_W = sigma_G / (self.Gmax - self.Gmin)

        if sigma_W.any():
            randMat = xp.random.normal(
                scale=self.Grange_norm,
                size=input_.shape,
            ).astype(input_.dtype)
            input_ = input_ + sigma_W * randMat

        return input_

    def nonlinear_current(self, Gmat, Vmat):
        """Returns the current through every cell."""
        # Convert from normalized to real conductance
        G = (
            self.Gmin
            + (self.Gmax - self.Gmin) * (Gmat - self.Gmin_norm) / self.Grange_norm
        )

        # Convert from normalized to real voltage
        V = Vmat * self.Vread

        # Compute the nonlinear coefficients for every device
        # Add 1e-16 to avoid nan's
        G_cond = G >= 1e-9
        NL2 = xp.polyval(self.p_NL2, xp.log10(G + 1e-16)) * G_cond
        NL3 = xp.polyval(self.p_NL3, xp.log10(G + 1e-16)) * G_cond

        # Compute the current for every device
        if Gmat.ndim == Vmat.ndim:
            Imat = G * V + NL2 * pow(V, 2) + NL3 * pow(V, 3)
        else:
            expand_dim = V.ndim - G.ndim
            Imat = G[(..., *(None,) * expand_dim)] * V
            Imat += NL2[(..., *(None,) * expand_dim)] * pow(V, 2)
            Imat += NL3[(..., *(None,) * expand_dim)] * pow(V, 3)

        # Normalize current
        Imat = (Imat / self.Vread) * self.Grange_norm / (
            self.Gmax - self.Gmin
        ) - Vmat * (
            self.Grange_norm * self.Gmin / (self.Gmax - self.Gmin) - self.Gmin_norm
        )

        return Imat

    def nonlinear_current_sum(self, Gmat, Vterm):
        """Returns the current sums accounting for I-V nonlinearity."""
        # Convert from normalized to real conductance
        G = (
            self.Gmin
            + (self.Gmax - self.Gmin) * (Gmat - self.Gmin_norm) / self.Grange_norm
        )

        # Convert from normalized to real voltage
        V = Vterm * self.Vread

        # Compute the nonlinear coefficients for every device
        G_cond = G >= 1e-9
        NL2 = xp.polyval(self.p_NL2, xp.log10(G + 1e-16)) * G_cond
        NL3 = xp.polyval(self.p_NL3, xp.log10(G + 1e-16)) * G_cond

        # Compute the current for every device
        Imat = xp.matmul(G, V)
        Imat += xp.matmul(NL2, pow(V, 2))
        Imat += xp.matmul(NL3, pow(V, 3))

        # Normalize current
        Imat = (Imat / self.Vread) * self.Grange_norm / (self.Gmax - self.Gmin)

        return Imat
