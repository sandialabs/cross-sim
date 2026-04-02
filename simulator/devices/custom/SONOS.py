#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#


from simulator.devices.idevice import EmptyDevice
import numpy as np
from scipy.interpolate import interp1d
from simulator.backend import ComputeBackend

xp = ComputeBackend()


class SONOS(EmptyDevice):
    """This is an empirical programming error and drift model for the 40nm SONOS
    charge trapping memory.

    Described in:

    V. Agrawal, et al, "Subthreshold operation of SONOS analog memory to enable
    accurate low-power neural network inference", IEEE International Electron
    Devices Meeting (IEDM) 2022, 21.7.1-21.7.4.
    https://ieeexplore.ieee.org/abstract/document/10019564/.

    SUGGESTED ON/OFF RATIO : 1e7 (~0 to 16 uS)

    Three non-idealities are modeled, based on the statistics of experimentally
    measured SONOS device currents from 128K devices. Both effects are time
    dependent. For details, see the paper.
    1) Drift in the expected value of the SONOS current, relative to the value
        at time of programming. By definition, this is not applied if time = 0.
    2) Random variability in the SONOS current around the expected value, due to
        random write errors and device-to-device variations.
    3) Read noise in the SONOS current around the programmed value

    The first effect is shown in Fig. 10(b) of the paper. At each point in time,
    the dependence of the drift vs SONOS state is fit using a 10th degree
    polynomial with 11 coefficients. This is evaluated and added to the target
    currents to get the mean values after drift. The coefficients were extracted
    for measurements taken at t = 1, 2, 3, 4, and 5 days. For values of t
    between these time steps, the coefficients are interpolated.

    The second effect is shown in Fig. 10(a) of the paper. At each point in
    time, the dependence of the random variation vs SONOS state is fit using a
    saturating exponential function with 2 free parameters (A and B). This
    function is used to scale a 2D matrix of random normal numbers and this is
    added to the SONOS current values, after mean drift is applied. The values
    of A and B were extracted for measurements taken at t = 1, 2, 3, 4, and 5
    days. For values of t between these steps, A and B are interpolated.
    If time > 5 days, the polynomial coefficients and A and B will be
    extrapolated, and are not guaranteed to be accurate.

    The third effect is shown in Fig. 11(b) of the paper. The state dependence
    of the variance of read noise is modeled by a saturating exponential
    function similar to device-to-device variability, but with a different pair
    of coefficients: A_noise and B_noise. This function is used to scale a 2D
    matrix of random normal numbers, and this is added to the SONOS current
    values every time an MVM is called. These coefficients do not vary with
    time.
    """

    def __init__(self, *args, **kwargs):
        """Initialize SONOS Device Model."""
        super().__init__(*args, **kwargs)
        # Max SONOS current to use for weight storage
        self.Gmin = 1 / self.device_params.Rmax
        self.Gmax = 1 / self.device_params.Rmin
        self.Imax = self.Gmax * self.device_params.Vread
        self.Imax *= 1e9  # nanoAmps
        if self.on_off_ratio == 0:
            self.Imin = 0
        else:
            self.Imin = self.Imax / self.on_off_ratio

        # Check that parameters are within the range of the model
        if self.Imax > 3200:
            raise ValueError(
                "When using any SONOS error model, please set "
                + "xbar.device.Rmin and xbar.device.Vread so that the max "
                + "SONOS current is <= 3200 nA.",
            )

    def _calculate_current(self, input_):
        """Computes matrix of SONOS cell currents that map a matrix of
        normalized input values.

        Current is treated equivalently to conductance here since the
        drain-source voltage bias on the SONOS cell only has one non-zero value.
        """
        I = (
            self.Imin
            + (self.Imax - self.Imin) * (input_ - self.Gmin_norm) / self.Grange_norm
        )
        return I

    def programming_error(self, input_):
        """See documentation in drift_error()."""
        return self.drift_error(input_, time=0)

    def _interpolate_drift(self, I, time):
        """Interpolates drift.

        Args:
            I: Current
            time: time elapsed single device programming, to simulate.
                Units: days.

        This function does two things:
            1) Applies drift to the mean currents corresponding to the given
               time
            2) Returns the parameters A and B used to add random errors to the
               mean currents

        If the time does not belong to one of the set time points where values
        were measured,the state-dependent mean drift and the values of A and B
        are interpolated
        """
        t_vec = np.array([0, 1, 2, 3, 4, 5])
        polynomials = np.zeros((len(t_vec), 11))
        As, Bs = np.zeros(len(t_vec)), np.zeros(len(t_vec))

        # Variability parameters at time 0
        polynomials[0, :] = np.zeros(11)
        As[0], Bs[0] = 19.88665, 176.3115

        # Drift coefficients and variability parameters at t = 1 day
        polynomials[1, :] = np.array(
            [
                1.72684e-01,
                5.72552e-02,
                -6.85715e-04,
                3.50271e-06,
                -9.37682e-09,
                1.40575e-11,
                -1.19803e-14,
                5.41293e-18,
                -8.85448e-22,
                -1.66869e-25,
                5.95180e-29,
            ],
        )
        As[1], Bs[1] = 25.59102, 186.85544

        # Drift coefficients and variability parameters at t = 2 days
        polynomials[2, :] = np.array(
            [
                2.50712e-01,
                6.62195e-02,
                -6.83218e-04,
                2.99442e-06,
                -6.78255e-09,
                7.70482e-12,
                -3.04983e-15,
                -2.15919e-18,
                2.94617e-21,
                -1.23359e-24,
                1.85329e-28,
            ],
        )
        As[2], Bs[2] = 28.72753, 193.56834

        # Drift coefficients and variability parameters at t = 3 days
        polynomials[3, :] = np.array(
            [
                -1.75300e-02,
                9.76883e-02,
                -9.63791e-04,
                4.55551e-06,
                -1.22391e-08,
                1.93786e-11,
                -1.85899e-14,
                1.07659e-17,
                -3.58211e-21,
                5.96785e-25,
                -3.30410e-29,
            ],
        )
        As[3], Bs[3] = 31.04318, 198.90332

        # Drift coefficients and variability parameters at t = 4 days
        polynomials[4, :] = np.array(
            [
                4.03338e-02,
                1.03111e-01,
                -1.04740e-03,
                4.83961e-06,
                -1.28132e-08,
                1.99484e-11,
                -1.85139e-14,
                1.00008e-17,
                -2.84944e-21,
                3.01008e-25,
                1.18747e-29,
            ],
        )
        As[4], Bs[4] = 33.76214, 204.77023

        # Drift coefficients and variability parameters at t = 5 days
        polynomials[5, :] = np.array(
            [
                3.19927e-01,
                1.06596e-01,
                -1.04054e-03,
                4.50290e-06,
                -1.10352e-08,
                1.56477e-11,
                -1.27592e-14,
                5.45513e-18,
                -7.26841e-22,
                -2.43893e-25,
                7.14608e-29,
            ],
        )
        As[5], Bs[5] = 34.57909, 207.34309

        # If T is exactly at the measured time points, no need to interpolate
        if time in t_vec:
            ind = np.argmin(np.abs(t_vec - time))
            polynomial = polynomials[ind, :]
            A = As[ind]
            B = Bs[ind]

        # Interpolate between the measured time points
        else:
            polynomial = np.zeros(polynomials.shape[1])
            for i in range(polynomials.shape[1]):
                interp_func0 = interp1d(
                    t_vec,
                    polynomials[:, i],
                    kind="linear",
                    copy=True,
                    fill_value="extrapolate",
                )
                polynomial[i] = interp_func0(time)
            A = interp1d(t_vec, As, kind="linear", copy=True, fill_value="extrapolate")(
                time,
            )
            B = interp1d(t_vec, Bs, kind="linear", copy=True, fill_value="extrapolate")(
                time,
            )

        # Interpolate drift characteristic vs conductance
        if time > 0:
            I_diff = xp.zeros(I.shape)
            for i in range(len(polynomial)):
                I_diff += polynomial[i] * pow(I, i)
            I += I_diff

        return I, A, B

    def drift_error(self, input_, time):
        """Apply the complete error: mean drift and variability."""
        # Check that parameters are within the range of the model
        if time > 0 and self.Imax >= 1800:
            raise ValueError(
                "When using the SONOS drift error model, "
                + "please set xbar.device.Rmin and xbar.device.Vread so that the "
                + "max SONOS current is <= 1800 nA.",
            )

        I = self._calculate_current(input_)

        # Apply mean drift and compute the variability parameters (A and B)
        I, A, B = self._interpolate_drift(I, time)

        # Apply random variability
        sigma_I = xp.maximum(A - A * xp.exp(-I / B), 0)
        sigma_W = sigma_I / (self.Imax - self.Imin)
        input_ = self.Gmin_norm + self.Grange_norm * (I - self.Imin) / (
            self.Imax - self.Imin
        )

        if sigma_W.any():
            randMat = xp.random.normal(
                scale=self.Grange_norm,
                size=input_.shape,
            ).astype(input_.dtype)
            input_ = input_ + sigma_W * randMat

        # Make sure resulting current is non-negative
        input_ = input_.clip(0, None)

        return input_

    def read_noise(self, input_):
        """Apply read noise."""
        I = self._calculate_current(input_)

        A_noise, B_noise = 12.58037, 215.36557

        # Apply read noise
        sigma_I = xp.maximum(A_noise - A_noise * xp.exp(-I / B_noise), 0)
        sigma_W = sigma_I / (self.Imax - self.Imin)

        if sigma_W.any():
            randMat = xp.random.normal(
                scale=self.Grange_norm,
                size=input_.shape,
            ).astype(input_.dtype)
            input_ = input_ + sigma_W * randMat

        # Make sure resulting current is non-negative
        input_ = input_.clip(0, None)

        return input_
