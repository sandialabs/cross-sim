#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import logging

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d, RectBivariateSpline

from simulator.devices.base_device import BaseDevice
from simulator.backend.compute import ComputeBackend
from simulator.parameters.base import RegisteredEnum

xp: np = ComputeBackend()  # Represents either cupy or numpy
log = logging.getLogger(__name__)


class ModeType(RegisteredEnum):
    """Mode determined by the inputs to the Interpolated Device function.

    "ONE_D": 1D Mode
    "TWO_D_MULTIDEVICE": 2D Multidevice Mode
    "TWO_D_DRIFT": 2D Drift Mode
    "THREE_D_USER_INTERP": 3D USER INTERPOLATED TIMESTAMP MODE
    "THREE_D_CROSSSIM_INTERP": 3D CROSSSIM INTERPOLATED TIMESTAMP MODE
    """

    ONE_D = 1
    TWO_D_MULTIDEVICE = 2
    TWO_D_DRIFT = 3
    THREE_D_USER_INTERP = 4
    THREE_D_CROSSSIM_INTERP = 5


class InterpolatedDevice(BaseDevice):
    """Interface to build a device model given raw data.

    There are 5 different ways to provide your data:
    - Option 1: 1D MODE
        data_array_1, data_array_2 as a 1D array of shape (len(targets),)
        where each entry is a the stdev/mean for the target. This assumes
        the user has pre-computed stdev/mean from their measurements.
        data_array_1 accounts for stdev and data_array_2 accounts for mean.
        This just creates a programming_error function.

    - Option 2: 2D MULTIDEVICE MODE
        data_array_1 as a 2D array of shape (len(targets), num_of_devices)
        where each row is the measured value for a single device.
        Internally CrossSim will compute the stdev/mean for each state
        and the result will look like 1D mode, programming error only.
        data_array_2 will remain None in this case.

    - Option 3: 2D DRIFT MODE
        data_array_1, data_array_2 as a 2D array of shape (len(targets),
        len(times)) where each row is the stdev or mean for the target at
        a given time. data_array_1 accounts for stdev and data_array_2
        accounts for mean. This assumes the user has pre-computed
        stdev/mean for their measurements. This creates a programming_error
        function and a drift function.

    - Option 4: 3D USER INTERPOLATED TIMESTAMP MODE
        data_array_1 as a 3D array of shape (len(targets), num_of_devices,
        len(times)) where data_array_1 is stack of 2D-Drift mode arrays
        and each stack corresponds to a different device. Internally we
        will calculate stdev/mean for each state,time pair and the result
        will look like 2D-Drift mode. data_array_2 will remain None in this
        case. This mode creates a drift and programming error function.
        The user will have to interpolate the raw data timestamps to a grid
        before passing it in.

    - Option 5: 3D CROSSSIM INTERPOLATED TIMESTAMP MODE
        data_array_1 as a 3D array of shape (len(targets), num_of_devices,
        len(times)) where data_array_1 is stack of 2D-Drift mode arrays
        and each stack corresponds to a different device. Internally we
        will calculate stdev/mean for each state,time pair and the result
        will look like 2D-Drift mode. data_array_2 will remain None in this
        case. This mode creates a drift and programming error function and
        CrossSim will handle the interpolation of the raw data timestamps
        to a grid after it has been passed in.

    """

    targets: npt.NDArray
    stdev_data: npt.NDArray
    mean_data: npt.NDArray
    conductance_data: npt.NDArray | None
    times: npt.NDArray | None
    mode: ModeType
    mean_interp_function: interp1d
    stdev_interp_function: interp1d

    def __init__(
        self,
        targets: npt.NDArray,
        data_array_1: npt.NDArray,
        data_array_2: npt.NDArray | None = None,
        times: npt.NDArray | None = None,
        *args,
        **kwargs,
    ):
        """Initializes a device based on the given measurements.

        Args:
            targets: List of target conductances for the characterized
                devices.
            data_array_1: 1D, 2D or 3D numpy array defined above.
            data_array_2: 1D, 2D, or 3D numpy array defined above
                (optional, if not included, no mean-shift is included).
            num_of_devices: Number of devices used for characterization
                (defaults to 1, the user has already pre-processed
                multi-device data).
            times: List of times at which devices were characterized (if list is
                empty or None, drift models are disabled).
        """
        self.targets = xp.asarray(targets)
        self.data_array_1 = data_array_1
        self.data_array_2 = data_array_2
        self.times = xp.asarray(times) if times is not None else None

        # Determine mode
        self.mode = self._determine_mode()

        # Validate inputs based on mode
        self._validate_inputs()

        super().__init__(*args, **kwargs)

        if self.device_params.Rmin <= 0 or self.device_params.Rmax <= 0:
            raise ValueError(
                "Minimum device resistance (Rmin) must be positive.",
            )

        self.Gmin = 1 / self.device_params.Rmax
        self.Gmax = 1 / self.device_params.Rmin

        if self.mode in [ModeType.ONE_D, ModeType.TWO_D_DRIFT]:
            self.stdev_data = self.data_array_1
            self.mean_data = self.data_array_2
        else:
            self.mean_data = np.mean(self.data_array_1, axis=1)
            self.stdev_data = np.std(self.data_array_1, axis=1)
        # Create mean interp function
        if self.mean_data is not None:
            if self.times is not None:
                self.mean_interp_function = RectBivariateSpline(
                    self.targets,
                    self.times,
                    self.mean_data,
                )
            else:
                self.mean_interp_function = interp1d(
                    self.targets,
                    self.mean_data,
                    kind="linear",
                    copy=True,
                    fill_value="extrapolate",
                )
        # Create stdev interp function
        if self.times is not None:
            self.stdev_interp_function = RectBivariateSpline(
                self.targets,
                self.times,
                self.stdev_data,
            )
        else:
            self.stdev_interp_function = interp1d(
                self.targets,
                self.stdev_data,
                kind="linear",
                copy=True,
                fill_value="extrapolate",
            )

    def _determine_mode(self):
        """Determines the implicit mode based on the explicit inputs."""
        if self.data_array_1.ndim == 1:
            return ModeType.ONE_D
        elif self.data_array_1.ndim == 2:
            if self.times is not None:
                return ModeType.TWO_D_DRIFT
            else:
                return ModeType.TWO_D_MULTIDEVICE
        elif self.data_array_1.ndim == 3:
            if self.times is None:
                raise ValueError(
                    "The times parameter must be provided when in a 3D mode.",
                )
            return (
                ModeType.THREE_D_USER_INTERP
                if self.data_array_1.shape[1] == len(self.times)
                else ModeType.THREE_D_CROSSSIM_INTERP
            )
        else:
            raise ValueError("data_array_1 must be a 1D, 2D, or 3D array.")

    def _validate_inputs(self):
        """Validates inputs based on the determined mode."""
        validation_func = {
            ModeType.ONE_D: self._validate_input_1d,
            ModeType.TWO_D_MULTIDEVICE: self._validate_input_2d_multidevice,
            ModeType.TWO_D_DRIFT: self._validate_input_2d_drift,
            ModeType.THREE_D_USER_INTERP: self._validate_input_3d_user_interp,
            ModeType.THREE_D_CROSSSIM_INTERP: self._validate_input_3d_crosssim_interp,
        }[self.mode]
        validation_func()

    def _validate_input_1d(self):
        """Validates inputs for 1D mode."""
        if self.data_array_1 is None:
            raise ValueError("data_array_1 must be provided for 1D mode.")
        if self.data_array_2 is None:
            raise ValueError("data_array_2 must be provided for 1D mode.")
        if self.times is not None:
            raise ValueError("times must not be provided for 1D mode.")
        if self.data_array_1.ndim != 1:
            raise ValueError("data_array_1 must be a 1D array for 1D mode.")
        if self.data_array_2.ndim != 1:
            raise ValueError("data_array_2 must be a 1D array for 1D mode.")
        if self.data_array_1.shape[0] != len(self.targets):
            raise ValueError(
                "The leading dimension of data_array_1 must be equal to len(targets).",
            )
        if self.data_array_2.shape[0] != len(self.targets):
            raise ValueError(
                "The leading dimension of data_array_2 must be equal to len(targets).",
            )

    def _validate_input_2d_multidevice(self):
        """Validates inputs for 2D multi-device mode."""
        if self.data_array_1 is None:
            raise ValueError("data_array_1 must be provided for 2D multi-device mode.")
        if self.times is not None:
            raise ValueError("times must not be provided for 2D multi-device mode.")
        if self.data_array_1.ndim != 2:
            raise ValueError(
                "data_array_1 must be a 2D array for 2D multi-device mode.",
            )
        if self.data_array_1.shape[0] != len(self.targets):
            raise ValueError(
                "The leading dimension of data_array_1 must be equal to len(targets).",
            )

    def _validate_input_2d_drift(self):
        """Validates inputs for 2D drift mode."""
        if self.data_array_1 is None or self.times is None or self.data_array_2 is None:
            raise ValueError(
                "data_array_1, data_array_2, and times must be provided for "
                "2D drift mode.",
            )
        if self.data_array_1.ndim != 2 or self.data_array_1.shape[1] != len(self.times):
            raise ValueError(
                "data_array_1 must be a 2D array with the 2nd dimension equal to "
                "len(times) for 2D drift mode.",
            )
        if self.data_array_2 is not None and (
            self.data_array_2.ndim != 2 or self.data_array_2.shape[1] != len(self.times)
        ):
            raise ValueError(
                "data_array_2 must be a 2D array with the second dimension equal to "
                "len(times) for 2D drift mode.",
            )
        if self.data_array_1.shape[0] != len(self.targets):
            raise ValueError(
                "The leading dimension of data_array_1 must be equal to len(targets).",
            )
        if self.data_array_2 is not None and self.data_array_2.shape[0] != len(
            self.targets,
        ):
            raise ValueError(
                "The leading dimension of data_array_2 must be equal to len(targets).",
            )
        if not np.all(
            np.diff(self.times) >= 0,
        ):  # Assert times is sorted in ascending order
            raise ValueError(
                ("times needs to be sorted in ascending order."),
            )

    def _validate_input_3d_user_interp(self):
        """Validates inputs for 3D user interpolated timestamp mode."""
        if self.data_array_1 is None or self.times is None:
            raise ValueError(
                "data_array_1 and times must be provided for 3D user interp mode.",
            )
        if self.data_array_1.ndim != 3 or self.data_array_1.shape[2] != len(self.times):
            raise ValueError(
                "data_array_1 must be a 3D array with the last dimension equal to "
                "len(times) for 3D user interpolated mode.",
            )
        if self.data_array_1.shape[0] != len(self.targets):
            raise ValueError(
                "The leading dimension of data_array_1 must be equal to len(targets).",
            )
        if not np.all(
            np.diff(self.times) >= 0,
        ):  # Assert times is sorted in ascending order
            raise ValueError(
                ("times needs to be sorted in ascending order."),
            )

    def _validate_input_3d_crosssim_interp(self):
        """Validates inputs for 3D CrossSim interpolated timestamp mode."""
        if self.data_array_1 is None or self.times is None:
            raise ValueError(
                "data_array_1 and times must be provided for 3D CrossSim interp mode.",
            )
        if self.data_array_1.ndim != 3 or self.data_array_1.shape[2] != len(self.times):
            raise ValueError(
                "data_array_1 must be a 3D array with the last dimension equal to "
                "len(times) for 3D CrossSim interpolated mode.",
            )
        if self.data_array_1.shape[0] != len(self.targets):
            raise ValueError(
                "The leading dimension of data_array_1 must be equal to len(targets).",
            )
        if not np.all(
            np.diff(self.times) >= 0,
        ):  # Assert times is sorted in ascending order
            raise ValueError(
                ("times needs to be sorted in ascending order."),
            )

    def programming_error(self, input_):
        """Calculates the programming error on given conductances values.

        Args:
            input_: An array of normalized conductance values that the user
                wants tosimulate programming error on.

        Returns:
            np.NDArray: An array of normalized conductance values with the
                error applied to them.
        """
        G = (
            self.Gmin
            + (self.Gmax - self.Gmin) * (input_ - self.Gmin_norm) / self.Grange_norm
        )
        W_error = self._calculate_error(G, time=0)

        randMat = np.random.normal(scale=self.Grange_norm, size=input_.shape).astype(
            input_.dtype,
        )
        randMat *= W_error
        input_ += randMat

        # Make sure resulting current is non-negative
        error_output = xp.clip(input_, 0, None)

        return error_output

    def drift_error(self, input_, time):
        """Calculates the drift error given some conductances and time.

        Args:
            input_: An array of normalized conductance values that the user
                wants to simulate drift error on.
            time: An int that represents the amount of time drifting before
                measuring error

        Raises:
            ValueError: Can't simulate drift error if the ModeType is
                option 1 or 2

        Returns:
            np.NDArray: An array of normalized conductance values with the
                error applied to them.
        """
        if self.mode in [ModeType.ONE_D, ModeType.TWO_D_MULTIDEVICE]:
            raise ValueError(
                "Drift error cannot be simulated by InterpolatedDevice,"
                " no time array provided",
            )
        # Values coming in are normalized and must be converted to true values
        G = (
            self.Gmin
            + (self.Gmax - self.Gmin) * (input_ - self.Gmin_norm) / self.Grange_norm
        )

        W_error = self._calculate_error(G, time=time)

        randMat = np.random.normal(scale=self.Grange_norm, size=input_.shape).astype(
            input_.dtype,
        )
        randMat *= W_error
        input_ += randMat

        # Make sure resulting current is non-negative
        error_output = xp.clip(input_, 0, None)

        return error_output

    def _calculate_error(self, G: np.ndarray, time: int):
        """Calculates complete error given the time and true conductance values.

        Args:
            G: True values of the conductances
            time: A time in which the user wants the measurements to
                be simulated

        Returns:
            np.NDArray: An array of sigma_W errors to be applied to the input
        """
        if self.times is not None:
            mean = self.mean_interp_function(G.flatten(), time)
            sigma_G = self.stdev_interp_function(mean, time).reshape(G.shape)
        else:
            mean = self.mean_interp_function(G)
            sigma_G = self.stdev_interp_function(mean)

        sigma_W = sigma_G / (self.Gmax - self.Gmin)

        return sigma_W
