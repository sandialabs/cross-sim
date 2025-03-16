#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""Base class for all CrossSim Torch layers.

AnalogLayer is the base class for all CrossSim equivalents of torch layers.
AnalogLayer provides a common interface for CrossSim Torch layers and basic
infrastructure.
"""


from __future__ import annotations

from abc import abstractmethod, ABC

from simulator import AnalogCore, CrossSimParameters
import torch
from torch.nn import Module
from warnings import warn

import numpy as np


class AnalogLayer(Module, ABC):
    """Base class for CrossSim torch layers.

    AnalogLayer is the base class for torch layers which internally use
    CrossSim for the forward pass. Implementing classes will provide
    implementations for the  specific matrix formation, and forward, and
    backward methods as well as functions for converting the layers to and from
    the base Torch layer types. Implementing  classes should inherit from the
    original torch layer and AnalogLayer as follows
    `AnalogLinear(torch.nn.Linear, AnalogLayer)` to ensure proper module
    resolution order. Implementing classes should exactly match the output of
    the original Torch layer when all error models are disabled for the
    forward and backward functions. Attributes
    below must be provided by implementing classes as AnalogLayer assumes
    their existence.

    Attributes:
        core:
            An AnalogCore or similar object for CrossSim simulation.
            self.core must provide the following functions and properties:
                get_matrix
                max
                min
                shape
                __setitem__
            In most cases these should be thin wrappers around the
            implementations in AnalogCore
        params:
            CrossSimParameters object or list of CrossSimParameters (for layers
            requiring multiple arrays) for the AnalogLinear layer. If a list, the
            length must match the number of arrays used within AnalogCore.
        analog_bias:
            Boolean indicating if the bias of the layer is part of the analog
            array or stored and computing digitally.
            Potential patterns for analog bias:
                self.analog_bias = bias and bias_rows > 0
                   (if bias can multiple rows)
                self.analog_bias = bias and not digital_bias
                    (if bias only has 1 row)
                self.analog_bias = False
                    (if layer does not support analog bias)
    """

    # Torch inexplicably doesn't have any native conversion from numpy dtypes to torch
    # Using a dict from the torch testing framework:
    # https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_utils.py#L1663
    _numpy_to_torch_dtype_dict = {
        np.dtype("int8"): torch.int8,
        np.dtype("int16"): torch.int16,
        np.dtype("int32"): torch.int32,
        np.dtype("int64"): torch.int64,
        np.dtype("float16"): torch.float16,
        np.dtype("float32"): torch.float32,
        np.dtype("float64"): torch.float64,
        np.dtype("complex64"): torch.complex64,
        np.dtype("complex128"): torch.complex128,
    }

    # MRO for implementing layers (Linear for example):
    # AnalogLinear, Linear, AnalogLayer, Module, ABC, Object

    def __init__(self):
        """Initialize AnalogLayer.

        All actual implementation is deferred to the original layer (e.g.
        torch.nn.Linear) and specific analog layer (e.g. AnalogLinear)
        """
        super().__init__()

    @abstractmethod
    def form_matrix(self) -> torch.Tensor:
        """Builds 2D weight matrix for programming into the array.

        Returns:
            2D Torch Tensor of the matrix.

        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements the inference (forward) operation for the layer.

        The result of forward should be identical to the original layer result
        when all error models are turned off.

        Args:
            x: Torch Tensor input to the layer operation.

        Returns:
            Torch Tensor result of the layer operation

        """
        raise NotImplementedError

    @abstractmethod
    def reinitialize(self) -> None:
        """Rebuilds the layer's internal core object.

        Allows parameters to be updated within a layer without rebuilding the
        layer. This will resample all initialization-time errors
        (e.g. programming error)  even if the models were not be changed.
        Alternatively,  reinitialize can be used to directly resample
        initialization-time errors.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_torch(
        cls,
        layer: Module,
        params: CrossSimParameters | list[CrossSimParameters],
        bias_rows: int,
    ) -> AnalogLayer:
        """Returns an analog version of a Torch layer.

        Creates a new Analog Layer with the same attributes as the original
        Torch layer.

        Arguments:
            layer: The Torch layer to copy
            params:
                CrossSimParameters object or list of CrossSimParameters objects
                (for layers requiring multiple arrays) for the analog layer.
            bias_rows:
                Integer indicating the number of analog rows to use for the bias.
                0 indicates a digital bias. Ignored if layer does not have a bias.

        Returns:
            A new analog equivalent layer

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def to_torch(cls, layer: AnalogLayer, physical_weights: bool = False) -> Module:
        """Returns a Torch layer from an analog layer.

        By default weights will be copied directly from the layer's weight
        attributes and therefore will not include non-idealities.
        If idealities are desired, use physical_weights = True.

        Arguments:
            layer: AnalogLayer to copy
            physical_weights:
                Bool indicating whether the torch layer should have ideal weights or
                weights with programming error applied.

        Returns:
            A new torch equivalent to the analog layer.

        """
        raise NotImplementedError

    def get_core_weights(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Gets the weight and bias tensors with errors applied.

        Returns:
            Tuple of Torch Tensors, one per variable.
        """
        w, b = self.core.get_core_weights()
        w = torch.from_dlpack(w)
        if not self.analog_bias:
            return (w, self.bias)
        else:
            return (w, torch.from_dlpack(b))

    def get_matrix(self) -> torch.Tensor:
        """Returns the programmed 2D analog array.

        Returns:
            Torch tensor of the 2D array with non-idealities applied.
        """
        return torch.Tensor(self.core.get_matrix())

    # TODO: add kwargs for set_matrix? After core rework is done
    def synchronize(self) -> None:
        """Updates the analog weight representation based on weight parameters.

        Fully resamples programming error for the entire analog representation
        (weights and bias if in analog). Designed to ensure weight consistency
        after using an in-place tensor update on the weights and/or bias
        (e.g. add_()). Primarily used for CrossSim-in-the-loop training as
        optimizers use in-place updates.
        """
        self.core.set_matrix(self.form_matrix().detach())

    def _set_weight(self, weight: torch.Tensor) -> None:
        """Updates the analog representation of the weights.

        Will fully resample the entire weight matrix. Scaling will be updated
        if needed for percentile weight scaling.
        """
        # For layers where matrix formation is more complicated than a reshape
        # (grouped convolutions), we still need to use the full matrix
        # formation function and then slice it
        formed_matrix = self.form_matrix().detach()
        if self._consistent_limits():
            self.core[self.core.weight_mask] = formed_matrix[self.core.weight_mask]
        else:
            # If this layer has an analog bias but it doesn't exist yet don't
            # bother forming the matrix with dummy data and forming the matrix
            # because we are just going to have to do it after bias is
            # allocated.
            if self.analog_bias and not hasattr(self, "bias"):
                return
            self.core.set_matrix(formed_matrix)

    def _set_bias(self, bias: torch.Tensor) -> None:
        """Updates the analog representation of the bias.

        Will fully resample the entire analog bias. Scaling will be updated
        if needed for percentile weight scaling.
        """
        # As with _set_bias, defer to the matrix formation to avoid needing to
        # reimplement layer-specific formation logic.
        formed_matrix = self.form_matrix().detach()
        if self._consistent_limits():
            self.core[self.core.bias_mask] = formed_matrix[self.core.bias_mask]
        else:
            self.core.set_matrix(formed_matrix)

    def _consistent_limits(self) -> bool:
        """Checks whether updates to internal attributes require rescaling.

        For layers using percentile weight scaling, updates to the weight
        or bias tensors that exceed the previous limits requiring re-scaling.
        This function checks whether either the bias or weight require exceed
        the previous limits. If the layer does not use percentile weight
        scaling limits are always consistent.

        Returns:
            Bool indicating if the weight and bias are within the core limits
        """
        # Params can be either a list or a single object set it up front
        if isinstance(self.params, CrossSimParameters):
            weight_params = self.params.core.mapping.weights
        elif isinstance(self.params, list):
            weight_params = self.params[0].core.mapping.weights
        else:
            raise ValueError("params must be CrossSimParameters or list.")

        # If we aren't using percentile scaling weights are always consistent
        # because they are specified in the params object.
        if not weight_params.percentile:
            return True

        (w_min, w_max) = AnalogCore._set_limits_percentile(
            weight_params,
            self.weight.detach(),
            reset=True,
        )
        w_consistent = w_max <= self.core.max

        # Condition on hasattr for the case where analog_bias is true but bias
        # has not been initialized yet
        if self.analog_bias and hasattr(self, "bias"):
            (b_min, b_max) = AnalogCore._set_limits_percentile(
                weight_params,
                self.bias.detach(),
                reset=True,
            )
            b_consistent = b_max <= self.core.max
        else:
            b_consistent = True

        return w_consistent and b_consistent

    @staticmethod
    def _set_device(
        device: torch.device | None,
        params: CrossSimParameters,
    ) -> torch.device:
        if not device:
            if params.simulation.useGPU:
                return torch.device("cuda:{}".format(params.simulation.gpu_id))
            else:
                return torch.device("cpu")

        if all((d not in str(device) for d in ("cpu", "cuda"))):
            warn(
                (
                    "Got device" + str(device) + ". "
                    "Only 'cpu' and 'cuda' devices are officially supported. Other "
                    "device types may result in additional copies or unexpected "
                    "behavior."
                ),
                category=RuntimeWarning,
                stacklevel=2,
            )
            return device

        if params.simulation.useGPU and "cuda" not in str(device):
            raise ValueError(
                "Device mismatch: layer device is not specified as 'cuda' but "
                "CrossSim specifies useGPU = True.",
            )
        if not params.simulation.useGPU and "cpu" not in str(device):
            raise ValueError(
                "Device mismatch: layer device is not specified as 'cpu' but "
                "CrossSim specifies useGPU = False.",
            )

        if (
            "cuda" in str(device)
            and device.index is not None
            and device.index != params.simulation.gpu_id
        ):
            warn(
                (
                    "device.index does not match params.simulation.gpu_id. This may "
                    "result in unexpected behavior."
                ),
                category=RuntimeWarning,
                stacklevel=2,
            )

        return device

    def __setattr__(self, name, value):
        """Triggers CrossSim-specific side effects for certain attributes.

        Several attributes require hooks to ensure consistency between the
        Torch view of the layer and the CrossSim view. For weight and bias,
        this updating the programmed array if the associated Tensor changes.
        This is needed for the consistency of the forward and backward
        directions as backward computations use the layers attributes.

        Note: in-place updates to the weight or bias Tensors (e.g. add_) are
        not seen by this change. After using in-place updates self.synchronize
        must be called explicitly.

        For changes to parameters or number of bias rows, self.core must be
        rebuilt with the correct matrix size and parameters.

        Implementing classes can add additional attributes with side effects
        such as multiple weight matrices or biases.
        """
        super().__setattr__(name, value)
        # If we haven't build the core yet we're still initializing and don't
        # need any special hooks
        if hasattr(self, "core"):
            if name == "weight":
                self._set_weight(value.data)
            if name == "bias" and self.analog_bias:
                self._set_bias(value.data)
            if name in ["params", "bias_rows"]:
                # TODO: technically a minor incompatibility with reinitialize and
                # lists of param objects, minor so error for now.
                if name == "params" and isinstance(value, list):
                    raise NotImplementedError(
                        "Setting params as a list after core creation is not supported."
                        "Make a new core with a list of parameters instead.",
                    )
                self.reinitialize()
