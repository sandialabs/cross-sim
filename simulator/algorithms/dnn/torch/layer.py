#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
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
    forward and backward functions. Attributes below must be provided by
    implementing classes as AnalogLayer assumes their existence.

    Attributes:
        core:
            An AnalogCore or similar object for CrossSim simulation.
            self.core must provide the following functions and properties:
                get_matrix
                set_matrix
                absmax
                get_core_weights
                __setitem__
            In most cases these should be thin wrappers around the
            implementations in AnalogCore
        params:
            CrossSimParameters object or list of CrossSimParameters (for layers
            requiring multiple arrays) for the AnalogLinear layer. If a list,
            the length must match the number of arrays used within AnalogCore.
        analog_bias:
            Boolean(s) indicating if the bias of the layer is part of the
            analog array or stored and computing digitally. There should be
            one boolean per variable in `_array_bias_variables` following the
            naming convention `analog_[bias name]`.
            Potential patterns for analog bias:
                self.analog_bias = bias and bias_rows > 0
                   (if bias can multiple rows)
                self.analog_bias = bias and not digital_bias
                    (if bias only has 1 row)
                self.analog_bias = False
                    (if layer does not support analog bias)
        _array_weight_variables:
            List containing the names of torch variables which corrospond to
            weights that would be programmed into the array. Names should be
            ordered based on the order of parameters from
            layer.named_parameters for consistency. This is used for partial
            array updates. It is assumed that all weight variables will
            always be programmed into the array. Values which are optionally
            included in the array should be specified as
            `_array_bias_variables` instead.
        _array_bias_variables:
            List containing the names of torch variables which corrospond to
            weights that would be programmed into the array. Names should be
            ordered based on the order of parameters from
            layer.named_parameters for consistency. This is used for partial
            array updates.
    """

    # Torch inexplicably doesn't have any native conversion from numpy dtypes to
    # torch. Using a dict from the torch testing framework:
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
        *bias_rows: int,
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
                Integers indicating the number of analog rows to use for the
                bias. Should have one integer for each value in
                `_array_bias_variables` 0 indicates a digital bias. Ignored
                if layer does not have a bias.

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
                Bool indicating whether the torch layer should have ideal
                weights or weights with programming error applied.

        Returns:
            A new torch equivalent to the analog layer.

        """
        raise NotImplementedError

    def get_core_weights(self) -> tuple[torch.Tensor | None, ...]:
        """Gets the weight and bias tensors with errors applied.

        Returns:
            Tuple of Torch Tensors, one per variable.
        """
        mats = self.core.get_core_weights()
        weight_mats = (
            torch.from_dlpack(i) for i in mats[: len(self._array_weight_variables)]
        )

        # For layers with multiple sublayers (e.g., RNNs) we add bias_ih
        # and bias_hh to _array_bias variables to simplify some other logic.
        # This creates an array lengh mismatch in zip so we need to manually
        # remove those. We assume that bias_ih and bias_hh are the last two
        # entries in _array_bias_variables because that is how they are all
        # implemented now.
        bias_iterator = zip(
            mats[len(self._array_weight_variables) :],
            self._array_bias_variables[: len(mats) - len(self._array_weight_variables)],
            strict=True,
        )
        bias_mats = [
            torch.from_dlpack(i[0])
            if self._analog_variable(i[1])
            # Some networks (e.g., RNNs) don't define bias attributes when
            # bias = False unlike other networks (including RNNCell)
            # To create consistent behavior default getattr to None
            else getattr(self, i[1], None)
            for i in bias_iterator
        ]
        return (*weight_mats, *bias_mats)

    def get_matrix(self) -> torch.Tensor | list[torch.Tensor]:
        """Returns the programmed 2D analog array.

        Returns:
            Torch tensor or list of torch tensors of the 2D array(s)
            with non-idealities applied.
        """
        mat = self.core.get_matrix()
        if isinstance(mat, list):
            return [torch.Tensor(m) for m in mat]
        else:
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

    def _set_analog_values(self, variable: str) -> None:
        """Updates the analog representation of a layer variable.

        If the specific portion of the matrix can be updated without modifying
        the array limits only that portion of the array will be updated.
        Otherwise the array will be fully resampled.

        Args:
            variable:
                The torch variable that has been updated. Must match one of the
                names in `_array_weight_variables` or `_array_bias_variables`
        """
        # Layers will set_matrix once they are initialized so to avoid errors
        # just skip this for now
        if not self.initialized:
            return

        # If this variable isn't in analog just skip
        if variable in self._array_bias_variables and not self._analog_variable(
            variable
        ):
            return

        # For layers where matrix formation is more complicated than a reshape
        # (grouped convolutions), we still need to use the full matrix
        # formation function and then slice it
        formed_matrix = self.form_matrix()

        # Layers involving multiple sub layers (e.g. multilayer RNNs) return
        # a list rather than a tensor. For these we need to use the "l[i]"
        # to decode which internal object we're using.
        if isinstance(formed_matrix, list):
            self._set_analog_value_list(variable, formed_matrix)
        else:
            self._set_analog_value_tensor(variable, formed_matrix)

    def _set_analog_value_list(self, variable: str, formed_matrix: list) -> None:
        variable_, layer = variable.split("_l", 1)
        layer = int(layer)
        matrix = formed_matrix[layer].detach()

        if self._consistent_limits(var_mask=f"_l{layer}", core_mask=layer):
            mask = self._mask_variable(variable)
            self.core[layer][mask] = matrix[mask]
        else:
            self.core[layer].set_matrix(matrix)

    def _set_analog_value_tensor(
        self, variable: str, formed_matrix: torch.Tensor
    ) -> None:
        matrix = formed_matrix.detach()
        if self._consistent_limits():
            mask = self._mask_variable(variable)
            self.core[mask] = matrix[mask]
        else:
            self.core.set_matrix(matrix)

    def _consistent_limits(
        self, var_mask: str | None = None, core_mask: int | None = None
    ) -> bool:
        """Checks whether updates to internal attributes require rescaling.

        For layers using percentile weight scaling, updates to the weight
        or bias tensors that exceed the previous limits requiring re-scaling.
        This function checks whether either the bias or weight require exceed
        the previous limits. If the layer does not use percentile weight
        scaling limits are always consistent.

        Args:
            var_mask:
                String indicating a pattern to check against variable names.
                Only variable names containing the mask will be considered for
                consistency.
            core_mask:
                Integer indicating which core object (if there are multiple
                cores) should be checked for consistency.

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

        for var in self.array_variables:
            # Since variables are initialized sequentially, during layer
            # initialization some values might not exist yet, if so just
            # skip them to avoid errors.
            if not hasattr(self, var):
                continue
            if var in self._array_bias_variables and not self._analog_variable(var):
                continue
            if var_mask is not None and var_mask not in var:
                continue
            weight = getattr(self, var).detach()
            (w_min, w_max) = AnalogCore._set_limits_percentile(
                weight_params,
                weight,
                reset=True,
            )
            w_absmax = max(w_max, abs(w_min))
            core_absmax = (
                self.core.absmax if core_mask is None else self.core.absmax[core_mask]
            )

            if w_absmax > core_absmax:
                return False
        return True

    def _validate_core(self) -> None:
        # Check the variable names, all weight names should be found
        # Bias names its ok not to have but all biases present in the layer
        # should be in the variables list
        names = {i[0] for i in self.named_parameters()}
        req_funcs = [
            "get_matrix",
            "set_matrix",
            "__setitem__",
            "get_core_weights",
            "absmax",
        ]
        self._check_variable_names(names)
        self._check_bias_variable_names(names)
        self._check_mask_attrs(names)
        self._check_analog_attrs()
        self._check_required_functions(req_funcs)

        # With the core validated, everything is now initialized
        self._initialized = True

    def _check_variable_names(self, names: list[str]):
        """Check if all the layer has all expected variables."""
        weight_set = set(self._array_weight_variables)
        if not weight_set.issubset(names):
            raise ValueError(
                f"Layer specifies {weight_set - names} as "
                f"an _array_weight_variable but it is not found in {names}. "
                f"Should this be an _array_bias_variable instead?"
                f"\nIf you are seeing this when converting a network, the "
                f"layer implementation is incorrect. This is probably not an "
                f"error with this specific class instantiation."
            )

    def _check_bias_variable_names(self, names: list[str]):
        """Check if optionally included names are specified as biases."""
        weight_set = set(self._array_weight_variables)
        bias_set = set(self._array_bias_variables)
        if not (names - weight_set).issubset(bias_set):
            raise ValueError(
                f"Layer contains {names - weight_set - bias_set} which is not "
                f"found in {self._array_bias_variables}. "
                f"This variable will not be included in the array"
                f"\nIf you are seeing this when converting a network, the "
                f"layer implementation is incorrect. This is probably not an "
                f"error with this specific class instantiation."
            )

    def _check_mask_attrs(self, names: list[str]):
        """Check that all variables have valid masks."""
        for var in names:
            if not (
                isinstance(self._mask_variable(var), tuple)
                and len(self._mask_variable(var)) == 2
                and all(isinstance(m, slice) for m in self._mask_variable(var))
            ):
                raise ValueError(
                    f"self.core has missing or invalid masks for parameter "
                    f"{var}. Masks are needed for getting and setting weight "
                    f"matrices."
                    f"\nIf you are seeing this when converting a network, the "
                    f"layer implementation is incorrect. This is probably not "
                    f"an error with this specific class instantiation."
                )

    def _check_analog_attrs(self):
        """Check that all biases have an associated analog_[bias] attribute."""
        for var in self._array_bias_variables:
            if self._analog_variable(var) is None:
                raise ValueError(
                    f"Layer is missing analog_{var} parameter."
                    f"\nIf you are seeing this when converting a network, the "
                    f"layer implementation is incorrect. This is probably not "
                    f"an error with this specific class instantiation."
                )

    def _check_required_functions(self, funcs: list[str]):
        for f in funcs:
            if not (hasattr(self.core, f) or f in dir(self.core)):
                raise ValueError(
                    f"self.core is missing function {f}."
                    f"\nIf you are seeing this when converting a network, the "
                    f"layer implementation is incorrect. This is probably not "
                    f"an error with this specific class instantiation."
                )

    @staticmethod
    def _set_device(
        device: torch.device | None,
        params: CrossSimParameters,
    ) -> torch.device:
        if not device:
            if params.simulation.useGPU:
                return torch.device(f"cuda:{params.simulation.gpu_id}")
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

    def __setattr__(self, name, value) -> None:
        """Triggers CrossSim-specific side effects for certain attributes.

        Several attributes require hooks to ensure consistency between the
        Torch view of the layer and the CrossSim view. For weights and biases,
        this updating the programmed array if the associated Tensor changes.
        This is needed for the consistency of the forward and backward
        directions as backward computations use the layers attributes.

        Note: in-place updates to the weight or bias Tensors (e.g. add_) are
        not seen by this change. After using in-place updates self.synchronize
        must be called explicitly.

        For changes to parameters or number of bias rows, self.core must be
        rebuilt with the correct matrix size and parameters.

        Implementing classes can add additional attributes with side effects.
        """
        super().__setattr__(name, value)
        if name == "core":
            self._validate_core()

        # If we haven't build the core yet we're still initializing and don't
        # need any special hooks
        if self.initialized:
            if name in self.array_variables:
                self._set_analog_values(name)
            if name == "params":
                if isinstance(value, list):
                    # TODO: technically a minor incompatibility with
                    # reinitialize and lists of param objects, minor so error
                    # for now.
                    raise NotImplementedError(
                        "Setting params as a list after core creation is not supported."
                        "Make a new core with a list of parameters instead.",
                    )
                self.reinitialize()
            if (
                "_rows" in name
                and name.removesuffix("_rows") in self._array_bias_variables
            ):
                self.reinitialize()

    def _analog_variable(self, var) -> bool | None:
        """Gets the 'analog_' prefixed version of a variable.

        Args:
            var: Variable to prefix

        Returns: The prefixed variable or None if that variable is not found.

        For tracking analog bias flags, certain variables are named according
        to the `analog_[var]` pattern. This returns the result from that
        pattern.
        """
        return getattr(self, f"analog_{var}", None)

    def _mask_variable(self, var) -> tuple[slice, slice] | None:
        """Gets the '_mask' suffixed version of a variable.

        Args:
            var: Variable to suffix

        Returns: The suffixeds variable or None if that variable is not found.

        For masked updates flags, certain variables are named according
        to the `[var]_mask` pattern. This returns the result from that
        pattern.
        """
        return getattr(self.core, f"{var}_mask", None)

    @property
    def array_variables(self) -> set[str]:
        """List of all names corrosponding to weight and bias variables."""
        return self._array_weight_variables + self._array_bias_variables

    @property
    def initialized(self):
        """Bool indicating if the layer has been fully initialized.

        Should be set automatically after the core object has been fully
        created.
        """
        return getattr(self, "_initialized", False)
