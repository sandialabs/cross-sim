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
from torch import Tensor
from torch.nn import Module


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
    forward function and always match for the backward function. Attributes
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
            A CrossSimParameters object or list of CrossSimParameters objects
            for layers requiring multiple arrays. Typically passed directly
            to the core object.
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
        weight_mask:
            A (slice, slice) tuple indicating which elements of the analog
            array store the weights of the matrix.
        bias_mask:
            A (slice, slice) tuple indicating which elements of the analog
            array store the bias if the bias is implemented in analog. Can be
            empty if `self.analog_bias = False`
    """

    # MRO for implementing layers (Linear for example):
    # AnalogLinear, Linear, AnalogLayer, Module, ABC, Object

    def __init__(self):
        """Initialize AnalogLayer.

        All actual implementation is deferred to the original layer (e.g.
        torch.nn.Linear) and specific analog layer (e.g. AnalogLinear)
        """
        super().__init__()

    @abstractmethod
    def form_matrix(self) -> Tensor:
        """Builds 2D weight matrix for programming into the array.

        Returns:
            2D Torch Tensor of the matrix.

        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
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
    def get_core_weights(self) -> tuple[Tensor, Tensor | None]:
        """Gets the weight and bias tensors with errors applied.

        Returns:
            Tuple of Torch Tensors, 2D for weights, 1D or None for bias
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
                Number of analog rows to use for the bias. 0 indicates a
                digital bias. Ignored if layer does not have a bias.

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
            physical_weights: bool indicating whether to use ideal weights

        Returns:
            A new torch equivalent to the analog layer.

        """
        raise NotImplementedError

    def get_matrix(self) -> Tensor:
        """Returns the programmed 2D analog array.

        Returns:
            Torch tensor version of the 2D array with non-idealities applied.
        """
        return Tensor(self.core.get_matrix())

    # Names are hard, maybe sync_weights?
    # TODO: add kwargs for set_matrix? After core rework is done
    def synchronize(self) -> None:
        """Updates the analog weight representation based on weight parameters.

        Fully resamples programming error for the entire analog representation
        (weights and bias if in analog). Designed to ensure weight consistency
        after using an in-place tensor update on the weights and/or bias
        (e.g. add_()). Primarily used forCrossSim-in-the-loop training as
        optimizers use in-place updates.
        """
        self.core.set_matrix(self.form_matrix().detach())

    def _set_weight(self, weight: Tensor) -> None:
        """Updates the analog representation of the weights.

        Will fully resample the entire weight matrix. Scaling will be updated
        if needed for percentile weight scaling.
        """
        if self._consistent_limits():
            self.core[self.weight_mask] = weight.detach()
        else:
            # If this layer has an analog bias but it doesn't exist yet don't
            # bother forming the matrix with dummy data and forming the matrix
            # because we are just going to have to do it after bias is
            # allocated.
            if self.analog_bias and not hasattr(self, "bias"):
                return
            self.core.set_matrix(self.form_matrix().detach())

    def _set_bias(self, bias: Tensor) -> None:
        """Updates the analog representation of the bias.

        Will fully resample the entire analog bias. Scaling will be updated
        if needed for percentile weight scaling.
        """
        if self._consistent_limits():
            bias_expanded = (
                (bias / self.bias_rows)
                .reshape((self.core.shape[0], 1))
                .repeat((1, self.bias_rows))
            )
            self.core[self.bias_mask] = bias_expanded.detach()
        else:
            self.core.set_matrix(self.form_matrix().detach())

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
        w_consistent = w_max == self.core.max

        # Condition on hasattr for the case where analog_bias is true but bias
        # has not been initialized yet
        if self.analog_bias and hasattr(self, "bias"):
            (b_min, b_max) = AnalogCore._set_limits_percentile(
                weight_params,
                self.bias.detach(),
                reset=True,
            )
            b_consistent = b_max == self.core.max
        else:
            b_consistent = True

        return w_consistent and b_consistent

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
                self.reinitialize()
