#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""Utility functions for converting and managing CrossSim Torch layers."""

from simulator import CrossSimParameters
from torch.nn import (
    Module,
    Linear,
    Conv1d,
    Conv2d,
    Conv3d,
    Identity,
    BatchNorm2d,
    Sequential,
)
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval

from copy import deepcopy

from warnings import warn

from simulator.algorithms.dnn.torch.conv import AnalogConv1d, AnalogConv2d, AnalogConv3d
from simulator.algorithms.dnn.torch.linear import AnalogLinear

from typing import Callable

_conversion_map = {
    Linear: AnalogLinear,
    Conv1d: AnalogConv1d,
    Conv2d: AnalogConv2d,
    Conv3d: AnalogConv3d,
}

CrossSimParameterList = list[CrossSimParameters]

# TODO: Apparently deepcopy is not fully defined for torch.nn.Module and can
# fail in some cases. Look into how to avoid it, and/or if it is a problem
# for our use case.


def from_torch(
    model: Module,
    params: CrossSimParameters | list[CrossSimParameters | CrossSimParameterList],
    bias_rows: int | list[int] = 0,
    fuse_batchnorm: bool = False,
) -> Module:
    """Convert a torch model to use CrossSim layers.

    This function provides the primary interface for using traditional torch
    networks to use CrossSim. from_torch iterates over the input model and
    replaces each layer of a convertible type (as defined in _conversion_map)
    into a CrossSim equivalent layer. By using lists of CrossSimParameters and
    bias_rows per-layer values can be applied during the conversion process.

    Args:
        model: Torch module to convert layers within
        params:
            CrossSimParameters or list (or list of lists) of CrossSimParameters
            to apply to the layers. Single object means a single set of
            parameters is used for all analog layers. A list of objects means
            one set of parameters per layer. If a list it must be the same
            length as convertible_modules(model). List of lists allows
            parameters object per subcore per layer.
        bias_rows:
            Integer or list of integers indicating how many bias rows are used
            for each layer. Single integer means the same number of bias rows
            per layer, list means one per layer. If a list must be the same
            length as convertible_modules(model)
        fuse_batchnorm:
            Bool indicating whether to fuse BatchNorm layers into the weights.
            For most analog systems BatchNorm is inefficient to compute so
            fusing them into the weight matrices is a good choice.

    Returns:
        A copy of the input model with layers replaced with CrossSim equivalents
    """
    # Currently no convertible layers can have children so just bail out early
    # if the model is actually convertible

    if type(model) in _conversion_map:
        return _conversion_map[type(model)].from_torch(model, params, bias_rows)

    # Enumerate the convertible layers for error checking
    convert_layers = convertible_modules(model)
    if isinstance(params, list) and len(convert_layers) != len(params):
        raise ValueError(
            f"Length of params list ({len(params)}) must match number "
            f"of convertible modules ({len(convert_layers)}).",
        )

    if isinstance(bias_rows, list) and len(convert_layers) != len(bias_rows):
        raise ValueError(
            f"Length of bias_rows list ({len(bias_rows)}) must match number "
            f"of convertible modules ({len(convert_layers)}).",
        )

    # Batchnorm fusion proceed in two passes, first fuse the layers then
    # convert. fuse_bn returns a copy just let it replace the copy for the
    # fusion case
    if fuse_batchnorm:
        analog_model = _fuse_all_bn(model)
    else:
        analog_model = deepcopy(model)

    params_ = params.copy()
    # Reverse the lists so we can just use pop()
    if isinstance(params, list):
        params_.reverse()

    if isinstance(bias_rows, list):
        bias_rows_ = bias_rows[::-1]
    else:
        bias_rows_ = bias_rows

    _convert_children_from_torch(analog_model, params_, bias_rows_)

    return analog_model


def to_torch(
    model: Module,
    physical_weights: bool | list[bool] = False,
) -> Module:
    """Convert CrossSim layers to torch equivalents.

    This function provides an interface for converting all CrossSim layers in
    a module to their torch equivalents. to_torch iterates over the input model
    and replaces each CrossSim layer (as defined in _conversion_map) with the
    equivalent basic torch layer.

    Args:
        model: Torch module to convert layers within
        physical_weights:
            Bool or list of bools indicating whether to use weights with
            physical non-idealities (e.g. programming and drift errors)
            applied. Single bool means the same value is applied to all layers,
            list allows per-layer specification. If a list must be the same
            length as analog_modules(model)

    Returns:
        A copy of the input model with CrossSim layers replaced with torch
        equivalents
    """
    # Currently no CrossSim layers can have children so just bail out early
    # if the model is already a CrossSim layer.
    if type(model) in _conversion_map.values():
        return type(model).to_torch(model, physical_weights)

    # Enumerate the convertible layers for error checking
    analog_layers = analog_modules(model)
    if isinstance(physical_weights, list) and len(analog_layers) != len(
        physical_weights,
    ):
        raise ValueError(
            f"Length of physical_weights list ({len(physical_weights)}) must match "
            f"number of convertible modules ({len(analog_layers)}).",
        )

    torch_model = deepcopy(model)

    if isinstance(physical_weights, list):
        physical_weights_ = physical_weights[::-1]
    else:
        physical_weights_ = physical_weights

    _convert_children_to_torch(torch_model, physical_weights_)

    return torch_model


# TODO: make generators
def convertible_modules(model: Module) -> list[Module]:
    """Returns a list of layers in a model with CrossSim equivalents.

    Args:
        model: Torch module to examine.

    Returns:
        List of all layers in the model which have a CrossSim version.
    """
    out = []
    if type(model) in _conversion_map:
        out += [model]

    return out + _enumerate_module(
        model,
        lambda module: type(module) in _conversion_map,
    )


def analog_modules(model: Module) -> list[Module]:
    """Returns a list of CrossSim layers in a models.

    Args:
        model: Torch module to examine.

    Returns:
        List of all layers in the model which use CrossSim.
    """
    out = []
    if type(model) in _conversion_map.values():
        out += [model]

    return out + _enumerate_module(
        model,
        lambda module: type(module) in _conversion_map.values(),
    )


def inconvertible_modules(model: Module) -> list[Module]:
    """Returns a list of layers in a model without CrossSim equivalents.

    Args:
        model: Torch module to examine.

    Returns:
        List of all layers in the model which do not have a CrossSim version.
    """
    out = []
    if type(model) not in _conversion_map:
        out += [model]

    return out + _enumerate_module(
        model,
        lambda module: type(module) not in _conversion_map,
    )


def synchronize(model: Module) -> None:
    """Synchronize layer.weight with the analog weight for all model layers.

    Synchronize is used for CrossSim-in-the-loop training to update model
    weights after an optimizer uses an in-place update. This should called
    after each optimizer iteration.

    Args:
        model: Torch module to synchronize weights.
    """
    for layer in analog_modules(model):
        layer.synchronize()


def reinitialize(model: Module) -> None:
    """Call reinitialize on all layers. Mainly used to re-sample random conductance errors.

    Args:
        model: Torch module to synchronize weights.
    """
    for layer in analog_modules(model):
        layer.reinitialize()


def _convert_children_from_torch(
    module: Module,
    params: CrossSimParameters | list[CrossSimParameters],
    bias_rows: int | list[int],
):
    # Use named children so we can modify children by name (with setattr) later
    for name, child in module.named_children():
        # If this is a child we know how to convert, convert it
        if type(child) in _conversion_map:
            if isinstance(params, CrossSimParameters):
                params_ = params
            else:
                params_ = params.pop()

            if isinstance(bias_rows, int):
                bias_rows_ = bias_rows
            else:
                bias_rows_ = bias_rows.pop()

            setattr(
                module,
                name,
                _conversion_map[type(child)].from_torch(child, params_, bias_rows_),
            )

        # Then recursively descend
        _convert_children_from_torch(child, params, bias_rows)


def _convert_children_to_torch(
    module: Module,
    physical_weights: bool | list[bool],
):
    # Use named children so we can modify children by name (with setattr) later
    for name, child in module.named_children():
        # If this is a child we know how to convert, convert it
        if type(child) in _conversion_map.values():
            if isinstance(physical_weights, bool):
                physical_weights_ = physical_weights
            else:
                physical_weights_ = physical_weights.pop()

            setattr(
                module,
                name,
                type(child).to_torch(child, physical_weights_),
            )

        # Then recursively descend
        _convert_children_to_torch(child, physical_weights)


def _fuse_all_bn(model: Module) -> Module:
    """ """
    fused_model = deepcopy(model)

    # Torch requires the model be in eval mode for fusion
    fused_model.eval()

    _fuse_module_bn(fused_model)

    # Because BatchNorm fusion can sometimes miss layers, check and warn here
    layer_types = {type(i) for i in fused_model.modules()}
    if BatchNorm2d in layer_types:
        warn(
            (
                "Not all BatchNorm layers removed after fusion. This may be expected "
                "in some networks. The network will still function correctly; however, "
                "some layers may not accurately reflect the proper matrix values after "
                "fusion. If all BatchNorm layers were expected to be fused, the "
                "network may need to be restructured for automated BatchNorm fusion to "
                "work correctly. "
            ),
            category=RuntimeWarning,
            stacklevel=2,
        )

    return fused_model


def _fuse_module_bn(module: Module):
    # Same basic structure as _convert_module, could be combined but keeping
    # them separate for simplicity

    # Initialized prev_name and child to avoid errors
    prev_name = None
    prev_child = None

    # Since we're iterating over an ordered dict we can't remove during
    # iteration. Instead add to list and remove at the end if possible.
    to_remove = []

    for name, child in module.named_children():
        # These are the layers we know how to fuse
        if isinstance(child, Linear | Conv2d):
            prev_name, prev_child = name, child

        # Batchnorm layer to fuse
        elif isinstance(child, BatchNorm2d):
            if prev_name is None or prev_child is None:
                # This happens when a BatchNorm layer is in a different
                # container from the layer to be fused. To avoid this you
                # probably need to flatten or otherwise restructure the
                # network.
                continue

            if isinstance(prev_child, Linear):
                fused_layer = fuse_linear_bn_eval(prev_child, child)
            elif isinstance(prev_child, Conv2d):
                fused_layer = fuse_conv_bn_eval(prev_child, child)
            setattr(module, prev_name, fused_layer)
            # Replace the batchnorm layer with an identity layer
            # For Sequential layers we can just outright delete, but for other
            # types of layers (ResNet blocks) subsequent layers are referred
            # to by name so we need an Identity layer to avoid name errors.
            setattr(module, name, Identity())
            to_remove.append(name)

            # Unset these values to avoid double fusing
            prev_name = None
            prev_child = None

        # Otherwise unset prev_name and prev_child to avoid accidentally
        # fusing things
        else:
            prev_name = None
            prev_child = None

        _fuse_module_bn(child)

    if isinstance(module, Sequential):
        for child in to_remove:
            delattr(module, child)


def _enumerate_module(
    module: Module,
    condition: Callable[
        [
            Module,
        ],
        bool,
    ],
) -> list[Module]:
    match_list = []

    for _, child in module.named_children():
        if condition(child):
            match_list.append(child)

        match_list += _enumerate_module(child, condition)
    return match_list
