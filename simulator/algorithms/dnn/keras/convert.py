#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""Utility functions for converting and managing CrossSim Keras layers."""

from simulator import CrossSimParameters
from keras.layers import Conv1D, Conv2D, Conv3D, Dense, Layer
from keras.models import Model, Sequential
from keras.src.models.functional import Functional
from keras.src.layers.normalization.batch_normalization import BatchNormalization

from copy import deepcopy, copy

from simulator.algorithms.dnn.keras.conv import AnalogConv1D, AnalogConv2D, AnalogConv3D
from simulator.algorithms.dnn.keras.dense import AnalogDense

import numpy as np

_conversion_map = {
    Conv1D: AnalogConv1D,
    Conv2D: AnalogConv2D,
    Conv3D: AnalogConv3D,
    Dense: AnalogDense,
}

CrossSimParameterList = list[CrossSimParameters]


def from_keras(
    model: Layer | Functional | Sequential,
    params: CrossSimParameters | list[CrossSimParameters | CrossSimParameterList],
    bias_rows: int | list[int] = 0,
    fuse_batchnorm: bool = False,
) -> Layer:
    """Convert a keras model to use CrossSim layers.

    This function provides the primary interface for using traditional keras
    networks to use CrossSim. from_keras iterates over the input model and
    replaces each layer of a convertible type (as defined in _conversion_map)
    into a CrossSim equivalent layer. By using lists of CrossSimParameters and
    bias_rows per-layer values can be applied during the conversion process.

    Args:
        model:
            Keras model to convert layers within. This can either be a singular
            a Sequential model, a functional model, or a model made through
            model subclassing
        params:
            CrossSimParameters or list (or list of lists) of CrossSimParameters
            to apply to the layers. Single object means a single set of
            parameters is used for all analog layers. A list of objects means
            one set of parameters per layer. If a list it must be the same
            length as convertible_models(model). List of lists allows
            parameters object per subcore per layer.
        bias_rows:
            Integer or list of integers indicating how many bias rows are used
            for each layer. Single integer means the same number of bias rows
            per layer, list means one per layer. If a list must be the same
            length as convertible_models(model)
        fuse_batchnorm:
            Bool indicating whether to fuse BatchNorm layers into the weights.
            For most analog systems BatchNorm is inefficient to compute so
            fusing them into the weight matrices is a good choice.

    Returns:
        A copy of the input model with layers replaced with CrossSim equivalents
    """
    # If the model itself is convertible, meaning it is a singular layer, convert
    # it so it doesn't fail when model.layers is called
    if type(model) in _conversion_map:
        return _conversion_map[type(model)].from_keras(model, params, bias_rows)

    # Enumerate the convertible layers for error checking
    convert_layers = convertible_layers(model)
    if isinstance(params, list) and len(convert_layers) != len(params):
        raise ValueError(
            f"Length of params list ({len(params)}) must match number "
            f"of convertible models ({len(convert_layers)}).",
        )

    if isinstance(bias_rows, list) and len(convert_layers) != len(bias_rows):
        raise ValueError(
            f"Length of bias_rows list ({len(bias_rows)}) must match number "
            f"of convertible models ({len(convert_layers)}).",
        )

    # Convert the model
    if isinstance(model, Sequential):
        converted_model = _convert_sequential_from_keras(
            model,
            params,
            bias_rows,
            fuse_batchnorm,
        )
    elif isinstance(model, Functional):
        converted_model = _convert_functional_from_keras(
            model,
            params,
            bias_rows,
            fuse_batchnorm,
        )
    else:
        raise NotImplementedError(
            "from_keras is not implemented for model subclassing.",
        )

    return converted_model


def to_keras(
    model: Layer | Functional | Sequential,
    physical_weights: bool | list[bool] = False,
) -> Layer:
    """Convert CrossSim layers to keras equivalents.

    This function provides an interface for converting all CrossSim layers in
    a Layer to their keras equivalents. to_keras iterates over the input model
    and replaces each CrossSim layer (as defined in _conversion_map) with the
    equivalent basic keras layer.

    Args:
        model:
            Keras model to convert layers within. This can either be a single layer,
            a Sequential model, a functional model. Other forms of model subclassing
            are not currently supported.
        physical_weights:
            Bool or list of bools indicating whether to use weights with
            physical non-idealities (e.g. programming and drift errors)
            applied. Single bool means the same value is applied to all layers,
            list allows per-layer specification. If a list must be the same
            length as analog_models(model)

    Returns:
        A copy of the input model with CrossSim layers replaced with keras
        equivalents
    """
    # If the model itself is convertible, meaning it is a singular layer, convert
    # it so it doesn't fail when model.layers is called
    if type(model) in _conversion_map.values():
        return type(model).to_keras(model, physical_weights)

    # Enumerate the convertible layers for error checking
    analog_layer = analog_layers(model)
    if isinstance(physical_weights, list) and len(analog_layer) != len(
        physical_weights,
    ):
        raise ValueError(
            f"Length of physical_weights list ({len(physical_weights)}) must match "
            f"number of convertible modules ({len(analog_layer)}).",
        )

    # Convert the model
    if isinstance(model, Sequential):
        converted_model = _convert_sequential_to_keras(model, physical_weights)
    elif isinstance(model, Functional):
        converted_model = _convert_functional_to_keras(model, physical_weights)
    else:
        raise NotImplementedError(
            "to_keras is not implemented for model subclassing.",
        )

    return converted_model


def convertible_layers(model: Layer) -> list[Layer]:
    """Returns a list of layers in a model with CrossSim equivalents.

    Args:
        model: keras model to examine.

    Returns:
        List of all layers in the model which have a CrossSim version.
    """
    # Bail out early since no layers have subclasses
    if type(model) in _conversion_map:
        return [model]

    return [l for l in model.layers if l in _conversion_map]


def analog_layers(model: Layer) -> list[Layer]:
    """Returns a list of CrossSim layers in a models.

    Args:
        model: Keras module to examine.

    Returns:
        List of all layers in the model which use CrossSim.
    """
    # Bail out early since no layers have subclasses
    if type(model) in _conversion_map.values():
        return [model]

    return [l for l in model.layers if l in _conversion_map.values()]


def inconvertible_layers(model: Layer) -> list[Layer]:
    """Returns a list of layers in a keras model without CrossSim equivalents.

    Args:
        model: Keras model to examine.

    Returns:
        List of all layers in the model which do not have a CrossSim version.
    """
    # Bail out early since no layers have subclasses
    if type(model) not in _conversion_map:
        return [model]

    return [l for l in model.layer if l not in _conversion_map]


def reinitialize(model: Layer) -> None:
    """Call reinitialize on all layers. Used to re-sample random conductance errors.

    Args:
        model: Keras model to synchronize weights.
    """
    for layer in analog_layers(model):
        layer.reinitialize()


# Function to fuse the weights of a Dense or Conv2D layer
def _fuse_bn_weights(
    input_layer: Dense | Conv2D,
    bn_layer: BatchNormalization,
):
    new_layer = deepcopy(input_layer)
    Wm_0 = input_layer.get_weights()
    Wm = Wm_0[0]
    if new_layer.use_bias:
        Wbias = Wm_0[1]

    if bn_layer.scale and bn_layer.center:
        gamma, beta, mu, var = bn_layer.get_weights()
    elif bn_layer.scale and not bn_layer.center:
        gamma, mu, var = bn_layer.get_weights()
        beta = 0
    elif not bn_layer.scale and bn_layer.center:
        beta, mu, var = bn_layer.get_weights()
        gamma = 1
    else:
        mu, var = bn_layer.get_weights()
        gamma, beta = 1, 0

    epsilon = bn_layer.epsilon
    if not new_layer.use_bias:
        Wbias = np.zeros(Wm.shape[-1])
        new_layer.use_bias = True

    Wm = gamma * Wm / np.sqrt(var + epsilon)
    Wbias = (gamma / np.sqrt(var + epsilon)) * (Wbias - mu) + beta
    new_layer.set_weights((Wm, Wbias))

    return new_layer


def _convert_sequential_from_keras(model, params, bias_rows, fuse_batchnorm: bool):
    added_layers = []
    new_model = deepcopy(model)
    # Need i because it is used to find the next layer for fuse batchnorm
    for i, layer in enumerate(model.layers):
        if type(layer) in _conversion_map:
            if fuse_batchnorm:
                # Check to make sure the layer is not the last layer in the sequential
                # and if its not, check that the following layer is a batchnorm
                if layer is not model.layers[-1] and isinstance(
                    model.layers[i + 1],
                    BatchNormalization,
                ):
                    layer = _fuse_bn_weights(
                        layer,
                        model.layers[i + 1],
                    )
            if isinstance(params, CrossSimParameters):
                params_ = params
            else:
                params_ = params[i]
            if isinstance(bias_rows, int):
                bias_rows_ = bias_rows
            else:
                bias_rows_ = bias_rows[i]

            analog_layer = _conversion_map[type(layer)].from_keras(
                layer,
                params_,
                bias_rows_,
            )
            added_layers.append(analog_layer)
        # if the layer is a batchnorm, skip it
        elif isinstance(layer, BatchNormalization) and fuse_batchnorm:
            continue
        else:
            analog_layer = copy(layer)
            added_layers.append(analog_layer)

    new_model._layers = added_layers
    return new_model


def _convert_functional_from_keras(model, params, bias_rows, fuse_batchnorm: bool):
    # Auxiliary dictionary to describe the network graph
    network_dict = {"input_layers_of": {}, "new_output_tensor_of": {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.operation.name
            if layer_name not in network_dict["input_layers_of"]:
                network_dict["input_layers_of"].update({layer_name: [layer.name]})
            elif layer.name not in network_dict["input_layers_of"][layer_name]:
                network_dict["input_layers_of"][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict["new_output_tensor_of"].update({model.layers[0].name: model.input})

    # Iterate over all layers after the inputmodel.summary(*)
    for idx, layer in enumerate(model.layers[1:]):
        change_made = False
        bn_update_layers = []
        # loop through the input layers of layer
        # if the input layer is a batchnorm layer, get the input layer of the
        # batchnorm layer so we can update the network_dict to skip the batchnorm
        for i in range(len(network_dict["input_layers_of"][layer.name])):
            if isinstance(
                model.get_layer(network_dict["input_layers_of"][layer.name][i]),
                BatchNormalization,
            ):
                bn_layer = network_dict["input_layers_of"][layer.name][i]
                input_layer = network_dict["input_layers_of"][bn_layer][0]
                bn_update_layers.append(input_layer)
                change_made = True
            else:
                # append the non-batchnorm layer to the list in case the next layer is
                # a batchnorm so when we update the network_dict, this layer isn't lost
                bn_update_layers.append(network_dict["input_layers_of"][layer.name][i])

        # change_made is true when the input layer of layer is a batchnorm, so update
        # the network_dict such that the new input layer of layer is the input layer
        # of the batchnorm layer, essentially jumping over the batchnorm layer
        if change_made and fuse_batchnorm:
            network_dict["input_layers_of"].update({layer.name: bn_update_layers})

        # Determine input tensors
        layer_input = [
            network_dict["new_output_tensor_of"][layer_aux]
            for layer_aux in network_dict["input_layers_of"][layer.name]
        ]

        if len(layer_input) == 1:
            layer_input = layer_input[0]

        x = layer_input
        # If type of layer is in _conversion_map, replace it with analog conversion
        if type(layer) in _conversion_map:
            if fuse_batchnorm is True:
                bn_layer = None
                # Loop through the succeeding layers of the model to find the
                # corresponding batchnormalization layer
                for succeeding_layer in model.layers[idx:]:
                    if network_dict["input_layers_of"][succeeding_layer.name][
                        0
                    ] == layer.name and isinstance(
                        succeeding_layer, BatchNormalization,
                    ):
                        bn_layer = succeeding_layer
                # Make sure a batchnorm layer was found
                if bn_layer is not None:
                    layer = _fuse_bn_weights(layer, bn_layer)

            if isinstance(params, CrossSimParameters):
                params_ = params
            else:
                params_ = params[idx]
            if isinstance(bias_rows, int):
                bias_rows_ = bias_rows
            else:
                bias_rows_ = bias_rows[idx]

            analog_layer = _conversion_map[type(layer)].from_keras(
                layer,
                params_,
                bias_rows_,
            )
            x = analog_layer(x)
        # if the layer is a batchnorm layer, skip it
        elif isinstance(layer, BatchNormalization) and fuse_batchnorm:
            continue
        else:
            # Rebuild a new layer to connect to the input tensor
            # and store the output tensor
            new_layer = type(layer)(**layer.get_config())
            if layer.built:
                new_layer.build_from_config(layer.get_build_config())
                new_layer.set_weights(layer.get_weights())
            # attach new layer to input tensor
            # new_layer's output tensor is stored in x
            x = new_layer(x)

        # Stores the output tensor of the new_layer to network_dict
        network_dict["new_output_tensor_of"].update({layer.name: x})

    return Model(inputs=model.inputs, outputs=x)


def _convert_sequential_to_keras(model, physical_weights):
    new_model = Sequential()
    for layer in model.layers:
        if type(layer) in _conversion_map.values():
            if isinstance(physical_weights, list):
                physical_weights_ = physical_weights.pop()
            else:
                physical_weights_ = physical_weights

            new_layer = type(layer).to_keras(layer, physical_weights_)
            new_model._layers.append(new_layer)
        else:
            new_layer = copy(layer)
            new_model._layers.append(new_layer)
    return new_model


def _convert_functional_to_keras(model, physical_weights):
    # Auxiliary dictionary to describe the network graph
    network_dict = {"input_layers_of": {}, "new_output_tensor_of": {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.operation.name
            if layer_name not in network_dict["input_layers_of"]:
                network_dict["input_layers_of"].update({layer_name: [layer.name]})
            elif layer.name not in network_dict["input_layers_of"][layer_name]:
                network_dict["input_layers_of"][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict["new_output_tensor_of"].update({model.layers[0].name: model.input})

    # Iterate over all layers after the inputmodel.summary(*)
    for layer in model.layers[1:]:
        # Determine input tensors
        layer_input = [
            network_dict["new_output_tensor_of"][layer_aux]
            for layer_aux in network_dict["input_layers_of"][layer.name]
        ]

        if len(layer_input) == 1:
            if isinstance(layer_input[0], list):
                layer_input = layer_input[0][0]
            else:
                layer_input = layer_input[0]

        x = layer_input
        # If type of layer is in _conversion_map, replace it with analog conversion
        if type(layer) in _conversion_map.values():
            if isinstance(physical_weights, list):
                physical_weights_ = physical_weights.pop()
            else:
                physical_weights_ = physical_weights

            new_layer = type(layer).to_keras(layer, physical_weights_)
            x = new_layer(x)
        else:
            # Rebuild a new layer to connect to the input tensor
            # and store the output tensor
            new_layer = type(layer)(**layer.get_config())
            if layer.built:
                new_layer.build_from_config(layer.get_build_config())
                new_layer.set_weights(layer.get_weights())
            # attach new layer to input tensor
            # new_layer's output tensor is stored in x
            x = new_layer(x)

        # Stores the output tensor of the new_layer to network_dict
        network_dict["new_output_tensor_of"].update({layer.name: x})

    return Model(inputs=model.inputs, outputs=x)
