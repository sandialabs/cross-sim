#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""
Utility functions for collecting profiled array input values and ADC input
values from CrossSim Torch layers.
"""

import os
import pickle
import numpy.typing as npt
import numpy as np
from torch.nn import Module
from simulator.algorithms.dnn.torch.convert import analog_modules
from simulator.algorithms.dnn.torch.conv import AnalogConv2d
from simulator.algorithms.dnn.torch.linear import AnalogLinear
from ....parameters.core_parameters import CoreStyle, BitSlicedCoreStyle

from ....backend import ComputeBackend
xp = ComputeBackend()


def get_profiled_xbar_inputs(model: Module, save_dir: str = None) -> npt.ArrayLike:
    """Retrieves profiled inputs on all layers and optionally saves them to file

    Args:
        model: Torch module to retrieve profiled inputs from
        save_dir: Directory in which to save profiled input values (None to disable)
    Returns:
        List of numpy arrays, each array contains profiled values for a given layer
    """
    n_layers = len(analog_modules(model))
    all_xbar_inputs = [None] * n_layers
    k = 0
    for layer in analog_modules(model):

        if layer.core.params.simulation.analytics.profile_xbar_inputs:

            if type(layer) is AnalogConv2d:
                if layer.core.params.simulation.useGPU:
                    all_xbar_inputs[k] = layer.core.xbar_inputs.get().flatten()
                else:
                    all_xbar_inputs[k] = layer.core.xbar_inputs.flatten()

            if type(layer) is AnalogLinear:
                all_xbar_inputs[k] = layer.xbar_inputs.cpu().numpy().flatten()

            k += 1

    if save_dir is not None:
        save_path = os.path.join(save_dir, "all_xbar_inputs.p")
        pickle.dump(all_xbar_inputs, open(save_path, "wb"))

    return all_xbar_inputs


def get_profiled_adc_inputs(model: Module, save_dir: str = None) -> npt.ArrayLike:
    """Retrieves profiled ADC inputs on all layers and optionally saves them to file

    Args:
        model: Torch module to retrieve profiled inputs from
        save_dir: Directory in which to save profiled input values (None to disable)
    Returns:
        List of numpy arrays, each array contains profiled values for a given layer
    """
    n_layers = len(analog_modules(model))
    all_adc_inputs = [None] * n_layers
    k = 0

    for layer in analog_modules(model):

        if layer.core.params.simulation.analytics.profile_adc_inputs:

            cores = layer.core.cores
            Ncores_r = layer.core.num_cores_row
            Ncores_c = layer.core.num_cores_col

            if layer.params.core.style != CoreStyle.BITSLICED:
                core0_outputs = cores[0][0].adc_inputs
                all_adc_inputs_k = xp.zeros((core0_outputs.shape[0],0))
                for r in range(Ncores_r):
                    for c in range(Ncores_c):
                        all_adc_inputs_k = xp.concatenate((
                            all_adc_inputs_k,
                            cores[r][c].adc_inputs),
                            axis=1)

            else:
                num_slices = layer.core.params.core.bit_sliced.num_slices

                for j in range(num_slices):
                    core0_outputs_j = cores[0][0].adc_inputs[j, :, :]
                    bitslice_inputs_j = xp.zeros((core0_outputs_j.shape[0],0))
                    for r in range(Ncores_r):
                        for c in range(Ncores_c):
                            bitslice_inputs_j = xp.concatenate((
                                bitslice_inputs_j,
                                cores[r][c].adc_inputs[j,:,:]),
                                axis=1)
                    if j == 0:
                        all_adc_inputs_k = xp.zeros(
                            (
                                num_slices,
                                bitslice_inputs_j.shape[0],
                                bitslice_inputs_j.shape[1],
                            )
                        )
                    all_adc_inputs_k[j, :, :] = bitslice_inputs_j

            all_adc_inputs[k] = all_adc_inputs_k
            k += 1

    if save_dir is not None:
        save_path = os.path.join(save_dir, "all_adc_inputs.p")
        pickle.dump(all_adc_inputs, open(save_path, "wb"))

    return all_adc_inputs


def get_conductance_matrices(model: Module) -> npt.ArrayLike:
    """Retrieves the conductance matrices implementing all the layers of the network.
    Conductances are normalized (i.e. values are G/G_max) and will include
    any applied quantization, programming errors, and drift.

    Args:
        model: Torch module to retrieve profiled inputs from
    Returns:
        List of dictionaries containing conductance matrices and metadata.
        1) Top level: a list of length = # layers
        2) For each layer, there is a dictionary:
           'core_style' : mapping style of the core ("BALANCED", "OFFSET", "BITSLICED")
           'bitsliced_core' : mapping style of each bit slice core ("NONE", "BALANCED", "OFFSET")
           'num_slices' : number of bit slices
           'Gmats' : a list of length = # row partitions x # column partitions
        3) 'Gmats' has for each partition:
           3a) If not using weight bit slicing, there is a dictionary
               3a-1) If using balanced core, 'Gmat_pos' and 'Gmat_neg' contains the array conductances for
                   the positive and negative weight sub-arrays, respectively
               3a-2) If using offset core, 'Gmat' contains the array conductances for the single offset core
           3b) If using weight bit slicing, each partition has a list with length = # slices
               For each slice, there is a dictionary
               The entries of the dictionary depend on whether bit sliced core uses balanced or offset, same as above
    """
    Gmats = []

    for layer in analog_modules(model):

        Gmat_i = {}
        cores = layer.core.cores
        style_str = corestyle_str(
            layer.core.params.core.style, layer.core.params.core.bit_sliced.style
        )
        Gmat_i["core_style"] = style_str[0]
        Gmat_i["bitsliced_core_style"] = style_str[1]
        Gmat_i["num_slices"] = (
            layer.core.params.core.bit_sliced.num_slices
            if layer.core.params.core.style == CoreStyle.BITSLICED
            else 1
        )
        Gmat_i["Gmats"] = []

        # Check for fail condition
        fast_balanced = layer.core.params.simulation.fast_balanced
        cond1 = layer.core.params.core.style == CoreStyle.BALANCED and fast_balanced
        cond2 = (
            layer.core.params.core.style == CoreStyle.BITSLICED
            and layer.core.params.core.bit_sliced.style == BitSlicedCoreStyle.BALANCED
            and fast_balanced
        )
        if cond1 or cond2:
            raise ValueError(
                "To export conductances, please set "
                "params.simulation.disable_fast_balanced = True"
            )

        # Iterate through all arrays
        for r in range(layer.core.num_cores_row):
            for c in range(layer.core.num_cores_col):
                Gmat_rc = {}
                if layer.core.params.core.style == CoreStyle.BALANCED:
                    Gmat_rc["Gmat_pos"] = cores[r][c].core_pos._read_matrix()
                    Gmat_rc["Gmat_neg"] = cores[r][c].core_neg._read_matrix()
                elif layer.core.params.core.style == CoreStyle.OFFSET:
                    Gmat_rc["Gmat"] = cores[r][c].core._read_matrix()
                elif layer.core.params.core.style == CoreStyle.BITSLICED:
                    Gmat_rc = []
                    for islice in range(layer.core.params.core.bit_sliced.num_slices):
                        Gmat_slice = {}
                        if (
                            layer.core.params.core.bit_sliced.style
                            == BitSlicedCoreStyle.BALANCED
                        ):
                            Gmat_slice["Gmat_pos"] = (
                                cores[r][c].core_slices[islice][0]._read_matrix()
                            )
                            Gmat_slice["Gmat_neg"] = (
                                cores[r][c].core_slices[islice][1]._read_matrix()
                            )
                        elif (
                            layer.core.params.core.bit_sliced.style
                            == BitSlicedCoreStyle.OFFSET
                        ):
                            Gmat_slice["Gmat"] = (
                                cores[r][c].core_slices[islice][0]._read_matrix()
                            )
                        Gmat_rc.append(Gmat_slice)
                Gmat_i["Gmats"].append(Gmat_rc)
        Gmats.append(Gmat_i)
    
    return Gmats


# Return string pair specifying style of core and bit sliced core
def corestyle_str(core_style, bitsliced_core_style):
    if core_style == CoreStyle.BALANCED:
        return ("BALANCED", "NONE")
    elif core_style == CoreStyle.OFFSET:
        return ("OFFSET", "NONE")
    elif core_style == CoreStyle.BITSLICED:
        if bitsliced_core_style == BitSlicedCoreStyle.BALANCED:
            return ("BITSLICED", "BALANCED")
        elif bitsliced_core_style == BitSlicedCoreStyle.OFFSET:
            return ("BITSLICED", "OFFSET")