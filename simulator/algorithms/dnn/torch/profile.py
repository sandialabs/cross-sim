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
from ....parameters.core_parameters import CoreStyle

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
                all_adc_inputs_rck = xp.zeros(
                    (
                        Ncores_r,
                        Ncores_c,
                        core0_outputs.shape[0],
                        core0_outputs.shape[1],
                    ),
                )
                for r in range(Ncores_r):
                    for c in range(Ncores_c):
                        all_adc_inputs_rck[r, c, :, :] = cores[r][c].adc_inputs

                # Flatten along the partitions
                all_adc_inputs_rck = all_adc_inputs_rck.transpose(0, 1, 3, 2)
                all_adc_inputs_k = all_adc_inputs_rck.reshape(
                    -1, all_adc_inputs_rck.shape[-1]
                )

            else:
                num_slices = layer.core.params.core.bit_sliced.num_slices

                for j in range(num_slices):
                    core0_outputs_j = cores[0][0].adc_inputs[j, :, :]
                    bitslice_inputs_j = xp.zeros(
                        (
                            Ncores_r,
                            Ncores_c,
                            core0_outputs_j.shape[0],
                            core0_outputs_j.shape[1],
                        ),
                    )
                    for r in range(Ncores_r):
                        for c in range(Ncores_c):
                            bitslice_inputs_j[r, c, :, :] = cores[r][c].adc_inputs[
                                j, :, :
                            ]

                    # Flatten along the partitions
                    bitslice_inputs_j = bitslice_inputs_j.transpose(0, 1, 3, 2)
                    bitslice_inputs_j = bitslice_inputs_j.reshape(
                        -1, bitslice_inputs_j.shape[-1]
                    )

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
        pickle.dump(all_xbar_inputs, open(save_path, "wb"))

    return all_adc_inputs