#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""
Utility functions for retrieving profiled values from CrossSim Torch layers
and calibrating input ranges for the crossbar inputs and ADCs
"""

import os
import pickle
import numpy.typing as npt
import numpy as np
from torch.nn import Module
from simulator.algorithms.dnn.torch.convert import analog_modules
from simulator.algorithms.dnn.torch.conv import AnalogConv2d
from simulator.algorithms.dnn.torch.linear import AnalogLinear
from scipy.optimize import minimize
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


def quantizationError_ReLU(eta, x, Nbits, norm_ord):
    """Quantizes values over a range from the minimum value to a high
    percentile value of the data. The percentile is only applied on
    large positive values, assuming ReLU activation is used.

    Args:
        eta: parameter that controls the percentile used for clipping
               (to be optimized)
        x: data values to be quantized
        Nbits: quantization resolution in bits
        norm_ord: power of the error norm used for the loss function
    """

    # Clip
    P = 100 * (1 - pow(10, eta))
    P = xp.clip(P, 0, 100)
    x_min = 0  # assume ReLU
    x_maxP = xp.percentile(x, P)
    x_Q = x.copy()
    x_Q = x_Q.clip(x_min, x_maxP)

    # Quantize
    qmult = (2**Nbits - 1) / (x_maxP - x_min)
    x_Q = (x_Q - x_min) * qmult
    x_Q = xp.rint(x_Q, out=x_Q)
    x_Q /= qmult
    x_Q += x_min
    err = xp.linalg.norm(x - x_Q, ord=norm_ord)
    return float(err)


def quantizationError_minMax(etas, x, Nbits, norm_ord):
    """Quantizes values over a range by optimizing the upper and lower
    percentiles of the range.

    Args:
        etas: tuple of two parameters that control the lower and upper percentile
            used for clipping (to be optimized)
        x: data values to be quantized
        Nbits: quantization resolution in bits
        norm_ord: power of the error norm used for the loss function
    """
    # Clip
    etaMin, etaMax = etas
    P_min = 100 * pow(10, etaMin)
    P_max = 100 * (1 - pow(10, etaMax))
    P_min = xp.clip(P_min, 0, 100)
    P_max = xp.clip(P_max, 0, 100)
    x_min = xp.percentile(x, P_min)
    x_max = xp.percentile(x, P_max)
    x_Q = x.copy()
    x_Q = x_Q.clip(x_min, x_max)

    # Quantize
    qmult = (2**Nbits - 1) / (x_max - x_min)
    x_Q = (x_Q - x_min) * qmult
    x_Q = xp.rint(x_Q, out=x_Q)
    x_Q /= qmult
    x_Q += x_min
    err = xp.linalg.norm(x - x_Q, ord=norm_ord)
    return float(err)


def calibrate_input_limits(
    all_xbar_inputs: list,
    Nbits: int = 0,
    norm_ord: float = 1.0,
) -> npt.ArrayLike:
    """Optimizes the input range for all layers in a network given profiled
    input values. This function is intended for use with ResNet CNNs where
    all but the first layer is precded by a ReLU, so inputs are strictly positive.

    Args:
        all_xbar_inputs: list of arrays, each array contains profiled input
            values for a layer
        Nbits: quantization resolution used in optimizer
            Set to 0 to set range based on max profiled value
        norm_ord: power of the error norm used for the loss function in optimizer
    Returns:
        NumPy array containing the (min, max) range for the inputs of every layer
    """

    n_layers = len(all_xbar_inputs)
    input_limits = np.zeros((n_layers, 2))

    # First layer is based on the normalization transformation used for CIFAR-10
    # Using these limits for the first layer ensures no clipping will occur
    input_limits[0, :] = np.array([-2.64, 2.64])

    for k in range(1, n_layers):

        xbar_inputs_k = xp.asarray(all_xbar_inputs[k])

        if Nbits > 0:
            eta0 = -4
            # Optimize the input percentile
            eta = minimize(
                quantizationError_ReLU,
                eta0,
                args=(xbar_inputs_k, Nbits, norm_ord),
                method="nelder-mead",
                tol=0.1,
            )
            percentile_max = 100 * (1 - pow(10, eta.x[0]))
            xmax = xp.percentile(xbar_inputs_k, percentile_max)
        else:
            xmax = xp.max(all_xbar_inputs[k])

        input_limits[k, :] = np.array([0, float(xmax)])

    return input_limits


def calibrate_adc_limits(
    model: Module,
    all_adc_inputs: list,
    Nbits: int = 0,
    norm_ord: float = 1.0,
) -> npt.ArrayLike:
    """Optimizes the ADC input range for all layers in a network given profiled
    input values.

    Args:
        model: Torch module with params on which to calibrate ADC ranges
        all_adc_inputs: list of arrays, each array contains profiled input
            values for a layer
        Nbits: quantization resolution used in optimizer
            Set to 0 to set range based on max profiled value
        norm_ord: power of the error norm used for the loss function in optimizer
    Returns:
        NumPy array containing the (min, max) range for the inputs of every layer
    """

    n_layers = len(all_adc_inputs)
    adc_limits = np.zeros((n_layers, 2))
    k = 0

    for layer in analog_modules(model):
        adc_inputs_k = xp.asarray(all_adc_inputs[k])

        if layer.params.core.style != CoreStyle.BITSLICED:
            adc_limits_k = optimize_adc_limits_unsliced(
                adc_inputs_k, Nbits=Nbits, norm_ord=norm_ord
            )
        else:
            num_slices = layer.params.core.bit_sliced.num_slices
            adc_limits_k = optimize_adc_limits_bitsliced(
                adc_inputs_k, num_slices, Nbits=Nbits, norm_ord=norm_ord
            )

        adc_limits[k, :] = adc_limits_k
        k += 1

    return adc_limits


def optimize_adc_limits_unsliced(
    adc_inputs_k: npt.ArrayLike,
    Nbits: int = 0,
    norm_ord: float = 1.0,
) -> npt.ArrayLike:
    """Optimizes the ADC input range for one layer which does not using weight bit slicing."""

    # Although input bit slices are profiled separately, the current calibration
    # method does not resolve data by input bit
    adc_inputs_k = adc_inputs_k.flatten()

    if Nbits > 0:
        etas0 = (-4, -4)
        # Optimize the input percentile
        etas = minimize(
            quantizationError_minMax,
            etas0,
            args=(adc_inputs_k, Nbits, norm_ord),
            method="nelder-mead",
            tol=0.1,
        )
        percentile_min = 100 * pow(10, etas.x[0])
        percentile_max = 100 * (1 - pow(10, etas.x[1]))
        xmin = xp.percentile(adc_inputs_k, percentile_min)
        xmax = xp.percentile(adc_inputs_k, percentile_max)
    else:
        xmin = xp.min(adc_inputs_k)
        xmax = xp.max(adc_inputs_k)

    adc_limits_k = np.array([float(xmin), float(xmax)])

    return adc_limits_k


def optimize_adc_limits_bitsliced(
    adc_inputs_k: npt.ArrayLike,
    num_slices: int = 2,
    Nbits: int = 0,
    norm_ord: float = 1.0,
) -> npt.ArrayLike:
    """Optimizes the ADC input range for one layer which uses weight bit slicing."""

    # Although input bit slices are profiled separately, the current calibration
    # method does not resolve data by input bit

    raise NotImplementedError(
        "ADC limits calibration with weight bit slicing has "
        + "not been implemented in the PyTorch interface yet."
    )
