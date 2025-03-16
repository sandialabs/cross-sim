#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""
Utility functions for calibrating input ranges for crossbar inputs and ADCs
based on profiled data. Compatible with both PyTorch and Keras interfaces.
These calibration methods are not guaranteed to be optimal.
"""

import os
import numpy.typing as npt
import numpy as np
from scipy.optimize import minimize
from simulator.parameters.core_parameters import CoreStyle, BitSlicedCoreStyle
from simulator.backend import ComputeBackend
xp = ComputeBackend()


def calibrate_input_limits(
    all_xbar_inputs: list,
    Nbits: int = 0,
    norm_ord: float = 1.0,
) -> npt.ArrayLike:
    """Optimizes the input range for all layers in a network given profiled
    input values. This function is intended for use with ResNet CNNs where
    all but the first layer is precded by a ReLU, so inputs are strictly positive.

    Note that this method may not work well for the first layer which has a different
    value distribution from the other layers!

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

    for k in range(n_layers):

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
    analog_layers: list,
    all_adc_inputs: list,
    Nbits: int = 0,
    norm_ord: float = 1.0,
    bitslice_pct: float = 99.99,
) -> npt.ArrayLike:
    """Optimizes the ADC input range for all layers in a network given profiled
    input values.

    Args:
        analog_layers: list of Torch analog modules or Keras analog layers containing 
            params that will be used to decide how to calibrate
        all_adc_inputs: list of arrays, each array contains profiled input
            values for a layer
        Nbits: quantization resolution used in optimizer
            Set to 0 to set range based on max profiled value
        norm_ord: power of the error norm used for the loss function in optimizer
            (Used for unsliced core only)
        bitslice_pct: desired percentile coverage of input distribution that is used to
            find ADC ranges. (Used for bitsliced core only)
    Returns:
        NumPy array containing the (min, max) range for the inputs of every layer
    """


    n_layers = len(all_adc_inputs)
    if analog_layers[0].params.core.style != CoreStyle.BITSLICED:
        adc_limits = np.zeros((n_layers, 2))
    else:
        # Allows non-uniform bit slice width across layers
        adc_limits = [None] * n_layers

    k = 0
    for layer in analog_layers:
        adc_inputs_k = xp.asarray(all_adc_inputs[k])

        if layer.params.core.style != CoreStyle.BITSLICED:
            adc_limits[k, :] = optimize_adc_limits_unsliced(
                adc_inputs_k, Nbits=Nbits, norm_ord=norm_ord
            )
        else:
            num_slices = layer.params.core.bit_sliced.num_slices
            adc_limits[k] = optimize_adc_limits_bitsliced(
                adc_inputs_k,
                num_slices,
                style = layer.params.core.bit_sliced.style,
                Nrows = Nrows,
                pct = bitslice_pct,
            )
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
    style: int = BitSlicedCoreStyle.BALANCED,
    Nrows: int = 1,
    pct: float = 99.99,
) -> npt.ArrayLike:
    """
    Optimizes the ADC input range for one layer which uses weight bit slicing.
    To reduce the overhead of bit slice digital post-processing, this method ensures
    that the ratio of the ADC limits of any two bit slices must be a power of 2.
    """

    # NOTE: Although input bit slices are profiled separately, the current calibration
    # method does not resolve data by input bit

    adc_limits_k = np.zeros((num_slices, 2))

    if style == BitSlicedCoreStyle.OFFSET:
        raise NotImplementedError(
            "ADC limits auto-calibration with weight bit slicing OFFSET "
            + "style not been implemented yet."
        )

    for i_slice in range(num_slices):
        adc_inputs_ik = adc_inputs_k[i_slice,:,:].flatten()
        adc_inputs_ik /= Nrows

        # Find the percentile extreme values of the ADC input distribution
        p_neg = xp.percentile(adc_inputs_ik, 100-pct)
        p_pos = xp.percentile(adc_inputs_ik, pct)
        p_out = xp.maximum(xp.abs(p_neg),xp.abs(p_pos))

        # Compute how much the ADC limits can be divided from the maximum possible,
        # and still cover the percentile extreme values
        clip_power_i = xp.floor(xp.log2(1/p_out)).astype(int)
        adc_limits_k[i_slice,0] = -Nrows / 2**clip_power_i
        adc_limits_k[i_slice,1] = Nrows / 2**clip_power_i

    return adc_limits_k



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
