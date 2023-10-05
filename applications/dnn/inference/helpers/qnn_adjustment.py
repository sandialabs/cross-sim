#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np
from simulator.parameters.core_parameters import CoreStyle, BitSlicedCoreStyle

def qnn_adjustment(model_name, params_m, Wm, Ncores, m):
    """
    Utility function to be called from inside inference_net/inference for special handling
    of quantization-aware trained neural networks
    For these networks, weight limits are not set using percentiles, and different layers may have
    different quantizations, so special care is required
    """

    params_0 = (params_m if Ncores == 1 else params_m[0])
    bits = params_0.core.weight_bits
    if bits > 0:
        cell_bits = params_0.xbar.device.cell_bits
        if model_name == "MobilenetV1-int8":
            bits += 1
            # Need to represent 9-bit weights
            # If not using bit slicing, need to raise cell precision by 1 bit
            # If using bit slicing, need to raise cell precision if OFFSET
            # If using bit slicing and BALANCED, get 9 bits for free
            if params_0.core.style != CoreStyle.BITSLICED:
                cell_bits += 1
            else:
                Nslices = params_0.core.bit_sliced.num_slices
                if bits % Nslices != 0:
                    if not params_0.core.bit_sliced.style == BitSlicedCoreStyle.BALANCED:
                        cell_bits += 1
            if m == 1 and cell_bits != params_0.xbar.device.cell_bits:
                print("** For MobilenetV1-int8, resolution changed to: {:d} bits/cell".format(cell_bits))
                
        elif model_name == "Resnet50-int4":
            if m == 1 or m == 54:
                bits = 8
            else:
                bits = 4
            if params_0.core.style == CoreStyle.BITSLICED:
                Nslices = params_0.core.bit_sliced.num_slices
                if bits % Nslices == 0:
                    cell_bits = int(bits / Nslices)
                elif params_0.core.bit_sliced.style == BitSlicedCoreStyle.BALANCED:
                    cell_bits = np.ceil((bits-1)/Nslices).astype(int)
                elif params_0.core.bit_sliced.style == BitSlicedCoreStyle.OFFSET:
                    cell_bits = np.ceil(bits/Nslices).astype(int)

        dW = np.min(np.diff(np.unique(Wm)))
        Wmin = -dW * (pow(2, bits-1)-1)
        Wmax = dW * (pow(2, bits-1)-1)

        if Ncores == 1:
            params_m.core.weight_bits = bits
            params_m.xbar.device.cell_bits = cell_bits
            params_m.core.mapping.weights.percentile = None
            params_m.core.mapping.weights.min = Wmin
            params_m.core.mapping.weights.max = Wmax
        else:
            for k in range(Ncores):
                params_m[k].core.weight_bits = bits
                params_m[k].xbar.device.cell_bits = cell_bits
                params_m[k].core.mapping.weights.percentile = None
                params_m[k].core.mapping.weights.min = Wmin
                params_m[k].core.mapping.weights.max = Wmax

    return params_m