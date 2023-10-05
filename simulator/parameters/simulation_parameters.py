#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from __future__ import annotations

from dataclasses import dataclass

from .base_parameters import BaseParameters

from .core_parameters import CoreStyle


@dataclass(
    repr=False,
)
class SimulationParameters(BaseParameters):
    """Parameters specifying simulation behavior.

    Attributes:
        convolution (ConvolutionParameters): Convolution parameters
        analytics (AnalyticsParameters): Analytics parameters
        useGPU (bool): Whether to use GPU
        gpu_id (int): ID of GPU to use
        Niters_max_parasitics (int):  Max number of iterations for parasitic circuit
            solver (exceeding this causes model to conclude Rp is too high)
        Verr_th_mvm (float):  MVM/VMM error threshold for convergence in parasitic
            circuit model. This is in normalized input units (-1 to 1), and is proportional
            to input voltage.
        relaxation_gamma (float): Relaxation parameter for parasitic circuit solver.
            gamma < 1 implements successive under-relaxation.
            gamma > 1 implements successive over-relaxation.
        disable_fast_balanced (bool): fast_balanced implements MVM in BalancedCore or
            BitSlicedCore rather than calling the method in NumericCore for speed. Will be
            done  automatically if the params are compatible with fast_balanced, unless this
            param is true
        disable_fast_matmul (bool): fast_matmul uses matrix multiplies instead of
            MVMs when performing matmul (including convolution) operations. This is
            faster and shouldn't impact accuracy. Will be done automatically if the params
            are compatible with this method, unless this param is true
        hide_convergence_msg (bool): Hide messages related to re-trials of circuit
            simulations with a reduced relaxation parameter
    """

    convolution: ConvolutionParameters = None
    analytics: AnalyticsParameters = None
    useGPU: bool = False
    gpu_id: int = 0
    Niters_max_parasitics: int = 100
    Verr_th_mvm: float = 1e-3
    relaxation_gamma: float = 1
    disable_fast_balanced: bool = False
    disable_fast_matmul: bool = False
    hide_convergence_msg: bool = False

    @property
    def fast_balanced(self):
        if self.disable_fast_balanced:
            return False
        params = self.root
        if params.core.style != CoreStyle.BALANCED:
            return False
        if (
            params.xbar.device.read_noise.enable
            and params.xbar.device.read_noise.model != "IdealDevice"
        ):
            return False
        if params.xbar.array.parasitics.enable:
            return False
        if not params.core.balanced.subtract_current_in_xbar:
            return False
        if params.core.balanced.interleaved_posneg:
            return False
        if params.xbar.array.Icol_max > 0:
            return False
        return True

    @property
    def fast_matmul(self):
        if self.disable_fast_matmul:
            return False
        params = self.root
        if (
            params.xbar.device.read_noise.enable
            and params.xbar.device.read_noise.model != "IdealDevice"
        ):
            return False
        if params.xbar.array.parasitics.enable:
            return False
        return True


@dataclass(repr=False)
class ConvolutionParameters(BaseParameters):
    """Parameters for mapping convolutions to analog cores.

    Attributes:
        is_conv_core (bool): Flag to mark core as a convolution core
        stride (int): Stride length of the kernel
        Kx (int): Conv filter size x
        Ky (int): Conv filter size y
        Noc (int): Number of output channels
        Nic (int): Number of input channels
        x_par (int): Number of sliding window in x to pack into one MVM
        y_par (int): Number of sliding window in y to pack into one MVM
        conv_matmul (bool): Whether to implement convolutions in one shot using
            AnalogCore's matrix-matrix multiplication interface
        weight_reorder (bool): Whether to enable weight reordering in conductance
            matrix to improve performance when using sliding window packing
        bias_row (bool): Whether to have a bias row
        Nwindows (int): Total number of sliding windows per CNN input example
    """

    is_conv_core: bool = False
    stride: int = 1
    Kx: int = 3
    Ky: int = 3
    Noc: int = 1
    Nic: int = 1
    x_par: int = 1
    y_par: int = 1
    weight_reorder: bool = False
    conv_matmul: bool = False
    bias_row: bool = False
    Nwindows: int = 1


@dataclass(repr=False)
class AnalyticsParameters(BaseParameters):
    """Parameters for capturing analytics.

    Attributes:
        profile_adc_inputs (bool): Profile pre-ADC input values inside core, to be
            saved and used for optimal calibration of ADC ranges
        ntest (int): Number of images in dataset, used to allocate storage for ADC
            input profiling
    """

    profile_adc_inputs: bool = False
    ntest: int = 0
