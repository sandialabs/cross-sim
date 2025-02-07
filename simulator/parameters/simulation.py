#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from __future__ import annotations

from dataclasses import dataclass

from simulator.parameters.base import BaseParameters, RegisteredEnum


class MultiplicationFunction(RegisteredEnum):
    """What function to use to do the mat-mat.

    "MATMUL": Use the np.matmul function
    "DOT": Use the np.dot function
    """

    MATMUL = 1
    DOT = 2


@dataclass(
    repr=False,
)
class SimulationParameters(BaseParameters):
    """Parameters specifying simulation behavior.

    Attributes:
        convolution: Convolution parameters
        analytics: Analytics parameters
        useGPU: Whether to use GPU
        gpu_id: ID of GPU to use
        Niters_max_parasitics:  Max number of iterations for parasitic circuit
            solver (exceeding this causes model to conclude Rp is too high)
        Verr_th_mvm:  MVM/VMM error threshold for convergence in parasitic
            circuit model. This is in normalized input units (-1 to 1), and is
            proportional to input voltage.
        Verr_matmul_criterion (str): When using parasitics in matmul mode, how to
            aggregate voltage errors across different MVMs to determine convergence
        relaxation_gamma: Relaxation parameter for parasitic circuit solver.
            gamma < 1 implements successive under-relaxation.
            gamma > 1 implements successive over-relaxation.
        disable_fast_balanced: fast_balanced implements MVM in BalancedCore or
            BitSlicedCore rather than calling the method in NumericCore for
            speed. Will be done  automatically if the params are compatible with
            fast_balanced, unless this param is true
        disable_fast_matmul: fast_matmul uses matrix multiplies instead of
            MVMs when performing matmul (including convolution) operations.
            This is faster and shouldn't impact accuracy. Will be done
            automatically if the params are compatible with this method, unless
            this param is true
        hide_convergence_msg: Hide messages related to re-trials of circuit
            simulations with a reduced relaxation parameter
    """

    convolution: ConvolutionParameters = None
    analytics: AnalyticsParameters = None
    useGPU: bool = False
    gpu_id: int = 0
    Niters_max_parasitics: int = 100
    Verr_th_mvm: float = 1e-3
    Verr_matmul_criterion: str = "max_mean"
    relaxation_gamma: float = 1
    disable_fast_balanced: bool = False
    disable_fast_matmul: bool = False
    hide_convergence_msg: bool = False
    ignore_logging_check: bool = False
    multiplication_function: MultiplicationFunction = MultiplicationFunction.MATMUL

    @property
    def fast_balanced(self) -> bool:
        """Returns True if params supports fast balance, False otherwise."""
        if self.disable_fast_balanced:
            return False
        params = self.root
        read_devices = params.xbar.search("**.device.read_noise").values()
        if any(d.enable and d.model != "IdealDevice" for d in read_devices):
            return False
        if any(params.xbar.search("**.array.parasitics.enable").values()):
            return False
        if max(params.xbar.search("**.array.Icol_max").values()) > 0:
            return False
        if any(v is True for v in params.search("**.interleaved_posneg").values()):
            return False
        if any(v is True for v in params.search("**.subtract_in_xbar").values()):
            return False
        return True

    @property
    def fast_matmul(self) -> bool:
        """Returns True if params support fast matmul, False otherwise."""
        if self.disable_fast_matmul:
            return False
        params = self.root
        read_devices = params.xbar.search("**.device.read_noise").values()
        if any(d.enable and d.model != "IdealDevice" for d in read_devices):
            return False
        return True


@dataclass(repr=False)
class ConvolutionParameters(BaseParameters):
    """Parameters for mapping convolutions to analog cores.

    Attributes:
        is_conv_core: Flag to mark core as a convolution core
        stride: Stride length of the kernel
        Kx: Conv filter size x
        Ky: Conv filter size y
        Noc: Number of output channels
        Nic: Number of input channels
        x_par: Number of sliding window in x to pack into one MVM
        y_par: Number of sliding window in y to pack into one MVM
        conv_matmul: Whether to implement convolutions in one shot using
            AnalogCore's matrix-matrix multiplication interface
        weight_reorder: Whether to enable weight reordering in conductance
            matrix to improve performance when using sliding window packing
        bias_row: Whether to have a bias row
        Nwindows: Total number of sliding windows per CNN input example
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
        profile_adc_inputs: Profile pre-ADC input values inside core, to be
            saved and used for optimal calibration of ADC ranges
        ntest: Number of images in dataset, used to allocate storage for ADC
            input profiling
    """

    profile_adc_inputs: bool = False
    ntest: int = 0
