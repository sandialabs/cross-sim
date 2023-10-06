#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any
from .base_parameters import BaseParameters, BasePairedParameters


class ADCRangeLimits(IntEnum):
    """Defines how ADC range limits are used.

    "CALIBRATED" : ADC min and max are specified manually
    "MAX" : ADC limits are computed to cover the max possible range of ADC inputs,
        given the size of the array
    "GRANULAR" : ADC limits are computed so that the ADC level spacing is the minimum
        possible separation of two ADC inputs given the target resolution of weights.
        Assumes input bit slicing is used (with 1-bit DACs).
    """

    CALIBRATED = 1
    MAX = 2
    GRANULAR = 3


@dataclass(repr=False)
class XbarParameters(BaseParameters):
    """Parameters that describe the behavior of the crossbar (xbar).

    Attributes:
        device (DeviceParameters): Parameters for the device used
        array (ArrayParameters):  Parameters for the array
        adc (PairedADCParameters):  Parameters for the ADC
        dac (PairedDACParameters):  Parameters for the DAC

    Raises:
        ValueError: Raised if per input bit slicing improperly configured
        ValueError: Raised if gate input mode used without input bit slicing
    """

    # Using field default factory to avoid isssues with declar
    device: DeviceParameters = None
    array: ArrayParameters = None
    adc: PairedADCParameters = None
    dac: PairedDACParameters = None

    def validate(self) -> None:
        super().validate()
        if (self.adc.mvm.adc_per_ibit and not self.dac.mvm.input_bitslicing) or (
            self.adc.vmm.adc_per_ibit and not self.dac.vmm.input_bitslicing
        ):
            raise ValueError(
                "ADC per input bit (adc_per_ibit) requires input bit slicing",
            )
        if self.array.parasitics.gate_input and not self.dac.mvm.input_bitslicing:
            raise ValueError("Gate input mode can only be used with input bit slicing")


@dataclass(repr=False)
class DeviceParameters(BaseParameters):
    """Parameters that describe device behavior.

    Attributes:
        cell_bits (int): Programmable bit resolution of device conductance
        Rmin (float): Minimum programmable resistance of the device in ohms
        Rmax (float): Maximum programmable resistance of the device in ohms
        infinite_on_off_ratio (bool): Whether to assume infinite conductance
            On/Off ratio. If True, simulates the case of infinite Rmax.
        read_noise (WeightErrorParameters): Parameters for device read noise
        programming_error (WeightErrorParameters): Parameters for device programming error
        drift_error (DriftErrorParameters):# Parameters for device conductance drift

    Returns:
        _type_: _description_
    """

    cell_bits: int = 0
    Rmin: float = 1000
    Rmax: float = 10000
    time: int | float = 0
    infinite_on_off_ratio: bool = False
    clip_conductance: bool = False
    read_noise: WeightErrorParameters = None
    programming_error: WeightErrorParameters = None
    drift_error: WeightErrorParameters = None

    def validate(self) -> None:
        super().validate()

        # if self.cell_type is not DeviceType.NONE:
        #     if self.read_noise.model is not WeightErrorModel.NONE:
        #         self.read_noise.model = value.name
        #     if self.programming_error.model is not WeightErrorModel.NONE:
        #         self.programming_error.model = value.name
        #     if self.drift_error.model is not WeightErrorModel.NONE:
        #         self.drift_error.model = value.name

    @property
    def Gmin_norm(self) -> float:
        # Return the minimum programmable conductance of the device, normalized
        # by the maximum programmable conductance
        gmin_norm = 0
        if not self.infinite_on_off_ratio:
            gmin_norm = self.Rmin / self.Rmax
        return gmin_norm

    @property
    def Gmax_norm(self) -> float:
        # Return the maximum programmable conductance normalized by itself, which is
        # 1 by definition
        return 1

    @property
    def Grange_norm(self) -> float:
        # Return the difference between the max and min programmable resistance, normalized
        # by the max programmable resistance
        return self.Gmax_norm - self.Gmin_norm


@dataclass(repr=False)
class ArrayParameters(BaseParameters):
    """Parameters to desribe the behavior of the array.

    Attributes:
        Icol_max (float): Maximum current in a column, in units of the maximum current that
            can be drawn by a single device in the array. Any column current that exceeds
            (-Icol_max, +Icol_max) will be clipped to these bounds
        parasitics (ParasiticsParameters): Parameters for array parasitics
    """

    Icol_max: float = 0
    parasitics: ParasiticParameters = None


# NOTE why are PairedADCParameters and PairedDACParameters separate?
@dataclass(repr=False)
class PairedADCParameters(BasePairedParameters):
    """Pairs ADC parameters for MVM and VMM operations.

    Attributes:
        _match (bool): Whether to sync mvm and vmm parameters
        mvm (ADCParameters): ADC parameters for mvm operations
        vmm (ADCParameters): VMM parameters for vmm operations
    """

    _match: bool = True
    mvm: ADCParameters = None
    vmm: ADCParameters = None

    # TODO: ADC type changer for 3.0, will be replaced in 3.1
    def _change_adc_type(self):
        if (
            self.mvm.model == "RampADC"
            or self.mvm.model == "SarADC"
            or self.mvm.model == "PipelineADC"
            or self.mvm.model == "CyclicADC"
        ):
            if self.mvm.model == "RampADC":
                new_mvm = RampADCParameters()
            elif self.mvm.model == "SarADC":
                new_mvm = SarADCParameters()
            elif self.mvm.model == "PipelineADC" or self.mvm.model == "CyclicADC":
                new_mvm = PipelineADCParameters()
            new_mvm.update(self.mvm.as_dict())
            new_mvm._parent = self
            self.mvm = new_mvm
            if self._match:
                self.vmm = self.mvm
        else:
            new_mvm = ADCParameters()
            for k in new_mvm.as_dict().keys():
                setattr(new_mvm, k, getattr(self.mvm, k))
            new_mvm._parent = self
            self.mvm = new_mvm
            if self._match:
                self.vmm = self.mvm

        if not self._match:
            if (
                self.vmm.model == "RampADC"
                or self.vmm.model == "SarADC"
                or self.vmm.model == "PipelineADC"
                or self.vmm.model == "CyclicADC"
            ):
                if self.vmm.model == "RampADC":
                    new_vmm = RampADCParameters()
                elif self.vmm.model == "SarADC":
                    new_vmm = SarADCParameters()
                elif self.vmm.model == "PipelineADC" or self.vmm.model == "CyclicADC":
                    new_vmm = PipelineADCParameters()
                new_vmm.update(self.vmm.as_dict())
                new_vmm._parent = self
                self.vmm = new_vmm
            else:
                new_vmm = ADCParameters()
                for k in new_vmm.as_dict().keys():
                    setattr(new_vmm, k, getattr(self.vmm, k))
                new_vmm._parent = self
                self.vmm = new_vmm


@dataclass(repr=False)
class PairedDACParameters(BasePairedParameters):
    """Pairs DAC parameters for MVM and VMM operations.

    Attributes:
        _match (bool): Whether to sync mvm and vmm parameters
        mvm (DACParameters): DAC parameters for mvm operations
        vmm (DACParameters): VMM parameters for vmm operations
    """

    _match: bool = True
    mvm: DACParameters = None
    vmm: DACParameters = None


@dataclass(repr=False)
class ADCParameters(BaseParameters):
    """Parameters for the behavior of the analog-to-digital converter used to digitize the analog
    MVM/VMM outputs from the array.

    Attributes:
        model (str): name of the ADC model. This must match the name of a child class of IADC,
            other than "ADC"
        bits (int): bit resolution of the ADC digital output
        stochastic_rounding (bool): whether to probabilistically round an ADC input value to one
            of its two adjacent ADC levels, with a probability set by the distance to the level.
            If False, value is always rounded to the closer level.
        adc_per_ibit (bool): whether to digitize the MVM result of each input bit slice. This is
            only used if input_bitslicing = True in the associated DACParameters. If False, it is
            assumed by shift-and-add accumulation of input bits is done using analog peripheral
            circuits and only the final result is digitized.
        calibrated_range (list): the manually specified ADC min and max. This is only used if
            adc_range_option = ADCRangeLimits.CALIBRATED. If not using BITSLICED core, this must be
            a 1D array of length 2. If using BITSLICED core, this must be a 2D array with shape
            (num_slices, 2) that stores the ADC min/max for each bit slice of the core.
        adc_range_option (ADCRangeLimits): Which method is used to set ADC range limits

    Raises:
        ValueError: Raised if granular ADC is enabled with incompatible options
    """

    model: str = "IdealADC"
    bits: int = 0
    signed: bool = True
    stochastic_rounding: bool = False
    adc_per_ibit: bool = False
    calibrated_range: list = None
    adc_range_option: ADCRangeLimits = ADCRangeLimits.CALIBRATED

    # TODO: Quick little hack for swapping param objects, just till 3.1 changes
    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == "model":
            if self.parent is not self:
                self.parent._change_adc_type()

    def validate(self) -> None:
        super().validate()
        if self.adc_range_option is ADCRangeLimits.GRANULAR and not self.adc_per_ibit:
            raise ValueError(
                "Granular ADC range is only supported for digital input shift and add (adc_per_ibit)",
            )


@dataclass(repr=False)
class RampADCParameters(ADCParameters):
    """Ramp ADC specific non-ideality parameters.

    Attributes:
        gain_db (float): Open-loop gain in decibels of the operational amplifier at the
            output of the capacitive DAC (CDAC) used to generate the voltage ramp
        sigma_capacitor (float): Standard deviation of the random variability in the minimum-sized
            capacitor in the CDAC, normalized by the minimum capacitance value.
        sigma_comparator (float): Standard deviation of the random variability in the input offset
            voltage of the comparator used for ramp comparison. There is a comparator associated with
            every array column (MVM) and/or row (VMM). The offset is normalized by the reference
            voltage used for the ramp.
        symmetric_cdac (bool): Whether to use the symmetric CDAC design that treats the ADC levels as
            two's complement signed integers. If False, uses an alternative CDAC design that treats
            the ADC levels as unsigned integers.
    """

    gain_db: float = 100
    sigma_capacitor: float = 0.0
    sigma_comparator: float = 0.0
    symmetric_cdac: bool = True


@dataclass(repr=False)
class SarADCParameters(ADCParameters):
    """SAR ADC specific non-ideality parameters.

    Attributes:
        gain_db (float): Open-loop gain in decibels of the operational amplifier at the
            output of the capacitive DAC (CDAC)
        sigma_capacitor (float): Standard deviation of the random variability in the minimum-sized
            capacitor in the CDAC, normalized by the minimum capacitance value.
        sigma_comparator (float): Standard deviation of the random variability in the input offset
            voltage of the comparator. The comparator compares the analog ADC input to the analog
            CDAC output during each SAR cycle. There is a comparator for every group of ADC inputs
        split_cdac (bool): Whether to use the split capacitor CDAC design to reduce the average size
            of the capacitor in the CDAC.
        group_size (int): Number of ADC inputs that share a SAR unit. Inputs within a group use the
            same CDAC and comparator. This corresponds to the number of grouped columns (MVM) or
            grouped rows (VMM) of the array.
    """

    gain_db: float = 100
    sigma_capacitor: float = 0.0
    sigma_comparator: float = 0.0
    split_cdac: bool = True
    group_size: int = 8


@dataclass(repr=False)
class PipelineADCParameters(ADCParameters):
    """Pipeline/Cycli ADC specific non-ideality parameters.

    Attributes:
        gain_db (float): Open-loop gain in decibels of the operational amplifier used as the
            residue amplifier in a 1.5-bit stage of the pipeline ADC.
        sigma_C1 (float): Standard deviation of the random variability in capacitor C1 used to
            amplify the voltage by 2X in the 1.5-bit switched-capacitor stage. The amplification
            factor is (1 + C1/C2), where C1 = C2 in the ideal case.
        sigma_C2 (float): Standard deviation of the random variability in capacitor C2 in the
            1.5-bit ADC stage, normalized by the nominal value of C2.
        sigma_Cpar (float): Standard deviation of the random variability in the parasitic
            capacitance at the negative input of the operational amplifier in he 1.5-bit stage,
            normalized by the nominal value of C1.
        sigma_comparator (float): Standard deviation of the random variability in the input offset
            voltage of the comparators used in the 1.5-bit stages.
        group_size (int): Number of ADC inputs that share single pipeline ADC and its random
            capacitor mismatches and comparator offsets. This corresponds to the number of grouped
            columns (MVM) or grouped rows (VMM) of the array.
    """

    gain_db: float = 100
    sigma_C1: float = 0.0
    sigma_C2: float = 0.0
    sigma_Cpar: float = 0.0
    sigma_comparator: float = 0.0
    group_size: int = 8


@dataclass(repr=False)
class DACParameters(BaseParameters):
    """Parameters for the digital-to-analog converter used to quantize the input signals
    that are passed to the array.

    Attributes:
        model (DACModel): name of the model used to specify quantization behavior. This must
            match the name of a child class of IDAC, other than "DAC"
        bits (int): bit resolution of the digital input
        input_bitslicing (bool): whether to bit slice the digital inputs to the MVM/VMM
            and accumulate the results from the different input bit slices using shift-and-add operations.
        sign_bit (bool): whether the digital input is encoded using sign-magnitude representation,
            with a range that is symmetric around zero
        slice_size (int): Default slice size for input bit slicing. Can be overridden
            from within the individual cores

    Raises:
        ValueError: Raised if input bitslicing is enabled with incompatible options
    """

    model: str = "IdealDAC"
    bits: int = 0
    input_bitslicing: bool = False
    signed: bool = True
    slice_size: int = 1

    @property
    def sign_bit(self):
        return self.min < 0

    def validate(self) -> None:
        super().validate()
        if self.input_bitslicing and self.bits == 0:
            raise ValueError("Cannot use input bit slicing if inputs are not quantized")


@dataclass(repr=False)  # Maybe rename this
class WeightErrorParameters(BaseParameters):
    """Parameters for the weight error model used.

    Attributes:
        enable (bool): Flag to enable adding weight errors
        model (WeightErrorModel): Weight error model to use. This must match the name of a
            child class of IDevice, other than "Device", "EmptyDevice", and "GenericDevice"
        magnitude (float): # Standard deviation of the random conductance error that is applied
            either as programming error or read noise when using one of the generic device models.
            This is normalized either to either the maximum device conductance (NormalIndependentDevice)
            or the target device conductance (NormalProportionalDevice)
    """

    enable: bool = False
    model: str = "IdealDevice"
    magnitude: float = 0


@dataclass(repr=False)
class ParasiticParameters(BaseParameters):
    """Parameters to describe behavior of parasitics.

    Attributes:
        enable (bool): Whether to enable parasitic resistance model. For bit sliced, this indicates
            whether parasitics is enabled for ANY of the slices
        Rp_row (float): Parasitic resistance of the row metallic interconnects
        Rp_col (float): Parasitic resistince of the column metallic interconnects
        gate_input (bool): If True, no parasitic voltage drops occur on the input side
            regardless of Rp value. This implements the configuration where the input row or column
            is connected to the gate of a transistor at every cell. Because the transistor behaves
            as a switch, the input signal must be binary. That means input bit slicing must be enabled
    """

    enable: bool = False
    Rp_row: float = 0
    Rp_col: float = 0
    gate_input: bool = False

    def __post_init__(self):
        return super().__post_init__()
