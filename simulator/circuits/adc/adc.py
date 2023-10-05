#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from .iadc import IADC

# Importing the different adc models so _create_adc works correctly
from .quantizer_adc import QuantizerADC, SignMagnitudeADC
from .ramp_adc import RampADC
from .sar_adc import SarADC
from .pipeline_adc import PipelineADC
from .cyclic_adc import CyclicADC


class ADC(IADC):
    def __init__(
        self,
        adc_params,
        dac_params,
        core_params,
        simulation_params,
        bitslice=0,
    ):
        self.mvm = self._create_adc(
            adc_params.mvm,
            dac_params.mvm,
            core_params,
            simulation_params,
            bitslice,
        )
        self.vmm = self._create_adc(
            adc_params.vmm,
            dac_params.vmm,
            core_params,
            simulation_params,
            bitslice,
        )

    def set_limits(self, matrix):
        self.mvm.set_limits(matrix)
        self.vmm.set_limits(matrix.T)

    def convert(self, vector):
        raise NotImplementedError(
            "Attemping to call ADC.convert directly, call mvm.convert or vmm.convert for the ADC model.",
        )

    def _create_adc(
        self,
        adc_params,
        dac_params,
        core_params,
        simulation_params,
        bitslice,
    ) -> IADC:
        """Creates a ADC according to the specification by the ADC parameters
        Args:
                adc_params (dict[str, Any]): Parameters to describe device behavior
        Raises:
                ValueError: Raised when an unknown read or write model is specified
        Returns:
                ADC: A device using the parameters listed.
        """
        adc_types = {subcls.__name__: subcls for subcls in IADC.get_all_subclasses()}
        # Remove invalid device types
        adc_types.pop("ADC")
        if adc_params.model not in adc_types:
            raise ValueError(
                "Invalid ADC model selected."
                + f"Either create a new ADC model or choose one of the following {list(adc_types.keys())}",
            )

        adc = adc_types[adc_params.model](
            adc_params,
            dac_params,
            core_params,
            simulation_params,
            bitslice,
        )

        return adc


class IdealADC(IADC):
    def convert(self, vector):
        return vector

    def set_limits(self, input):
        pass
