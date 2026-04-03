#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from .iadc import IADC

# Importing the different adc models so _create_adc works correctly
from .quantizer_adc import QuantizerADC, SignMagnitudeADC  # noqa:F401
from .ramp_adc import RampADC  # noqa:F401
from .sar_adc import SarADC  # noqa:F401
from .pipeline_adc import PipelineADC  # noqa:F401
from .cyclic_adc import CyclicADC  # noqa:F401


class ADC(IADC):
    """Analog to digital converter object.

    Creates an object that performs ADC operations for both MVM and VMM
    operations.
    """

    def __init__(
        self,
        adc_params,
        dac_params,
        core_params,
        simulation_params,
        bitslice=0,
    ):
        """Initializes an ADC using the provided parameters.

        Optimal tuning of an ADC may require additional infomation about
        the configuration of other hardware in the system. Therefore the
        DAC and core params must also be provided.

        Args:
            adc_params: Parameters to describe device behavior
            dac_params: DAC parameters used for "full precision guarantee"
            core_params: Core parameters used for "full precision guarantee"
            simulation_params: Additional parameters about the simulation
                configuration
            bitslice: Which bitslice the ADC belongs to. Defaults to 0.
        """
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
        """Sets the limits of the ADC for both MVM and VMM operations.

        Args:
            matrix: Matrix to set limit value with.
        """
        self.mvm.set_limits(matrix)
        self.vmm.set_limits(matrix.T)

    def convert(self, vector):
        """Converts values from analog to digital.

        Expects that set_limits() has been called before.

        Args:
            vector: Vector to be converted.

        Returns:
            npt.NDArray: Digital version of the provided analog vector.

        Raises:
            NotImplementedError: Raised when calling directly on the ADC.

        """
        raise NotImplementedError(
            "Attemping to call ADC.convert directly, call mvm.convert or "
            "vmm.convert for the ADC model.",
        )

    def _create_adc(
        self,
        adc_params,
        dac_params,
        core_params,
        simulation_params,
        bitslice,
    ) -> IADC:
        """Creates a ADC according to the specification by the ADC parameters.

        Returns:
            ADC: A device using the parameters listed.

        Raises:
            ValueError: Raised when an unknown read or write model is specified
        """
        adc_types = {subcls.__name__: subcls for subcls in IADC.get_all_subclasses()}
        # Remove invalid device types
        adc_types.pop("ADC")
        if adc_params.model not in adc_types:
            raise ValueError(
                "Invalid ADC model selected. "
                f"Either create a new ADC model or choose one of the following "
                f"{list(adc_types.keys())}",
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
    """An ideal ADC which performs perfect conversions."""

    def convert(self, vector):
        """Ideal case, converting is the identity function.

        Args:
            vector: Value to convert.

        Returns:
            Returns an array of the same value.
        """
        return vector

    def set_limits(self, matrix):
        """Sets limits for the ADC.

        Args:
            matrix: Unused in ideal case.
        """
        pass
