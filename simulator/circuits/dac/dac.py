#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from .idac import IDAC
from .quantizer_dac import QuantizerDAC, SignMagnitudeDAC  # noqa: F401


class DAC(IDAC):
    """Digital to analog converter object.

    Creates an object that performs DAC operations for both MVM and VMM
    operations.
    """

    def __init__(self, dac_params, core_params):
        """Initializes an ADC using the provided parameters."""
        self.mvm = self._create_DAC(dac_params.mvm, core_params.mapping.inputs.mvm)
        self.vmm = self._create_DAC(dac_params.vmm, core_params.mapping.inputs.vmm)

    # Putting this in here in case set limits actually gets used
    def set_limits(self, matrix) -> None:
        """Sets the limits for a DAC.

        Args:
            matrix: Matrix to set limits based off of.
        """
        self.mvm.set_limits(matrix)
        self.vmm.set_limits(matrix.T)

    def convert(self, vector):
        """Converts a vector from digital value to analog values."""
        raise NotImplementedError(
            "Attemping to call DAC.convert directly, call mvm.convert or "
            "vmm.convert for the DAC model.",
        )

    def convert_sliced(self, vector, slice_size):
        """Returns an list of vectors corrosponding to analog slices."""
        raise NotImplementedError(
            "Attemping to call DAC.convert_sliced directly, call "
            "mvm.convert_sliced or vmm.convert_sliced for the DAC model.",
        )

    @staticmethod
    def _create_DAC(dac_params, core_params) -> IDAC:
        """Creates a DAC according to the specification by the DAC parameters
        Args:
            dac_params (dict[str, Any]): Parameters to describe device behavior
        Raises:
            ValueError: Raised when an unknown DAC is specified
        Returns:
            DAC: A device using the parameters listed.
        """
        dac_types = {subcls.__name__: subcls for subcls in IDAC.get_all_subclasses()}
        dac_types.pop("DAC")
        if dac_params.model not in dac_types:
            raise ValueError(
                "Invalid DAC model selected. "
                f"Either create a new DAC model or choose one of the "
                f"following {list(dac_types.keys())}",
            )
        dac = dac_types[dac_params.model](dac_params, core_params)
        return dac


class IdealDAC(IDAC):
    """An ideal DAC which performs perfect conversions."""

    def set_limits(self, matrix):
        """Sets the limits of the DAC.

        In the ideal case this is ignored.
        """
        pass

    def convert(self, vector):
        """Converts a digital vector to an analog value.

        In the ideal case this is the identity function.
        """
        return vector

    def convert_sliced(self, vector, slice_size):
        """Returns an list of vectors corrosponding to analog slices."""
        raise NotImplementedError("IdealDAC cannot do a sliced conversion")
