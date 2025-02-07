#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
import logging
import warnings
from typing import Iterator

import numpy as np
import numpy.typing as npt

from simulator.circuits.dac.idac import IDAC, _InputBitslice
from simulator.backend import ComputeBackend

log = logging.getLogger(__name__)
xp: np = ComputeBackend()  # Represents either cupy or numpy


class IdealDAC(IDAC):
    """An ideal DAC which performs perfect conversions."""

    def set_limits(self, matrix: npt.ArrayLike):
        """Sets the limits of the DAC.

        In the ideal case this is ignored.

        Args:
            matrix: Matrix to set the limits with.
                Ignored in by an ideal DAC.
        """
        pass

    def convert(self, vector: npt.ArrayLike) -> npt.NDArray:
        """Converts a digital vector to an analog value.

        In the ideal case this is the identity function.

        Args:
            vector: Vector to be converted.

        Returns:
            npt.NDArray: Converted vector.
        """
        return xp.asarray(vector)

    def convert_sliced(
        self,
        vector: npt.ArrayLike,
        slice_size: int | None = None,
    ) -> Iterator[_InputBitslice]:
        """Converts a vector into ideal slices.

        Converts with respect to the limits set for the dac.

        Args:
            vector: Vector to be converted.
            slice_size: Size of slice to be converted.

        Raises:
            NotImplementedError: Method not implemented for ideal case.

        Returns:
            _InputBitslice: Output of the input bitslice slicing.
        """
        if slice_size is not None:
            self.slice_size = slice_size

        warnings.warn(
            "Input bitslicing with an IdealDAC is very inefficient due to high levels "
            "of precision required to match the true value.",
            category=RuntimeWarning,
            stacklevel=2,
        )
        # WARNING: This isn't meant to be efficient
        #          It is meant to at least have an implementation
        #          for an IdealDAC
        bin_values = [_float2binstr(v) for v in vector.flatten()]

        # Separate int/frac parts, determine width
        sign = xp.array([(-1) ** (v[0] == "-") for v in bin_values])
        bin_values = [v.replace("-", "") for v in bin_values]
        int_parts = [v.partition(".")[0] for v in bin_values]
        frac_parts = [v.partition(".")[-1] for v in bin_values]
        max_int_width = 0
        max_frac_width = 0
        for v in bin_values:
            int_width, _, frac_width = map(len, v.partition("."))
            max_int_width = max(int_width, max_int_width)
            max_frac_width = max(frac_width, max_frac_width)

        # Align, transpose, slice
        int_parts = [v.rjust(max_int_width, "0") for v in int_parts]
        frac_parts = [v.ljust(max_frac_width, "0") for v in frac_parts]
        processed = [f"{i}{f}" for i, f in zip(int_parts, frac_parts, strict=True)]
        processed = xp.array([[*v][::-1] for v in processed], dtype=float)
        processed = processed.T * sign
        width = 0
        output = _InputBitslice(
            islice=np.zeros_like(vector, dtype=np.float64),
            idx=0,
            correction_factor=1,
        )

        # Iterate over string of binary fraction backwards
        # in chunks of size slice_size. If the string isn't
        # long enough to fill the entire chunk, pad with 0
        idx = -len(frac_parts[0])
        it = iter(processed)
        while True:
            islice = next(it, None)
            pad_slice = islice is None
            if pad_slice:
                # Iterator has ended and is being padded with zeros
                islice = xp.zeros_like(vector)
            if pad_slice and width == 0:
                # Iterator has ended and we have just yielded
                # Break out of the loop.
                break
            output.islice = (output.islice / 2) + islice.reshape(vector.shape)
            output.idx = idx
            width += 1
            if width == self.slice_size:
                if self.slice_size > 1:
                    # When input slice widths are > 1, the range goes to [0, 2)
                    # Adjust this at the end so we renormalize to [0, 1)
                    output.islice /= 2
                    output.idx += 1
                    pass
                yield output
                output = _InputBitslice(
                    islice=np.zeros_like(vector, dtype=np.float64),
                    idx=0,
                    correction_factor=1,
                )
                width = 0
            idx += 1


def _float2binstr(value: float) -> str:
    """Converts a float value to a binary string representation.

    Args:
        value: Value to convert

    Example:
        _float2binstr(2.5) -> 10.1
        _float2binstr(8.25) -> 1000.01
        _float2binstr(-1.25) -> -1.01

    Returns:
       str: String of the fractional binary value.
    """
    sign = ""
    if value < 0:
        sign = "-"
        value *= -1
    int_val, frac_val = divmod(value, 1)
    binint = f"{int(int_val):b}"
    if frac_val == 0:
        return f"{sign}{binint}"

    frac_results = []
    while frac_val > 0:
        frac_val *= 2
        bit, frac_val = divmod(frac_val, 1)
        bit = str(int(bit))
        frac_results.append(bit)
    return f"{sign}{binint}.{''.join(frac_results)}"
