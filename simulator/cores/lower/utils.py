#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from __future__ import annotations

from typing import Any, Callable, TypeAlias, Protocol
from dataclasses import dataclass, field

import numpy.typing as npt

from simulator.parameters.mapping import MappingParameters
from simulator.cores.physical.numeric_core import NumericCore
from simulator.circuits import IADC, IDAC

MultiplyFunction: TypeAlias = Callable[
    [npt.ArrayLike | None, NumericCore | None],
    npt.NDArray,
]


class _MultiplyFunction(Protocol):
    """Protocol used by numeric core multiplication operations."""

    def __call__(
        self,
        vector: npt.ArrayLike | None = None,
        core_neg: NumericCore | None = None,
    ) -> npt.NDArray:
        """Simulates a matrix multiplication using the crossbar.

        Args:
            vector: Vector to use. If no vector is specified then the input
                vector for mvm/vmm currently set is used instead.
                Defaults to None.
            core_neg: For use when interleaved option is True. Performs the
                multiplication by interleaving the provided cores.

        Returns:
            npt.NDArray: Result of the matrix multiply using the crossbar
        """
        pass


@dataclass
class _SingleOperationCore:
    """Simple packing of objects and functions to simplify lower core usage."""

    # I hate this name, but its a start
    input_mapping: MappingParameters
    adc: dict[Any, IADC] = field(default_factory=dict)
    dac: dict[Any, IDAC] = field(default_factory=dict)
    multiply: dict[Any, _MultiplyFunction] = field(default_factory=dict)
    vector: dict[Any, npt.NDArray] = field(default_factory=dict)
    correcting_sum: float | npt.NDArray = 0


@dataclass
class _InputBitsliceOutput:
    """Objects for the output of a input bitslice operation."""

    islice: npt.NDArray
    correction_factor: float
    idx: int
    is_corrected: bool
    is_analog: bool
