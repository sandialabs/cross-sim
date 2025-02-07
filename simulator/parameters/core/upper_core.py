#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from __future__ import annotations

from typing import TypeVar
from dataclasses import dataclass, field

from simulator.parameters.utils import expand_key
from simulator.parameters.core.core import CoreParameters, SubcoreParameters
from simulator.parameters.core.lower_core import (
    UnsignedCoreParameters,
)

T = TypeVar("T")


@dataclass(repr=False)
class OffsetCoreParameters(CoreParameters):
    """Parameters for describing offset core behavior.

    Args:
        core_type: Type of core associated with the core parameter
        style: Style of the offset core
    """

    core_type: str = "OffsetCore"
    subcores: SubcoreParameters = field(
        default_factory=lambda: {0: UnsignedCoreParameters()},
    )


@dataclass(repr=False)
class BitslicedCoreParameters(CoreParameters):
    """Parameters for describing bit sliced core behavior.

    Args:
        core_type: Core type to use, must be "BitslicedCore"
        slice_size: Size of each slice, as a number, not in bits
        quantize_weights: Whether to automatically quantize weights passed to
            the core. Disabling quantized weights implies a continious range for
            weights, which has implications on how values are mapped to the
            physical core. As a result the entire conductance range might not be
            utilized.
    """

    core_type: str = "BitslicedCore"
    slice_size: int | tuple[int] = None
    quantize_weights: bool = True
    signed: bool = True

    def __post_init__(self):
        """Runs after dataclass initialization."""
        super().__post_init__()
        if self.subcores is not None and not isinstance(self.slice_size, tuple):
            keys = []
            for raw_key in self.subcores.keys():
                keys.extend(expand_key(raw_key=raw_key))
            self.slice_size = tuple([self.slice_size] * len(keys))
        self.validate()

    def validate(self):
        """Checks the parameters for invalid settings."""
        super().validate()
        if self.subcores is None:
            return
        raw_keys = list(self.subcores.keys())
        keys = []
        for raw_key in raw_keys:
            keys.extend(expand_key(raw_key=raw_key))

        if len(keys) != len(set(keys)):
            raise KeyError(
                f"Expanded keys have overlapped keys:\n"
                f"Raw keys: {raw_keys}\n"
                f"Expanded keys: {keys}",
            )

        key_min, key_max, key_len = min(keys), max(keys), len(keys)

        inferred_slice_size = False
        slice_size = self.slice_size
        if isinstance(slice_size, int):
            slice_size = [slice_size]
            inferred_slice_size = True

        if key_min != 0 or (key_max - key_min + 1) != key_len:
            raise ValueError(
                "BitSlicedCoreParameters subcore keys must include all integers 0-n. ",
                f"Got keys: {keys = }",
            )
        if not all(isinstance(k, int) for k in keys):
            raise ValueError(
                "BitSlicedCoreParameters subcore keys must be integers or be able ",
                'to be expanded to integers (e.g. "0-3"). ',
                f"Got keys: {list(keys)}.",
            )
        if not all(isinstance(s, int) for s in slice_size):
            raise ValueError(
                "Slice size must be a positive integer or list of positive integers. ",
                f"Got {self.slice_size = }",
            )
        elif any(s <= 0 for s in slice_size):
            raise ValueError(
                "Slice size must be a positive integer or list of positive integers. ",
                f"Got {self.slice_size = }",
            )
        elif not inferred_slice_size and len(self.slice_size) != len(keys):
            raise ValueError(
                "Slice size length must match subcore length. ",
                f"Got {len(self.slice_size) = }, ",
                f"{len(keys) = }. ",
            )
