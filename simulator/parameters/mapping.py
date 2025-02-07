#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import sys

import numpy as np

from simulator.parameters.base import BaseParameters, RegisteredEnum
from simulator.backend.compute import ComputeBackend

xp: np = ComputeBackend()


class PercentileMode(RegisteredEnum):
    """Way to calculate min/max values for percentile mapping."""

    SYMMETRIC = 1
    UNSIGNED = 2
    CENTERED = 3


@dataclass(repr=False)
class CoreMappingParameters(BaseParameters):
    """Parameters for mapping values.

    Attributes:
        weights: Value map for weights.
        mvm: Value map for mvm inputs.
        vmm: Value map for vmm inputs.
    """

    weights: MappingParameters = None
    mvm: MappingParameters = None
    vmm: MappingParameters = None


@dataclass(repr=False)
class MappingParameters(BaseParameters):
    """Parameters for mapping outputs.

    Attributes:
        clipping: Whether to use clipping or not
        min: Minimum value before clipping
        max: Maximum value before clipping
    """

    clipping: bool = True
    min: float = -1.0
    max: float = 1.0

    def __post_init__(self):
        """Runs after dataclass initialization."""
        if isinstance(self.min, (int, float)) and self.min == self.max:
            eps = sys.float_info.epsilon
            self.min -= eps
            self.max += eps
        return super().__post_init__()

    @property
    def range(self) -> float:
        """Range of the mapping."""
        return self.max - self.min

    @property
    def absmax(self) -> float:
        """The maximum magnitude of either the min or max."""
        return max(abs(self.max), abs(self.min))

    @property
    def midpoint(self) -> float:
        """Midpoint between the minimum and maximum mapping value."""
        return (self.min + self.max) / 2

    def validate(self) -> None:
        """Checks the parameters for invalid settings."""
        if not isinstance(self.min, (float, int)):
            raise TypeError(
                f"Attribute min on MappingParameters must be an int or float, "
                f"got {type(self.min)}",
            )
        if not isinstance(self.max, (float, int)):
            raise TypeError(
                f"Attribute max on MappingParameters must be an int or float, "
                f"got {type(self.max)}",
            )
        if not isinstance(self.clipping, bool):
            raise TypeError(
                f"Attribute clipping on MappingParameters must be a bool, "
                f"got {type(self.clipping)}",
            )
        if self.min > self.max:
            raise ValueError(
                f"Invalid mapping, {self.min = } > {self.max = }",
            )
        super().validate()


@dataclass(repr=False)
class PercentileMappingParameters(MappingParameters):
    """Parameters for mapping outputs.

    Attributes:
        clipping: Whether to use clipping or not
        min: Minimum value before clipping
        max: Maximum value before clipping
        percentile: Percentile to set bounds with
        percentile_mode: Method to calculate percentile bounds
    """

    min: float = None
    max: float = None
    percentile: float = 1.0
    percentile_mode: PercentileMode = PercentileMode.SYMMETRIC

    def update_mapping(self, value: float | Iterable[float]):
        """Updates the mapping according to a value and the percentile."""
        mapping = build_mapping_with_percentile(
            value=value,
            percentile=self.percentile,
            percentile_mode=self.percentile_mode,
        )
        self.min = mapping.min
        self.max = mapping.max

    def validate(self) -> None:
        """Validates the parameters."""
        if self.percentile is None:
            return super().validate()
        if self.min is not None or self.max is not None:
            raise ValueError("Min and max must be None when percentile is set.")


@dataclass(repr=False)
class PercentileCoreMappingParameters(CoreMappingParameters):
    """Parameters for mapping outputs.

    Attributes:
        weights: Value map for weights.
        mvm: Value map for mvm inputs.
        vmm: Value map for vmm inputs.
    """

    weights: PercentileMappingParameters = None
    mvm: PercentileMappingParameters = None
    vmm: PercentileMappingParameters = None


def build_mapping_with_percentile(
    value: float | Iterable[float],
    percentile: float,
    percentile_mode: PercentileMode,
) -> MappingParameters:
    """Returns a mapping based on a value, percentile and percentile mode."""
    if not isinstance(percentile, (float, int)):
        raise ValueError(f"Cannot update mapping bounds with '{percentile = }'")
    if percentile_mode == PercentileMode.UNSIGNED:
        max_ = float(max(xp.max(value), 0))
        min_ = 0
    elif percentile_mode == PercentileMode.SYMMETRIC:
        if percentile < 1.0:
            X_posmax = xp.percentile(value, 100 * percentile)
            X_negmax = xp.percentile(value, 100 - 100 * percentile)
            absmax = xp.max(xp.abs([X_posmax, X_negmax]))
        else:
            absmax = float(xp.max(xp.abs(value))) * percentile
        max_ = absmax
        min_ = -1 * absmax
    elif percentile_mode == PercentileMode.CENTERED:
        min_value = float(xp.min(value))
        max_value = float(xp.max(value))
        midpoint = (min_value + max_value) / 2
        width = (max_value - min_value) / 2
        if width == 0:
            width = xp.finfo(float).eps
        scaled_width = width * percentile
        max_ = float(midpoint + scaled_width)
        min_ = float(midpoint + scaled_width)
    else:
        raise NotImplementedError(
            f"Unknown percentile mode '{percentile_mode}'",
        )
    if min_ == max_:
        eps = xp.finfo(float).eps
        min_ -= eps
        max_ += eps
    mapping = MappingParameters(min=min_, max=max_)
    return mapping
