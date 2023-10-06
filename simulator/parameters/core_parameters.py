#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from .base_parameters import BasePairedParameters, BaseParameters


class CoreStyle(IntEnum):
    """What style of core to use.

    "BALANCED" : Represent each positive or negative matrix value using the conductance difference
        of two devices
    "BITSLICED" : Split the bits of resolution of each matrix value into multiple slices. Each bit
        slice is represented by a balanced or offset cell
    "OFFSET" : Represent each positive or negative matrix value using a single device whose conductance
        is related to the value by an offset, which is later subtracted. Positive weights use
        conductances above the offset, negative weights are below
    """

    BALANCED = 1
    BITSLICED = 2
    OFFSET = 3


class BalancedCoreStyle(IntEnum):
    """What style of balanced core to use.

    "ONE_SIDED" : One of the devices is always at the lowest conductance
    "TWO_SIDED" : The sum of the two conductances is fixed
    """

    ONE_SIDED = 1
    TWO_SIDED = 2


class BitSlicedCoreStyle(IntEnum):
    """What style of core to use within each slice of a bit sliced core.

    "BALANCED" : Each bit slice is mapped to a balanced core
    "OFFSET" : Each bit slice is mapped to an offset core
    """

    BALANCED = 1
    OFFSET = 3


class OffsetCoreStyle(IntEnum):
    """How the offset to subtract in an OffsetCore or BitSlicedCore is computed.

    "DIGITAL_OFFSET" : Offset is computed digitally and subtracted from digitized MVM output
    "UNIT_COLUMN_SUBTRACTION" : A unit column (or zero-point column) is allocated inside the
        memory array. The analog MVM result of this column is the offset. This is digitized
        and subtracted from the digitized results of all other columns.
    """

    DIGITAL_OFFSET = 1
    UNIT_COLUMN_SUBTRACTION = 2


class PartitionStrategy(IntEnum):
    """How to split rows/cols across multiple physical arrays.

    "MAX": Use the maximum number of rows/cols in each array, one will be underfilled
    "EQUAL": Split rows/cols as equally as possible across all arrays
    """

    MAX = 1
    EQUAL = 2


@dataclass(repr=False)
class CoreParameters(BaseParameters):
    """Parameters to describe the behavior of a core.

    Attributes:
        style (CoreStyle): Which style of core to use
        rows_max (int): Maximum number of rows in the core
        cols_max (int): Maximum number of columns in the core
        weight_bits (int): How many weight bits to use
        complex_matrix (bool): Whether to use complex valued matrices
        complex_input (bool): Whether inputs are complex valued
        balanced (BalancedCoreParameters): Parameters to use for balanced cores
        bit_sliced (BitSlicedCoreParameters): Parameters to use for bit sliced cores
        offset (OffsetCoreParameters): Parameters to use for offset cores
        mapping (CoreMappingParameters): Parameters to use when mapping values
    """

    style: CoreStyle = CoreStyle.BALANCED
    rows_max: int = 1024
    cols_max: int = 1024
    weight_bits: int = 0
    complex_matrix: bool = False
    complex_input: bool = False
    balanced: BalancedCoreParameters = None
    bit_sliced: BitSlicedCoreParameters = None
    offset: OffsetCoreParameters = None
    mapping: CoreMappingParameters = None


@dataclass(repr=False)
class BalancedCoreParameters(BaseParameters):
    """Parameters for describing balanced core behavior.

    Args:
        style (BalancedCoreStyle): Style of the balanced core
        interleaved_posneg (bool): Whether devices that implement the positive and negative
            portions of the weight are interleaved on the same column (for MVM) or row (for VMM).
            This option only makes a difference if parasitic resistance effects are simulated.
        subtract_current_in_xbar (bool): Whether to subtract current in the crossbar
    """

    style: BalancedCoreStyle = BalancedCoreStyle.ONE_SIDED
    interleaved_posneg: bool = False
    subtract_current_in_xbar: bool = True


@dataclass(repr=False)
class BitSlicedCoreParameters(BaseParameters):
    """Parameters for describing bit sliced core behavior.

    Args:
        style (BitSlicedCoreStyle): Style of the bit sliced core
        num_slices (int): Number of bit slices in the core
    """

    style: BitSlicedCoreStyle = BitSlicedCoreStyle.BALANCED
    num_slices: int = 2


@dataclass(repr=False)
class OffsetCoreParameters(BaseParameters):
    """Parameters for describing offset core behavior.

    Args:
        style (OffsetCoreStyle): Style of the offset core
    """

    style: OffsetCoreStyle = OffsetCoreStyle.DIGITAL_OFFSET


@dataclass(repr=False)
class MappingParameters(BaseParameters):
    """Parameters for mapping outputs.

    Attributes:
        clipping (bool): Whether to use clipping or not
        min (float): Minimum value before clipping
        max (float): Maximum value before clipping
        percentile (float): Percentile to clip at, if used
    """

    clipping: bool = False
    min: float = -1.0
    max: float = 1.0
    percentile: float = None

    @property
    def range(self):
        return self.max - self.min

    def validate(self) -> None:
        super().validate()
        if self.percentile is None:
            if self.max is None and self.min is None:
                raise ValueError(
                    "All MappingParameters are None, must set either percentile or min/max",
                )
        else:
            if self.min is not None and self.max is not None:
                raise ValueError(
                    "All MappingParameters are set, either percentile or min/max must be None",
                )
            if self.max is not None or self.min is not None:
                raise ValueError("min and max must be None for percentile mapping")


@dataclass(repr=False)
class WeightMappingParameters(MappingParameters):
    """Parameters for mapping weights specifically.

    Attributes:
        row_partition_priority (list[int]): Ordered list of divisors to check for total
            row divisibility. For applications where we want to split rows in a
            specific way
        col_partition_priority (list[int]): Ordered list of divisors to check for total
            column divisibility. For applications where we want to split columns in a
            specific way
        row_partition_strategy (PartitionStrategy): How to split matrix rows across
            multiple subarrays. This is used only when no equal divisors are found with
            row_partition_priority.
        col_partition_strategy (PartitionStrategy): How to split matrix columns across
            multiple subarrays. This is used only when no equal divisors are found with
            col_partition_priority.
    """

    row_partition_priority: list[int] = field(default_factory=list)
    col_partition_priority: list[int] = field(default_factory=list)
    row_partition_strategy: PartitionStrategy = PartitionStrategy.EQUAL
    col_partition_strategy: PartitionStrategy = PartitionStrategy.EQUAL


@dataclass(repr=False)
class MatmulParameters(BasePairedParameters):
    """Parameters used to describe matrix operations.

    Attributes:
        _match (bool): Whether or not to sync mvm or vmm settings.
            If true, mvm settings take precedence.
        mvm (MappingParameters): Value mapping parameters for mvm operations
        vmm (MappingParameters): Value mapping parameters for vmm operations

    Raises:
        ValueError: if match is True, but mvm and vmm are not equal
    """

    _match: bool = True
    mvm: MappingParameters = None
    vmm: MappingParameters = None


@dataclass(repr=False)
class CoreMappingParameters(BaseParameters):
    """Parameters for mapping values inside a core.

    Args:
        weights (MappingParameters): Parameters for mapping values of weights
        inputs (MatmulParameters): Parameters for mapping values of inputs
    """

    weights: WeightMappingParameters = None
    inputs: MatmulParameters = None

    def __post_init__(self):
        if self.weights._implicitly_initialized:
            self.weights = WeightMappingParameters(
                clipping=True,
                min=None,
                max=None,
                percentile=1.0,
            )
        return super().__post_init__()
