#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from __future__ import annotations

from dataclasses import dataclass, field

from simulator.parameters.base import BaseParameters, RegisteredEnum
from simulator.parameters.core.core import CoreParameters
from simulator.parameters.mapping import PercentileCoreMappingParameters


class OutputDTypeStrategy(RegisteredEnum):
    """How is the output dtype of an AnalogCore determined.

    "NATIVE": Use the type promotion semantics of the backend
    "MATRIX": Use the matrix's dtype
    "INPUT": Use the inputs's dtype
    "FLOAT16": Use backends float16 (xp.float16)
    "FLOAT32": Use backends float32 (xp.float32)
    "FLOAT64": Use backends float64 (xp.float64)
    """

    NATIVE = 1
    MATRIX = 2
    INPUT = 3
    FLOAT16 = 4
    FLOAT32 = 5
    FLOAT64 = 6


@dataclass(repr=False)
class AnalogCoreParameters(CoreParameters):
    """Parameters to describe the behavior of a core.

    Attributes:
        core_type: Type of core associated with the core parameter
        max_size: Tuple of two integers (mvm_inputs, mvm_outputs) denoting the
            maximum size of a single partition. Note this convention is reversed
            from the ordering of the dimensions of a NumPy matrix.
        complex_matrix: If the core accepts complex valued weights
        complex_input: If the core accepts complex valued inputs
        output_dtype: A dtype that determines the data type of the output
            the user receives after the operations
        weight_mapping: Defines the parameters for mapping rows/cols
            across multiple physical arrays
    """

    core_type: str = "AnalogCore"
    max_partition_size: tuple[int, int] = (1024, 1024)
    complex_matrix: bool = False
    complex_input: bool = False
    output_dtype_strategy: OutputDTypeStrategy = OutputDTypeStrategy.NATIVE
    partitioning: WeightPartitionParameters = None
    mapping: PercentileCoreMappingParameters = None


class PartitionStrategy(RegisteredEnum):
    """How to split rows/cols across multiple physical arrays.

    "MAX": Use the maximum number of rows/cols in each array, one will be
        underfilled
    "EQUAL": Split rows/cols as equally as possible across all arrays
    """

    MAX = 1
    EQUAL = 2


class PartitionMode(RegisteredEnum):
    """How to organize the subcores across the partitioned matrix.

    "AUTO": User supplies one subcore, that applies to all of the
        partitioned matricies. Asserts there is only one provided
        subcore, and the key of that subcore must be 'default'.
    "COORDINATE": User defines a (row,col) coordinate for each subcore
        that maps it to the corresponding partitioned matrix. This
        asserts all of the subcore keys are in the form (int,int)
        and aren't outside of the range of the partitioned matricies.
    """

    AUTO = 1
    COORDINATE = 2


@dataclass(repr=False)
class WeightPartitionParameters(BaseParameters):
    """Parameters for partioning weights.

    Attributes:
        row_partition_priority (list[int]): Ordered list of divisors to check
            for total row divisibility. For applications where we want to split
            rows in a specific way
        col_partition_priority (list[int]): Ordered list of divisors to check
            for total column divisibility. For applications where we want to
            split columns in a specific way
        row_partition_strategy (PartitionStrategy): How to split matrix rows
            across multiple subarrays. This is used only when no equal divisors
            are found with row_partition_priority.
        col_partition_strategy (PartitionStrategy): How to split matrix columns
            across multiple subarrays. This is used only when no equal divisors
            are found with col_partition_priority.
    """

    row_partition_priority: list[int] = field(default_factory=list)
    col_partition_priority: list[int] = field(default_factory=list)
    row_partition_strategy: PartitionStrategy = PartitionStrategy.EQUAL
    col_partition_strategy: PartitionStrategy = PartitionStrategy.EQUAL
    partition_mode: PartitionMode = PartitionMode.AUTO
