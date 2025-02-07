#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
import logging

import numpy as np
import numpy.typing as npt

from simulator.backend.compute import ComputeBackend
from simulator.cores.interfaces.icore_lower import ICoreLower
from simulator.parameters.mapping import (
    CoreMappingParameters,
    MappingParameters,
)
from simulator.cores.lower.utils import _SingleOperationCore

xp: np = ComputeBackend()
log = logging.getLogger(__name__)


class UnsignedCore(ICoreLower):
    """Simplest possible lower core to handle strictly non-negative weights.

    An unsigned core contains a single subcore with the key "pos"
    """

    subcore_names = ["pos"]

    def set_matrix(
        self,
        matrix: npt.ArrayLike,
        apply_errors: bool = True,
        error_mask: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
        """Sets the matrix that the core will use.

        Args:
            matrix: Matrix value to set.
            apply_errors: Whether to apply errors when setting the matrix.
                This option is independent of the "enable" option for the
                models found in DeviceParameters. Defaults to True.
            error_mask: Boolean mask with the same shape as matrix to indicate
                which values of the matrix should have errors applied.
        """
        matrix = xp.asarray(matrix)
        self._shape = matrix.shape
        if self.mapping.weights.clipping:
            matrix = matrix.clip(self.mapping.weights.min, self.mapping.weights.max)
        self.matrix_col_sum = matrix.sum(axis=0)
        self.matrix_row_sum = matrix.sum(axis=1)
        normalized_matrix = self.scale_weights(
            weights=matrix,
            source=self.mapping.weights,
            target=self.subcores["pos"].mapping.weights,
        )
        self.subcores["pos"].dac.set_limits(normalized_matrix)
        self.subcores["pos"].adc.set_limits(normalized_matrix)
        self.subcores["pos"].set_matrix(
            matrix=normalized_matrix,
            apply_errors=apply_errors,
            error_mask=error_mask,
        )

    def read_matrix(self, apply_errors: bool = True) -> npt.NDArray:
        """Read the matrix set by simulation.

        Note that if the matrix was written with errors enabled, then reading
        with apply_errors=False may still produce a matrix different than the
        value it was originally set with.

        Args:
            apply_errors: If True, the matrix will be read using the error model
                that was configured. If False, the matrix will be read without
                using the error models for reading the matrix. Defaults ot True.
        """
        crange = self.mapping.weights.range
        gmin = self.subcores["pos"].mapping.weights.min
        grange = self.subcores["pos"].mapping.weights.range
        matrix = self.subcores["pos"].read_matrix(apply_errors=apply_errors)
        return (matrix - gmin) * crange / grange

    def run_xbar_operation_unsliced(self, core: _SingleOperationCore) -> npt.NDArray:
        """Performs matrix operation for unsliced MVM or VMM.

        Args:
            core: Contains objects for single core operations.

        Returns:
            npt.NDArray: Digital result of the MVM or VMM operation.
        """
        # Enumerate each possible case
        # Probably the easiest way to keep the code readable
        # (Even if some lines are repeated across cases)
        grange = self.subcores["pos"].mapping.weights.range
        gmin = self.subcores["pos"].mapping.weights.min

        core.vector["pos"] = core.dac["pos"].convert(core.vector["pos"])
        result = core.multiply["pos"](vector=core.vector["pos"])
        result = result.clip(-self.Icol_limit, self.Icol_limit)
        result = core.adc["pos"].convert(result)

        # Perform necessary transformations to get logical value
        # from the numeric core's output. This transformation is
        # dependent on the transformation used by the lower core
        result -= gmin * core.correcting_sum
        result /= grange
        return result

    def run_xbar_operation_input_bitsliced(
        self,
        core: _SingleOperationCore,
    ) -> npt.NDArray:
        """Performs either MVM/VMM using input bitslicing.

        Args:
            core: Contains objects for single core operations.

        Returns:
            npt.NDArray: Result of the MVM or VMM operation.
        """
        _backup_vec_mvm = self.subcores["pos"].vector_mvm
        _backup_vec_vmm = self.subcores["pos"].vector_vmm
        grange = self.subcores["pos"].mapping.weights.range
        gmin = self.subcores["pos"].mapping.weights.min

        result = None
        for pos_slice in core.dac["pos"].convert_sliced(core.vector["pos"]):
            output_k = core.multiply["pos"](
                vector=pos_slice.islice,
            ).clip(-self.Icol_limit, self.Icol_limit)
            output_k = core.adc["pos"].convert(output_k)
            output_k *= pos_slice.correction_factor

            if result is None:
                result = xp.zeros_like(output_k)
            result += output_k * (2**pos_slice.idx)
        result -= gmin * core.correcting_sum
        result /= grange
        self.subcores["pos"].vector_mvm = _backup_vec_mvm
        self.subcores["pos"].vector_vmm = _backup_vec_vmm
        return result

    @staticmethod
    def scale_weights(
        weights: npt.ArrayLike,
        source: MappingParameters,
        target: MappingParameters,
    ) -> npt.NDArray:
        """Scales matrix weights appropriately to a child core.

        Args:
            weights: Weights to be scaled
            source: Mapping parameters of the source data
            target: Mapping parameters of the target data

        Returns:
            npt.NDArray: Scaled representation of weights
        """
        weights = xp.asarray(weights) / source.max
        gmin = target.min
        grange = target.range
        normalized_weights = gmin + grange * weights
        return normalized_weights

    def generate_mapping(self) -> CoreMappingParameters:
        """Return the core's default mapping parameters."""
        mapping = CoreMappingParameters(
            weights={"min": 0.0, "max": 1.0, "clipping": self.clipping},
            mvm={"min": -1.0, "max": 1.0, "clipping": self.clipping},
            vmm={"min": -1.0, "max": 1.0, "clipping": self.clipping},
        )
        return mapping
