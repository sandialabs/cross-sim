#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from __future__ import annotations
import logging

import numpy as np
import numpy.typing as npt

from simulator.cores.interfaces.icore_internal import ICoreInternal
from simulator.backend.compute import ComputeBackend
from simulator.parameters.mapping import CoreMappingParameters

xp: np = ComputeBackend()
log = logging.getLogger(__name__)


class OffsetCore(ICoreInternal):
    """Turns underlying unsigned cores into signed cores via digital offset."""

    def set_matrix(
        self,
        matrix: npt.ArrayLike,
        apply_errors: bool = True,
        error_mask: npt.ArrayLike | None = None,
    ):
        """Sets the matrix that the core will use.

        Args:
            matrix: Matrix value to set.
            apply_errors: Whether to apply errors when setting the matrix.
                This option is independent of the "enable" option for the
                models found in DeviceParameters. Defaults to True.
            error_mask: Boolean mask with the same shape as matrix to indicate
                which values of the matrix should have errors applied.
                Defaults to None.
        """
        self._shape = matrix.shape
        if self.mapping.weights.clipping:
            matrix = matrix.clip(self.mapping.weights.min, self.mapping.weights.max)
        self.matrix_col_sum = matrix.sum(axis=0)
        self.matrix_row_sum = matrix.sum(axis=1)
        scaled_matrix = self.scale_weights(
            weights=matrix,
            source=self.mapping.weights,
            target=self.subcores[0].mapping.weights,
        )
        self.subcores[0].set_matrix(
            matrix=scaled_matrix,
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
        output = self.scale_weights(
            weights=self.subcores[0].read_matrix(apply_errors=apply_errors),
            source=self.subcores[0].mapping.weights,
            target=self.mapping.weights,
        )
        return output

    def run_xbar_mvm(self, vector: npt.ArrayLike | None = None) -> npt.NDArray:
        """Simulates a matrix vector multiplication using the crossbar.

        Args:
            vector: Vector to use. If no vector is specified then the input
                vector for mvm currently set is used instead. Defaults to None.

        Returns:
            npt.NDArray: Result of the matrix vector multiply using the crossbar
        """
        if vector is not None:
            self.set_mvm_inputs(vector=vector)
        result = self.subcores[0].run_xbar_mvm()
        output = self.subcores[0].scale_mvm_output(
            x=result,
            source=self.subcores[0].mapping,
            target=self.mapping,
        )
        return output

    def run_xbar_vmm(self, vector: npt.ArrayLike | None = None) -> npt.NDArray:
        """Simulates a vector matrix multiplication using the crossbar.

        Args:
            vector: Vector to use. If no vector is specified then the input
                vector for mvm currently set is used instead. Defaults to None.

        Returns:
            npt.NDArray: Result of the matrix vector multiply using the crossbar
        """
        if vector is not None:
            self.set_vmm_inputs(vector=vector)
        result = self.subcores[0].run_xbar_vmm()
        output = self.subcores[0].scale_vmm_output(
            x=result,
            source=self.subcores[0].mapping,
            target=self.mapping,
        )
        return output

    def generate_mapping(self) -> CoreMappingParameters:
        """Returns the default mapping of signed cores."""
        # TODO: Add some params to control these values.
        #       At the moment, this core is still pretty useless.
        mapping_defaults = CoreMappingParameters(
            weights={"min": -128.0, "max": 127.0, "clipping": self.clipping},
            mvm={"min": -1.0, "max": 1.0, "clipping": self.clipping},
            vmm={"min": -1.0, "max": 1.0, "clipping": self.clipping},
        )
        return mapping_defaults
