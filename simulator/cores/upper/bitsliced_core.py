#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from __future__ import annotations

import math
import logging
from typing import Callable

import numpy as np
import numpy.typing as npt

from simulator.cores.interfaces.icore_internal import ICore, ICoreInternal
from simulator.backend.compute import ComputeBackend
from simulator.parameters.crosssim import CrossSimParameters
from simulator.parameters.core.upper_core import (
    BitslicedCoreParameters,
)
from simulator.parameters.mapping import CoreMappingParameters

xp: np = ComputeBackend()
log = logging.getLogger(__name__)


class BitslicedCore(ICoreInternal):
    """Combines multiple cores by assigning individual weight slices."""

    def __init__(
        self,
        xsim_parameters: CrossSimParameters,
        core_parameters: BitslicedCoreParameters,
        parent: ICore | None = None,
        key: str | None = None,
    ):
        """Initialize a core object.

        xsim_parameters: Parameters for the entirety of CrossSim
        core_parameters: Parameters for the initialization of a specific core
        parent: Parent core to the core, if applicable
        key: The core's key in the parent's subcore dictionary.
        """
        self.slice_sizes = core_parameters.slice_size
        self.quantize_weights = core_parameters.quantize_weights
        self.signed = core_parameters.signed
        super().__init__(
            xsim_parameters=xsim_parameters,
            core_parameters=core_parameters,
            parent=parent,
            key=key,
        )
        self.matrix_row_sum_map = {}
        self.matrix_col_sum_map = {}

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
        # TODO: BitslicedCore needs an option for 'unsigned'
        #       which behaves in the following way:
        #       Unsigned = False, subcores are unsigned:
        #           If the subcores are unsigned, then the weight mapping
        #           should be offset (e.g. [-128, 127]), when setting the matrix
        #           it needs to be scaled appropriately
        #       Unsigned = True, subcores are unsigned:
        #           Slices can be done as normal
        #       Subcores are signed:
        #           Error if unsigned = True?, otherwise normal
        matrix = xp.array(matrix)
        self._shape = matrix.shape
        if self.quantize_weights:
            matrix = xp.rint(matrix)
        if self.mapping.weights.clipping:
            matrix = matrix.clip(self.mapping.weights.min, self.mapping.weights.max)
        absmatrix = xp.abs(matrix)
        sign = xp.sign(matrix)
        # Be careful to make sure the keys are sorted in increasing order
        subcores = sorted(self.subcores.items(), key=lambda t: t[0])
        for key, subcore in subcores:
            slice_mapping = self._create_slice_mapping(key=key)
            absmatrix, subcore_weights = xp.divmod(
                absmatrix,
                self.slice_sizes[key],
            )
            subcore_weights *= sign
            subcore_weights = self.scale_weights(
                weights=subcore_weights,
                source=slice_mapping.weights,
                target=subcore.mapping.weights,
            )
            self.matrix_col_sum_map[key] = subcore_weights.sum(axis=0)
            self.matrix_row_sum_map[key] = subcore_weights.sum(axis=1)
            subcore.set_matrix(
                matrix=subcore_weights,
                apply_errors=apply_errors,
                error_mask=error_mask,
            )
        if xp.any(absmatrix > 0):
            raise RuntimeError("Not all values have been sliced.")

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
        output = None
        slice_sizes = self.slice_sizes

        for key, subcore in self.subcores.items():
            slice_mapping = self._create_slice_mapping(key=key)
            subcore_output = subcore.read_matrix(apply_errors=apply_errors)
            subcore_output = self.scale_weights(
                weights=subcore_output,
                source=subcore.mapping.weights,
                target=slice_mapping.weights,
            )
            if output is None:
                output = xp.zeros_like(subcore_output)
            product = math.prod(slice_sizes[0:key])
            output += subcore_output * product
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
        output = self._run_xbar_operation(
            xbar_func=lambda k: self.subcores[k].run_xbar_mvm,
            scaling_function=lambda k: self.subcores[k].scale_mvm_output,
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
        output = self._run_xbar_operation(
            xbar_func=lambda k: self.subcores[k].run_xbar_vmm,
            scaling_function=lambda k: self.subcores[k].scale_vmm_output,
        )
        return output

    def _run_xbar_operation(
        self,
        xbar_func: Callable,
        scaling_function: Callable,
    ) -> npt.NDArray:
        """Generic function to perform either an MVM or VMM operation.

        Args:
            xbar_func: Function to get xbar operation for a specified subcore.
            scaling_function: Function to perform xbar output scaling.

        Returns:
            npt.NDArray: Result of the core's xbar operation
        """
        output = None
        slice_sizes = self.slice_sizes
        for key, subcore in self.subcores.items():
            slice_mapping = self._create_slice_mapping(key=key)
            subcore_xbar_func = xbar_func(key)
            subcore_xbar_output = subcore_xbar_func()
            self.matrix_col_sum = self.matrix_col_sum_map[key]
            self.matrix_row_sum = self.matrix_row_sum_map[key]
            scaling_func = scaling_function(key)
            subcore_output = scaling_func(
                x=subcore_xbar_output,
                source=subcore.mapping,
                target=slice_mapping,
            )
            if output is None:
                output = xp.zeros_like(subcore_output)
            product = math.prod(slice_sizes[0:key])
            output += subcore_output * product
        return output

    def _create_slice_mapping(self, key):
        mapping = self.mapping.copy()
        signed = self.mapping.weights.min != 0
        slice_max = self.slice_sizes[key]
        if self.quantize_weights:
            mapping.weights.max = slice_max - 1
            mapping.weights.min = -1 * mapping.weights.max * signed
        else:
            mapping.weights.max = slice_max
            mapping.weights.min = -1 * mapping.weights.max * signed
        return mapping

    def generate_mapping(self) -> CoreMappingParameters:
        """Returns the default mapping of the core."""
        signed = self.signed
        w_max = math.prod(self.slice_sizes) - 1
        w_min = -1.0 * w_max * signed
        mapping_defaults = CoreMappingParameters(
            weights={"min": w_min, "max": w_max, "clipping": self.clipping},
            mvm={"min": -1.0, "max": 1.0, "clipping": self.clipping},
            vmm={"min": -1.0, "max": 1.0, "clipping": self.clipping},
        )
        return mapping_defaults
