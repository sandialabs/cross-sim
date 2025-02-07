#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""Defines an interface for Core objects."""

from __future__ import annotations

import logging
from abc import abstractmethod, ABC

import numpy as np
import numpy.typing as npt

import simulator.cores.utils as core_utils
from simulator.parameters.crosssim import CrossSimParameters
from simulator.parameters.core.core import (
    CoreParameters,
)
from simulator.parameters.mapping import (
    CoreMappingParameters,
    MappingParameters,
)
from simulator.backend.registry import register_subclasses
from simulator.backend.compute import ComputeBackend

xp: np = ComputeBackend()
log = logging.getLogger(__name__)


@register_subclasses
class ICore(ABC):
    """Iterface for MVM/VMM capable cores."""

    params: CrossSimParameters
    core_params: CoreParameters
    parent: ICore | None = None
    subcores: dict[str, ICore] | None = None
    mvm_input_col_sum: npt.NDArray | None = None
    vmm_input_row_sum: npt.NDArray | None = None
    matrix_row_sum: npt.NDArray | None = None
    matrix_col_sum: npt.NDArray | None = None

    def __init__(
        self,
        xsim_parameters: CrossSimParameters,
        core_parameters: CoreParameters,
        parent: ICore | None = None,
        key: str | None = None,
    ):
        """Initialize a core object.

        Args:
            xsim_parameters: Parameters for the entirety of CrossSim
            core_parameters: Parameters for the initialization of the core
            parent: Parent core to the core, if applicable
            key: The core's key in the parent's subcore dictionary.
        """
        if core_parameters.root is not xsim_parameters:
            raise ValueError(
                "core_parameters must be a child of the CrossSimParameters object",
            )
        self.params = xsim_parameters.copy()
        core_params_path = core_parameters.get_path_from_root()
        self.core_params = self.params[core_params_path]
        self.parent = parent
        self.key = key
        core_utils.check_logging(
            ignore_check=self.params.simulation.ignore_logging_check,
        )
        self._shape: tuple[int, int] = (0, 0)
        self._clipping = core_parameters.clipping
        self._mapping = self.generate_mapping()
        ComputeBackend(
            use_gpu=self.params.simulation.useGPU,
            gpu_num=self.params.simulation.gpu_id,
        )
        self.subcores = core_utils.make_subcores(
            xsim_parameters=self.params,
            core_parameters=self.core_params,
            parent=self,
        )

    @property
    def identifier(self) -> str:
        """Returns a name for the core."""
        if self.parent is None or self.parent.identifier is None:
            return f"{self.key}"
        return f"{self.parent.identifier}.{self.key}"

    @property
    def clipping(self) -> bool:
        """Returns the clipping behavior of the core."""
        return self._clipping

    @clipping.setter
    def clipping(self, clipping: bool) -> bool:  # noqa: F811
        """Sets the clipping behavior of the core."""
        if not isinstance(clipping, bool):
            raise TypeError("Clipping must be a bool.")
        mapping = self._mapping
        mapping.weights.clipping = clipping
        mapping.mvm.clipping = clipping
        mapping.vmm.clipping = clipping
        self._mapping = mapping

    @abstractmethod
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
        # When implementing this method, remember to do the following:
        # 1. Set the core's _shape to the appropriate size (matrix.shape)
        # 2. If implementing an internal core, calculate the matrix's row sums
        #    *before scaling to subcores* (matrix.sum(axis=1))
        # 3. Process the matrix according to relevant logic of core type
        # 4. Scale resulting matrix(s) to mapping of each appropriate subcore
        # 5. Set the matrix for each respective subcore
        raise NotImplementedError

    @abstractmethod
    def read_matrix(self, apply_errors: bool = True):
        """Read the matrix set by simulation.

        Note that if the matrix was written with errors enabled, then reading
        with apply_errors=False may still produce a matrix different than the
        value it was originally set with.

        Args:
            apply_errors: If True, the matrix will be read using the error model
                that was configured. If False, the matrix will be read without
                using the error models for reading the matrix. Defaults ot True.
        """
        raise NotImplementedError

    @property
    def mapping(self) -> CoreMappingParameters:
        """Returns the mapping parameters for the core."""
        return self._mapping

    @property
    def shape(self) -> tuple[int, int]:
        """Returns the shape of the user-facing matrix."""
        return self._shape

    def scale_weights(
        self,
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
        # In most cases this is the same as ICore.scale_inputs
        # However it is different for lower cores which have an
        # implementation specific mapping onto the NumericCore.
        return ICore.scale_input(
            x=weights,
            source=source,
            target=target,
        )

    @staticmethod
    def scale_input(
        x: npt.ArrayLike,
        source: MappingParameters,
        target: MappingParameters,
    ) -> npt.NDArray:
        """Scales values between two provided ranges.

        Args:
            x: Array like value to scale
            source: Mapping parameters of the source data
            target: Mapping parameters of the target data

        Returns:
            npt.NDArray: Scaled data
        """
        x = xp.asarray(x)
        if source.clipping:
            x = x.clip(source.min, source.max)
        source_offset = -1 * source.min
        target_offset = -1 * target.min
        xhat = (x + source_offset) / source.range * target.range - target_offset
        if target.clipping:
            xhat = xhat.clip(target.min, target.max)
        return xhat

    def scale_mvm_output(
        self,
        x: npt.ArrayLike,
        source: MappingParameters,
        target: MappingParameters,
    ) -> npt.NDArray:
        """Scales values between two provided ranges after a mvm operation.

        Output is proportional to weight and input range.

        Args:
            x: Array like value to scale
            source: Mapping parameters of the source data
            target: Mapping parameters of the target data

        Returns:
            npt.NDArray: Scaled data
        """
        # . NOTE: target is AB
        #         source is CD
        #         self is target

        # Uses equations:
        # Range X = [-x - eps_x, x - eps_x]
        # x is the 'half width' of the valid range
        # eps_x is the offset of the range
        # min_x = -x - eps_x is the minimum valid value in the range
        # A is a M*N matrix, B is a N*1 vector
        # C is a M*N matrix, D is a N*1 vector
        # with the relations
        # 1. A = (C + min_c)/c*a + min_a
        # 2. B = (D + min_d)/d*b + min_b
        #
        # given CD (arg x of this function)
        # we can calculate AB

        # Primtives.
        M, N = self.shape  # noqa: F841
        a = target.weights.range
        b = target.mvm.range
        c = source.weights.range
        d = source.mvm.range

        min_a = target.weights.min
        min_b = target.mvm.min
        min_c = source.weights.min
        min_d = source.mvm.min

        # Useful quantities.
        ab_cd = (a * b) / (c * d)
        a_c = a / c
        b_d = b / d

        min_ca = a_c * min_c - min_a
        min_db = b_d * min_d - min_b

        C_sum = self.matrix_row_sum
        D_sum = self.mvm_input_col_sum

        # Correction related to offset inputs
        correction_1 = -1 * a_c * min_db * C_sum

        # Correction related to offset weights
        correction_2 = -1 * b_d * min_ca * D_sum

        # Matrix @ matrix multiply was used
        # Reshape to properly broadcast values
        if x.ndim == 2:
            correction_1 = correction_1.reshape(-1, 1)
            correction_2 = correction_2.reshape(1, -1)
        elif x.ndim == 3:
            correction_1 = correction_1.reshape(1, -1, 1)
            correction_2 = correction_2.reshape(correction_2.shape[0], 1, -1)
        elif x.ndim > 3:
            raise ValueError(
                "inputs greater than 3D were expected be reshaped to 3D internally",
            )

        # Correction related to interaction between
        # both offset weights and inputs
        correction_3 = N * min_ca * min_db

        result = ab_cd * x + correction_1 + correction_2 + correction_3

        return result

    def scale_vmm_output(
        self,
        x: npt.ArrayLike,
        source: MappingParameters,
        target: MappingParameters,
    ) -> npt.NDArray:
        """Scales values between two provided ranges after a vmm operation.

        Output is proportional to weight and input range.

        Args:
            x: Array like value to scale
            source: Mapping parameters of the source data
            target: Mapping parameters of the target data

        Returns:
            npt.NDArray: Scaled data
        """
        # . NOTE: target is AB
        #         source is CD
        #         self is target

        # Uses equations:
        # Range X = [-x - eps_x, x - eps_x]
        # x is the 'half width' of the valid range
        # eps_x is the offset of the range
        # min_x = -x - eps_x is the minimum valid value in the range
        # A is a M*N matrix, B is a N*1 vector
        # C is a M*N matrix, D is a N*1 vector
        # with the relations
        # 1. A = (C + min_c)/c*a + min_a
        # 2. B = (D + min_d)/d*b + min_b
        #
        # given CD (arg x of this function)
        # we can calculate AB

        # Primtives.
        M, N = self.shape  # noqa: F841
        a = target.weights.range
        b = target.vmm.range
        c = source.weights.range
        d = source.vmm.range

        min_a = target.weights.min
        min_b = target.vmm.min
        min_c = source.weights.min
        min_d = source.vmm.min

        # Useful quantities.
        ab_cd = (a * b) / (c * d)
        a_c = a / c
        b_d = b / d

        min_ca = a_c * min_c - min_a
        min_db = b_d * min_d - min_b

        C_sum = self.matrix_col_sum
        D_sum = self.vmm_input_row_sum

        # Correction related to offset inputs
        correction_1 = -1 * a_c * min_db * C_sum

        # Correction related to offset weights
        correction_2 = -1 * b_d * min_ca * D_sum

        # Matrix @ matrix multiply was used
        # Reshape to properly broadcast values
        if x.ndim == 2:
            correction_1 = correction_1.reshape(1, -1)
            correction_2 = correction_2.reshape(-1, 1)
        elif x.ndim == 3:
            correction_1 = correction_1.reshape(1, 1, -1)
            correction_2 = correction_2.reshape(correction_2.shape[0], -1, 1)
        elif x.ndim > 3:
            raise ValueError(
                "inputs greater than 3D were expected be reshaped to 3D internally",
            )

        # Correction related to interaction between
        # both offset weights and inputs
        correction_3 = M * min_db * min_ca

        result = ab_cd * x + correction_1 + correction_2 + correction_3

        return result

    def set_vmm_inputs(self, vector: npt.NDArray):
        """Sets the inputs that will be used in vector matrix multiplication.

        Args:
            vector: Input vector to set.
        """
        vector = vector.clip(self.mapping.vmm.min, self.mapping.vmm.max)
        if vector.ndim == 1:
            self.vmm_input_row_sum = vector.reshape(1, -1).sum(axis=1)
        elif vector.ndim == 2:
            self.vmm_input_row_sum = vector.sum(axis=-1).reshape(-1, 1)
        elif vector.ndim == 3:
            self.vmm_input_row_sum = vector.sum(axis=-1).reshape(vector.shape[0], -1, 1)
        else:
            raise ValueError(
                "inputs greater than 3D were expected be reshaped to 3D internally",
            )
        for _, subcore in self.subcores.items():
            scaled_vector = self.scale_input(
                x=vector,
                source=self.mapping.vmm,
                target=subcore.mapping.vmm,
            )
            subcore.set_vmm_inputs(vector=scaled_vector)

    def set_mvm_inputs(self, vector: npt.NDArray):
        """Sets the inputs that will be used in matrix vector multiplication.

        Args:
            vector: Input vector to set.
        """
        vector = vector.clip(self.mapping.mvm.min, self.mapping.mvm.max)
        if vector.ndim <= 2:
            self.mvm_input_col_sum = vector.sum(axis=0)
        elif vector.ndim == 3:
            self.mvm_input_col_sum = vector.sum(axis=-2).reshape(
                *vector.shape[:-2],
                1,
                -1,
            )
        else:
            raise ValueError(
                "inputs greater than 3D were expected be reshaped to 3D internally",
            )
        for _, subcore in self.subcores.items():
            scaled_vector = self.scale_input(
                x=vector,
                source=self.mapping.mvm,
                target=subcore.mapping.mvm,
            )
            subcore.set_mvm_inputs(vector=scaled_vector)

    @abstractmethod
    def generate_mapping(self) -> CoreMappingParameters:
        """Generate a mapping for the core."""
        raise NotImplementedError
