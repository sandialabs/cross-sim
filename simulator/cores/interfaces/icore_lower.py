#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""Defines an interface for lower core objects."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

from simulator.backend.compute import ComputeBackend
from simulator.parameters.core.core import (
    CoreParameters,
)
from simulator.parameters.mapping import (
    MappingParameters,
)
from simulator.parameters.crosssim import CrossSimParameters
from simulator.cores.interfaces.icore_internal import ICore, ICoreInternal
from simulator.cores.physical.numeric_core import NumericCore
from simulator.cores.lower.utils import _SingleOperationCore

log = logging.getLogger(__name__)
xp: np = ComputeBackend()


class ICoreLower(ICoreInternal):
    """Iterface for lower MVM/VMM capable cores.

    Lower cores are distinguished in that their subcores are work in physical
    units (conductance, current, etc.), and their parents use logical units
    (numbers). The process of mapping logical values to analog values often
    novel additional steps to account for physical constraints (e.g. conducance
    can't be negative, etc.)

    To achieve this, a lower core has two primary responsilities:
    1.  Define the correspondance between logical values and physical values
    2.  Manage the conversion between logical and physical values using the
        ADCs and DACs available to the lower core.

    Note:
    While the lower core is responsible to use the ADCs/DACs, these objects
    belong to the physical core. This structuring is because, in general, it is
    possible that the underlying physical cores used might not share hardware.
    """

    subcore_names: list[Any]

    def __init__(
        self,
        xsim_parameters: CrossSimParameters,
        core_parameters: CoreParameters,
        parent: ICore | None = None,
        key: str | None = None,
    ):
        """Initialize the lower core.

        Args:
            xsim_parameters: Parameters for the entirety of CrossSim
            core_parameters: Parameters for the initialization of the core
            parent: Parent core to the core, if applicable
            key: The core's key in the parent's subcore dictionary.
        """
        super().__init__(
            xsim_parameters=xsim_parameters,
            core_parameters=core_parameters,
            parent=parent,
            key=key,
        )
        self.xbar_params = self.params.xbar.match(self.identifier)
        self.Icol_limit = self.xbar_params.array.Icol_max
        if self.Icol_limit <= 0:
            self.Icol_limit = xp.inf
        self.input_bitslicing = self.xbar_params.dac.mvm.input_bitslicing
        self.use_conv_packing = (
            self.params.simulation.convolution.x_par > 1
            or self.params.simulation.convolution.y_par > 1
        )
        self.subcores: dict[Any, NumericCore] = {
            key: NumericCore(params=xsim_parameters, parent=self, key=key)
            for key in self.subcore_names
        }

    @abstractmethod
    def scale_weights(
        self,
        weights: npt.NDArray,
        source: MappingParameters,
        target: MappingParameters,
    ) -> npt.NDArray:
        """Maps weights from a logical value to an analog value.

        In lower cores, this mapping (and it's inverse) are dependent on the
        style of lower core being modeled.

        Args:
            weights: Logical values to be mapped to analog values.
            source: Source mapping parameters.
            target: Target mapping parameters.

        Returns:
            npt.NDArray: Values to be set on the analog matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def read_matrix(self, apply_errors: bool = True) -> npt.NDArray:
        """Read the matrix set by simulation.

        In lower cores, read_matrix() is responsible for mapping the analog
        values back into logical values

        Note that if the matrix was written with errors enabled, then reading
        with apply_errors=False may still produce a matrix different than the
        value it was originally set with.

        Args:
            apply_errors: If True, the matrix will be read using the error model
                that was configured. If False, the matrix will be read without
                using the error models for reading the matrix. Defaults ot True.
        """
        raise NotImplementedError

    def run_xbar_mvm(self, vector: npt.ArrayLike | None = None) -> npt.NDArray:
        """Simulates a matrix vector multiplication using the crossbar.

        Args:
            vector: Vector to use.

        Returns:
            npt.NDArray: Result of the matrix vector multiply using the crossbar
        """
        if vector is not None:
            self.set_mvm_inputs(vector=vector)
        core = _SingleOperationCore(input_mapping=self.mapping.mvm)
        for key in self.subcores:
            core.adc[key] = self.subcores[key].adc.mvm
            core.dac[key] = self.subcores[key].dac.mvm
            core.multiply[key] = self.subcores[key].run_xbar_mvm
            core.vector[key] = self.subcores[key].vector_mvm
            core.correcting_sum = self.mvm_input_col_sum
            if core.vector[key] is None:
                raise RuntimeError("MVM input not set before MVM operation.")
        output = self.run_xbar_operation(
            core=core,
            input_bitslicing=self.input_bitslicing,
        )
        return output

    def run_xbar_vmm(self, vector: npt.ArrayLike | None = None) -> npt.NDArray:
        """Simulates a matrix vector multiplication using the crossbar.

        Args:
            vector: Vector to use.

        Returns:
            npt.NDArray: Result of the matrix vector multiply using the crossbar
        """
        if vector is not None:
            self.set_vmm_inputs(vector=vector)
        core = _SingleOperationCore(input_mapping=self.mapping.vmm)
        for key in self.subcores:
            core.adc[key] = self.subcores[key].adc.vmm
            core.dac[key] = self.subcores[key].dac.vmm
            core.multiply[key] = self.subcores[key].run_xbar_vmm
            core.vector[key] = self.subcores[key].vector_vmm
            core.correcting_sum = self.vmm_input_row_sum
            if core.vector[key] is None:
                raise RuntimeError("VMM input not set before VMM operation.")
        output = self.run_xbar_operation(
            core=core,
            input_bitslicing=self.input_bitslicing,
        )
        return output

    def run_xbar_operation(
        self,
        core: _SingleOperationCore,
        input_bitslicing: bool,
    ) -> npt.NDArray:
        """Generalized xbar_function for performing mvm and vmm operations.

        Args:
            core: Structure used to hold info for single operation core use.
            input_bitslicing: If True, operation will be done by slicing the
                input bits and combining the results.

        Returns:
            npt.NDArray: Result of the mvm or vmm operation
        """
        if input_bitslicing:
            output = self.run_xbar_operation_input_bitsliced(core=core)
        else:
            output = self.run_xbar_operation_unsliced(core=core)
        if self.use_conv_packing:
            output = self._process_conv_output(output)
        return output

    @abstractmethod
    def run_xbar_operation_unsliced(self, core: _SingleOperationCore) -> npt.NDArray:
        """Performs matrix operation for unsliced MVM or VMM.

        Args:
            core: Contains objects for single core operations.

        Returns:
            npt.NDArray: Digital result of the MVM or VMM operation.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    def _process_conv_output(self, output: npt.NDArray) -> npt.NDArray:
        """Internal function to reshape output of convolution."""
        x_par = self.params.simulation.convolution.x_par
        y_par = self.params.simulation.convolution.y_par
        output = output.reshape((x_par * y_par, len(output) // (x_par * y_par)))
        for m in range(x_par * y_par):
            output[m, 1:] = output[m, 1:] - output[m, 0]
        output = output[:, 1:].flatten()
