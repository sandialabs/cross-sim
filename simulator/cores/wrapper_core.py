#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from abc import abstractmethod, ABCMeta
from . import ICore
from simulator.parameters.core_parameters import CoreStyle
from simulator.backend import ComputeBackend

xp = ComputeBackend()


class WrapperCore(ICore, metaclass=ABCMeta):
    """Superclass for "wrapper" cores -- such as OffsetCore, BalancedCore, and BitslicedCore.
    Instances of WrapperCore are created and called by the outermost (algorithm-facing) core, which is AnalogCore.
    The subclass must implement the _wrapper_* methods.
    """

    def __init__(self, clipper_core_factory, params):
        """:param clipper_core_factory:
        :param params:
        :type params: Parameters
        :return:
        """
        self.params = params
        self.clipper_core_factory = clipper_core_factory
        self.core_params = params.core

        self.scale_wtmodel = 2 if self.params.core.style == CoreStyle.BALANCED else 1
        self.mvm_in_prefactor = 2 if self.params.xbar.dac.mvm.signed else 1
        self.vmm_in_prefactor = 2 if self.params.xbar.dac.vmm.signed else 1

        self.mvm_input_percentile_scaling = (
            self.core_params.mapping.inputs.mvm.percentile is not None
        )
        self.vmm_input_percentile_scaling = (
            self.core_params.mapping.inputs.vmm.percentile is not None
        )

    def set_matrix(self, matrix, weight_limits=None, error_mask=None):
        # The weight clipping range might not be known during initialization
        # This is because of the percentile option which determines the limits when matrix is set
        # So all the clipping limits are now set inside set_matrix()

        if weight_limits:
            self.min, self.max = weight_limits

        # compute weight scaling factor
        self.weight_scale = self.range / self.params.xbar.device.Grange_norm
        self.out_prefactor = self.weight_scale / self.scale_wtmodel

        # If we aren't using input percentile scaling we can input and output scaling
        # here. Otherwise we'll do it inside the set_***_inputs function
        if not self.vmm_input_percentile_scaling:
            self.vmm_in_scale = (
                self.vmm_in_prefactor / self.params.core.mapping.inputs.vmm.range
            )
            self.vmm_out_scale = self.weight_scale / (
                self.vmm_in_scale * self.scale_wtmodel
            )

        if not self.mvm_input_percentile_scaling:
            self.mvm_in_scale = (
                self.mvm_in_prefactor / self.params.core.mapping.inputs.mvm.range
            )
            self.mvm_out_scale = self.weight_scale / (
                self.mvm_in_scale * self.scale_wtmodel
            )

        self.nrows = matrix.shape[0]
        self.ncols = matrix.shape[1]
        return self._wrapper_set_matrix(
            matrix,
            weight_limits=None,
            error_mask=error_mask,
        )

    def set_vmm_inputs(self, vector, input_limits=None):
        if input_limits and self.vmm_input_percentile_scaling:
            self.vmm_in_scale = self.vmm_in_prefactor / (
                input_limits[1] - input_limits[0]
            )
            self.vmm_out_scale = self.out_prefactor / self.vmm_in_scale
        elif input_limits and not self.vmm_input_percentile_scaling:
            raise ValueError(
                "set_vmm_inputs received no input limits with percentile_scaling",
            )

        vector_in = vector * self.vmm_in_scale
        return self._wrapper_set_vmm_inputs(vector_in)

    def set_mvm_inputs(self, vector, input_limits=None):
        if input_limits and self.mvm_input_percentile_scaling:
            self.mvm_in_scale = self.mvm_in_prefactor / (
                input_limits[1] - input_limits[0]
            )
            self.mvm_out_scale = self.out_prefactor / self.mvm_in_scale
        elif input_limits and not self.mvm_input_percentile_scaling:
            raise ValueError(
                "set_mvm_inputs received no input limits with percentile_scaling",
            )

        vector_in = vector * self.mvm_in_scale
        return self._wrapper_set_mvm_inputs(vector_in)

    def run_xbar_vmm(self, vector=None, input_limits=None):
        # Set the vector if an argument is passed
        if vector is not None:
            self.set_vmm_inputs(vector, input_limits)

        # Run VMM and scale output
        output = self._wrapper_run_xbar_vmm()
        output *= self.vmm_out_scale
        return output

    def run_xbar_mvm(self, vector=None, input_limits=None):
        # Set the vector if an argument is passed
        if vector is not None:
            self.set_mvm_inputs(vector, input_limits)

        # Run MVM and scale output
        output = self._wrapper_run_xbar_mvm()
        output *= self.mvm_out_scale
        return output

    def _read_matrix(self):
        if self.params.simulation.useGPU:
            return self._wrapper_read_matrix().get()
        else:
            return self._wrapper_read_matrix()

    def _save_matrix(self):
        return self._wrapper__save_matrix()

    def _restore_matrix(self, matrix):
        return self._wrapper__restore_matrix(matrix)

    @abstractmethod
    def _wrapper_set_matrix(self, matrix):
        """Wrapper-specific implementation of :meth:`set_matrix`."""
        raise NotImplementedError

    @abstractmethod
    def _wrapper_set_vmm_inputs(self, vector):
        """Wrapper-specific implementation of :meth:`set_vmm_inputs`."""
        raise NotImplementedError

    @abstractmethod
    def _wrapper_set_mvm_inputs(self, vector):
        """Wrapper-specific implementation of :meth:`set_mvm_inputs`."""
        raise NotImplementedError

    @abstractmethod
    def _wrapper_run_xbar_vmm(self):
        """Wrapper-specific implementation of :meth:`run_xbar_vmm`."""
        raise NotImplementedError

    @abstractmethod
    def _wrapper_run_xbar_mvm(self):
        """Wrapper-specific implementation of :meth:`run_xbar_mvm`."""
        raise NotImplementedError

    @abstractmethod
    def _wrapper_read_matrix(self):
        """Wrapper-specific implementation of :meth:`_read_matrix`."""
        raise NotImplementedError

    @abstractmethod
    def _wrapper_save_matrix(self):
        """Wrapper-specific implementation of :meth:`_save_matrix`."""
        raise NotImplementedError

    @abstractmethod
    def _wrapper_restore_matrix(self, matrix):
        """Wrapper-specific implementation of :meth:`_restore_matrix`."""
        raise NotImplementedError
