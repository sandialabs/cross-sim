#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from simulator.cores.interfaces.icore_lower import ICore, ICoreLower
from simulator.backend.compute import ComputeBackend
from simulator.parameters.crosssim import CrossSimParameters
from simulator.parameters.core.lower_core import (
    SignedCoreParameters,
)
from simulator.parameters.core import (
    BalancedCoreStyle,  # TODO: Move this elsewhere? Here? Lower core params?
)
from simulator.parameters.mapping import (
    MappingParameters,
    CoreMappingParameters,
)
from simulator.cores.lower.utils import _SingleOperationCore, _InputBitsliceOutput
from simulator.circuits.dac.idac import _InputBitslice, _InputBitslices

xp: np = ComputeBackend()
log = logging.getLogger(__name__)


class SignedCore(ICoreLower):
    """Lower core to handle positive and negative weights.

    An signed core contains a two subcore with the keys "pos" and "neg"
    """

    subcore_names = ["pos", "neg"]

    def __init__(
        self,
        xsim_parameters: CrossSimParameters,
        core_parameters: SignedCoreParameters,
        parent: ICore | None = None,
        key: str | None = None,
    ):
        """Initialize the signed core.

        Args:
            xsim_parameters: Parameters for the entirety of CrossSim
            core_parameters: Parameters for the initialization of the core
            parent: Parent core to the core, if applicable
            key: The core's key in the parent's subcore dictionary.
        """
        # NOTE: These need to be initialized first, before the __init__!
        #       This is because the NumericCore subcores will look at the parent
        #       core to see if it is interleaved.
        #
        #       I think that is dangerous and needs a better way of passing that
        #       information to NumericCore - but defining core specific params
        #       first works fine for now.
        self.style = core_parameters.style
        self.interleaved = core_parameters.interleaved_posneg
        self.subtract_in_xbar = core_parameters.subtract_in_xbar
        self.fast_balanced = xsim_parameters.simulation.fast_balanced
        super().__init__(
            xsim_parameters=xsim_parameters,
            core_parameters=core_parameters,
            parent=parent,
            key=key,
        )

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
                models found in DeviceParameters. Defaults to True.i
            error_mask: Boolean mask with the same shape as matrix to indicate
                which values of the matrix should have errors applied.
        """
        matrix = xp.asarray(matrix)
        self._shape = matrix.shape
        if self.mapping.weights.clipping:
            matrix = matrix.clip(self.mapping.weights.min, self.mapping.weights.max)
        self.matrix_col_sum = matrix.sum(axis=0)
        self.matrix_row_sum = matrix.sum(axis=1)

        # TODO: Assumes subcores "pos" and "neg" have the same weight mapping
        #       This feels reasonable, but if not, this will get a bit more
        #       complex.
        #       @Curtis
        #       2024-11-22
        positive_weights, negative_weights = self.scale_weights(
            weights=matrix,
            source=self.mapping.weights,
            target=self.subcores["pos"].mapping.weights,  # NOTE: See TODO vvv
        )

        # TODO: Figure out the specifics of how we are setting limits
        #       Now that we are going to be passing in the conducatance matrix
        #       to the ADC to set the limits
        #
        #       My general sense is that it should go
        #       2. Perform mapping to "ideal" conductance matrix
        #       3. Set DAC limits
        #       4. Set ADC adds matrices using "actual" conductance matrix value
        #       5. ADC limit setting is deferred until all matrices are set
        #          - Probably at the end of AnalogCore.set_matrix()?
        #          - Some convenience function for this, so other could use it
        #            as necessary
        #          - Probably need an extra function on ADC add_matrix()
        #            and extra flag limits_set (bool)
        #
        #       One thing I am unsure about is if we would want to set the
        #       limits based on the "ideal" or "actual" conductance matrix
        #       (or matrices)
        #
        #       Not adding this just yet, but I will have to come back to this
        #       @Curtis
        #       2024-11-20

        self.subcores["pos"].dac.set_limits(positive_weights)
        self.subcores["neg"].dac.set_limits(negative_weights)
        self.subcores["pos"].adc.set_limits(positive_weights)
        self.subcores["neg"].adc.set_limits(negative_weights)
        self.subcores["pos"].set_matrix(
            matrix=positive_weights,
            apply_errors=apply_errors,
            error_mask=error_mask,
        )
        self.subcores["neg"].set_matrix(
            matrix=negative_weights,
            apply_errors=apply_errors,
            error_mask=error_mask,
        )

        if self.fast_balanced:
            # TODO: How does setting ADC limits relate with fast balanced?
            #       The conductance matrix used would be not possible.
            #       Is that okay if the ADC were actually using the conductance
            #       matrix?
            fast_balanced_weights = self.subcores["pos"].read_matrix(
                apply_errors=False,
            ) - self.subcores["neg"].read_matrix(apply_errors=False)
            self.subcores["pos"].dac.set_limits(fast_balanced_weights)
            self.subcores["pos"].adc.set_limits(fast_balanced_weights)
            self.subcores["pos"].set_matrix(fast_balanced_weights, apply_errors=False)

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
        grange = self.subcores["pos"].mapping.weights.range
        if self.fast_balanced:
            weights_pos = self.subcores["pos"].read_matrix(apply_errors=apply_errors)
            weights_diff = weights_pos
        else:
            weights_pos = self.subcores["pos"].read_matrix(apply_errors=apply_errors)
            weights_neg = self.subcores["neg"].read_matrix(apply_errors=apply_errors)
            weights_diff = weights_pos - weights_neg

        if self.style in [BalancedCoreStyle.ONE_SIDED, BalancedCoreStyle.TWO_SIDED]:
            matrix = weights_diff * crange / grange / 2
        else:
            raise NotImplementedError(f"Cannot read using unknown style: {self.style}")
        return matrix

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
        core.vector["pos"] = core.dac["pos"].convert(core.vector["pos"])
        core.vector["neg"] = core.dac["neg"].convert(core.vector["neg"])
        if self.interleaved:
            result = core.multiply["pos"](
                vector=core.vector["pos"],
                core_neg=self.subcores["neg"],
            ).clip(-self.Icol_limit, self.Icol_limit)
        elif self.fast_balanced:
            result_pos = core.multiply["pos"](vector=core.vector["pos"])
            result = result_pos.clip(-self.Icol_limit, self.Icol_limit)
            result = core.adc["pos"].convert(result)
        elif self.subtract_in_xbar:
            result_pos = core.multiply["pos"](vector=core.vector["pos"])
            result_neg = core.multiply["neg"](vector=core.vector["neg"])
            result = result_pos - result_neg
            result = result.clip(-self.Icol_limit, self.Icol_limit)
            result = core.adc["pos"].convert(result)
        else:
            result_pos = core.multiply["pos"](vector=core.vector["pos"])
            result_neg = core.multiply["neg"](vector=core.vector["neg"])
            result_pos = result_pos.clip(-self.Icol_limit, self.Icol_limit)
            result_neg = result_neg.clip(-self.Icol_limit, self.Icol_limit)
            result_pos = core.adc["pos"].convert(result_pos)
            result_neg = core.adc["neg"].convert(result_neg)
            result = result_pos - result_neg
        return result / grange

    def run_xbar_operation_input_bitsliced(  # noqa: C901
        self,
        core: _SingleOperationCore,
    ) -> npt.NDArray:
        """Performs either MVM/VMM using input bitslicing.

        Args:
            core: Contains objects for single core operations.

        Returns:
            npt.NDArray: Result of the MVM or VMM operation.
        """
        # Assumptions:
        # - All subcores use the same ADC/DAC configuration
        #   Currently it isn't possible to have multiple ADC's configured
        grange = self.xbar_params.device.Grange_norm
        adc_per_ibit = core.adc["pos"].adc_per_ibit
        output = None
        islice_results = {}
        islice_correction = {}

        pos_sliced: _InputBitslices = core.dac["pos"].convert_sliced(core.vector["pos"])
        neg_sliced: _InputBitslices = core.dac["neg"].convert_sliced(core.vector["neg"])
        slice_iter = zip(pos_sliced, neg_sliced, strict=True)
        for pos_slice, neg_slice in slice_iter:
            # This gets a bit complicated, maybe could find a way to refactor it
            # But this should be good as is.
            # We have four options
            # 1. adc_per_ibit
            # 2. subtract in xbar
            # 3. Whether the ADC happens before combining results
            # 4. Interleaved pos/neg
            if pos_slice.idx != neg_slice.idx:
                msg = f"Core slices not aligned. {pos_slice.idx=}, {neg_slice.idx=}"
                raise IndexError(msg=msg)

            input_slice = {
                "pos": pos_slice,
                "neg": neg_slice,
            }
            subcore_outputs_k = self._run_xbar_operation_input_single_bitslice(
                core=core,
                input_slice=input_slice,
                adc_per_ibit=adc_per_ibit,
            )

            # TODO: Clarify, how are analog islice_results aggregated?
            #
            # Do iterated approach as _run_xbar_operation_input_single_bitslice
            # may either have keys ["pos", "neg"], or ["output_k"]
            # depending on the settings used.
            for key, subcore_output_k in subcore_outputs_k.items():
                islice = subcore_output_k.islice
                corr = subcore_output_k.correction_factor
                idx = subcore_output_k.idx
                is_analog = subcore_output_k.is_analog
                if is_analog and islice_correction.setdefault(key, corr) != corr:
                    raise ValueError(
                        "Cannot combine analog bitslice outputs with different "
                        "correction factors",
                    )
                if islice_results.get(key) is None:
                    islice_results[key] = xp.zeros_like(islice)
                islice_results[key] += islice * (2**idx)

        # Produce final output depending on settings.
        if adc_per_ibit:
            output = islice_results["output"]
        elif self.fast_balanced or self.interleaved:
            output = islice_results["output"]
            output = core.adc["pos"].convert(output)
            output *= islice_correction["output"]
        elif self.subtract_in_xbar:
            output = islice_results["pos"] - islice_results["neg"]
            output = core.adc["pos"].convert(output)
            if islice_correction["pos"] != islice_correction["neg"]:
                raise ValueError(
                    "Different correction factors when combining for subtract in xbar.",
                )
            output *= islice_correction["pos"]
        elif not self.subtract_in_xbar:
            islice_results["pos"] = core.adc["pos"].convert(islice_results["pos"])
            islice_results["neg"] = core.adc["neg"].convert(islice_results["neg"])
            islice_results["pos"] *= islice_correction["pos"]
            islice_results["neg"] *= islice_correction["neg"]
            output = islice_results["pos"] - islice_results["neg"]
        return output / grange

    def _run_xbar_operation_input_single_bitslice(
        self,
        core: _SingleOperationCore,
        input_slice: dict[str, _InputBitslice],
        adc_per_ibit: bool,
    ) -> dict[Any, _InputBitsliceOutput]:
        """Performs either MVM/VMM for a single bit slice.

        Args:
            core: Contains object for single core operations
            input_slice: Input slice to perform multiply with
            adc_per_ibit: If true, use adc on each input bitslicing output.

        Returns:
            dict[Any, _InputBitsliceOutput]: Dictionary result of the MVM or VMM
                operations. If the results were able to be combined, the
                dictionary contains a single key of "output". Otherwise, there
                is a key for the name of each subcore.
        """
        # Run multiplication operation
        subcore_outputs_k: dict[str, _InputBitsliceOutput] = {}
        if self.interleaved:
            result = core.multiply["pos"](
                vector=input_slice["pos"].islice,
                core_neg=self.subcores["neg"],
            ).clip(-self.Icol_limit, self.Icol_limit)
            subcore_outputs_k["output"] = _InputBitsliceOutput(
                islice=result,
                correction_factor=input_slice["pos"].correction_factor,
                idx=input_slice["pos"].idx,
                is_corrected=False,
                is_analog=True,
            )
        elif self.fast_balanced:
            result = core.multiply["pos"](
                vector=input_slice["pos"].islice,
            ).clip(-self.Icol_limit, self.Icol_limit)
            subcore_outputs_k["output"] = _InputBitsliceOutput(
                islice=result,
                correction_factor=input_slice["pos"].correction_factor,
                idx=input_slice["pos"].idx,
                is_corrected=False,
                is_analog=True,
            )
        else:
            result_pos = core.multiply["pos"](
                vector=input_slice["pos"].islice,
            ).clip(-self.Icol_limit, self.Icol_limit)
            result_neg = core.multiply["neg"](
                vector=input_slice["neg"].islice,
            ).clip(-self.Icol_limit, self.Icol_limit)
            subcore_outputs_k["pos"] = _InputBitsliceOutput(
                islice=result_pos,
                correction_factor=input_slice["pos"].correction_factor,
                idx=input_slice["pos"].idx,
                is_corrected=False,
                is_analog=True,
            )
            subcore_outputs_k["neg"] = _InputBitsliceOutput(
                islice=result_neg,
                correction_factor=input_slice["neg"].correction_factor,
                idx=input_slice["neg"].idx,
                is_corrected=False,
                is_analog=True,
            )

        # Combine results if possible
        if not adc_per_ibit:
            return subcore_outputs_k

        if self.fast_balanced or self.interleaved:
            output_k = subcore_outputs_k["output"]
            output_k.islice = core.adc["pos"].convert(output_k.islice)
            output_k.is_analog = False
            output_k.islice *= output_k.correction_factor
            output_k.is_corrected = True
        elif self.subtract_in_xbar:
            # If correction factors or idx are different,
            # not sure what to do about this
            output_k_pos = subcore_outputs_k.pop("pos")
            output_k_neg = subcore_outputs_k.pop("neg")

            if output_k_pos.correction_factor != output_k_neg.correction_factor:
                raise ValueError("Unequal correction factors during subtract in xbar")
            if output_k_pos.idx != output_k_neg.idx:
                raise ValueError("Unequal idx during subtract in xbar")

            output_k_result = output_k_pos.islice - output_k_neg.islice
            output_k_result = core.adc["pos"].convert(output_k_result)
            output_k_result *= output_k_pos.correction_factor
            output_k = _InputBitsliceOutput(
                islice=output_k_result,
                correction_factor=output_k_pos.correction_factor,
                idx=output_k_pos.idx,
                is_corrected=True,
                is_analog=False,
            )
            subcore_outputs_k["output"] = output_k
        elif not self.subtract_in_xbar:
            # Process positive core
            output_k_pos = subcore_outputs_k.pop("pos")
            output_k_neg = subcore_outputs_k.pop("neg")

            if output_k_pos.idx != output_k_neg.idx:
                raise ValueError("Cannot combine islices with different indexes.")

            output_k_pos.islice = core.adc["pos"].convert(
                vector=output_k_pos.islice,
            )
            output_k_pos.is_analog = False
            output_k_pos.islice *= output_k_pos.correction_factor
            output_k_pos.is_corrected = True

            # Process negative core
            output_k_neg.islice = core.adc["neg"].convert(
                vector=output_k_neg.islice,
            )
            output_k_neg.is_analog = False
            output_k_neg.islice *= output_k_neg.correction_factor
            output_k_neg.is_corrected = True

            # Combine core results digitally
            output_k = _InputBitsliceOutput(
                islice=output_k_pos.islice - output_k_neg.islice,
                correction_factor=1,
                idx=output_k_pos.idx,
                is_corrected=True,
                is_analog=False,
            )
            subcore_outputs_k["output"] = output_k
        return subcore_outputs_k

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
        grange = target.range
        gmin = target.min
        wmax = source.max
        weights = xp.asarray(weights) / wmax

        if self.style is BalancedCoreStyle.ONE_SIDED:
            min_resolution = 0
            cell_bits = self.xbar_params.device.cell_bits
            if cell_bits > 1:
                min_resolution = pow(2, -(cell_bits + 1))
            mat_pos = gmin + (grange * weights * (weights >= min_resolution))
            mat_neg = gmin + (grange * -weights * (weights <= -min_resolution))
        elif self.style is BalancedCoreStyle.TWO_SIDED:
            mat_pos = gmin + grange * (1 + weights) / 2
            mat_neg = gmin + grange * (1 - weights) / 2
        else:
            msg = f"Cannot map weight for unknown style: {self.style}"
            raise NotImplementedError(msg=msg)
        return xp.array([mat_pos, mat_neg])

    def generate_mapping(self) -> CoreMappingParameters:
        """Returns the default mapping of signed cores."""
        SignedCoreMappingDefaults = CoreMappingParameters(
            weights={"min": -1.0, "max": 1.0, "clipping": self.clipping},
            mvm={"min": -1.0, "max": 1.0, "clipping": self.clipping},
            vmm={"min": -1.0, "max": 1.0, "clipping": self.clipping},
        )
        return SignedCoreMappingDefaults
