#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from __future__ import annotations

import logging
import textwrap
import warnings
from typing import Callable, Any

import numpy as np

from simulator.parameters.crosssim import CrossSimParameters
from simulator.parameters.core.core import (
    CoreParameters,
    ConvertingConfiguration,
)
from simulator.parameters.utils import (
    expand_key,
    interpret_key,
)
from simulator.cores.interfaces.icore_internal import ICore, ICoreInternal
from simulator.cores.interfaces.icore_lower import ICoreLower
from simulator.backend.registry import RegistryManager
from simulator.backend.compute import ComputeBackend
from simulator.circuits import ADC, DAC


xp: np = ComputeBackend()
log = logging.getLogger(__name__)


def make_core(
    xsim_parameters: CrossSimParameters,
    core_parameters: CoreParameters,
    parent: ICore | None = None,
    key: str | None = None,
) -> ICoreInternal:
    """Makes creates a core according to simulation and its own parameters.

    Args:
        xsim_parameters: CrossSim parameters
        core_parameters: Parameters for the core to create.
        parent: Parent of the core to be created. Defaults to None.
        key: The core's key in the parent's subcore dictionary.

    Raises:
        ValueError: Raised when the core type cannot be found by the registry
            manager.

    Returns:
        ICore: The created core.
    """
    registry_manager = RegistryManager()
    core_registry = registry_manager.get(ICore)
    core_type = core_registry.get(core_parameters.core_type)
    if core_type is None:
        raise ValueError(
            f"Failed to make core of type {core_parameters.core_type}, "
            "was it imported?",
        )
    core = core_type(
        xsim_parameters=xsim_parameters,
        core_parameters=core_parameters,
        parent=parent,
        key=key,
    )
    add_adc_to_core(
        xsim_parameters=xsim_parameters,
        core_parameters=core_parameters,
        core=core,
    )
    add_dac_to_core(
        xsim_parameters=xsim_parameters,
        core_parameters=core_parameters,
        core=core,
    )
    return core


def make_subcores(
    xsim_parameters: CrossSimParameters,
    core_parameters: CoreParameters,
    parent: ICore | None = None,
) -> dict[Any, ICore]:
    """Creates the subcores of a given core.

    Args:
        xsim_parameters: CrossSim parameters.
        core_parameters: Core parameters to make subcores from.
        parent: Core set as parent of the subcores. Defaults to None.
        key: The core's key in the parent's subcore dictionary.

    Returns:
        A dictionary that maps each made subcore to a coordinate
    """
    cores = {}
    if core_parameters.subcores is not None:
        for raw_key, subcore_parameters in core_parameters.subcores.items():
            for key in expand_key(raw_key=raw_key):
                k = interpret_key(key)
                cores[k] = make_core(
                    xsim_parameters=xsim_parameters,
                    core_parameters=subcore_parameters,
                    parent=parent,
                    key=key,
                )
    return cores


def add_adc_to_core(
    xsim_parameters: CrossSimParameters,
    core_parameters: CoreParameters,
    core: ICore,
):
    """Creates the ADC(s) for a core to use.

    If adc_scheme is "SKIP_CONVERSION", then no action is taken.

    Args:
        xsim_parameters: CrossSim parameters
        core_parameters: Parameters of core to use
        core: Core to act on

    Raises:
        ValueError: Raised if the ADC cannot be created.
    """
    xbar_parameters = xsim_parameters.xbar.match(core.identifier)
    adc_scheme = core_parameters.adc_scheme
    if adc_scheme == ConvertingConfiguration.SKIP_CONVERSION:
        pass
    elif adc_scheme == ConvertingConfiguration.SHARED_PER_CHILD:
        # TODO: Figure out where bit slice param fits into this
        #       @Curtis, 2023-11-28
        #
        # General idea, probably best if instead of 'bitslice' param
        # we have a key parameter?
        shared_adc = ADC(
            adc_params=xbar_parameters.adc,
            dac_params=xbar_parameters.dac,
            core_params=core_parameters,
            simulation_params=xsim_parameters.simulation,
        )
        add_adc_to_children(core=core, adc_factory=lambda: shared_adc)
    elif adc_scheme == ConvertingConfiguration.UNIQUE_PER_CHILD:
        # Special care is taken to support upper cores and lower cores
        # Lower cores don't define subcore parameters because they will
        # create a numeric core from xbar parameters.
        is_lower_core = isinstance(core, ICoreLower)
        subcores = core_parameters.subcores
        if is_lower_core:
            subcores = core.subcores
        elif subcores is None:
            raise ValueError(
                "Cannot create unique ADC per child, no subcores defined.",
            )
        for raw_key, subcore_parameters in subcores.items():
            if is_lower_core:
                subcore_parameters = core.core_params
            for key in expand_key(raw_key=raw_key):
                unique_adc = ADC(
                    adc_params=xbar_parameters.adc,
                    dac_params=xbar_parameters.dac,
                    core_params=subcore_parameters,
                    simulation_params=xsim_parameters.simulation,
                    bitslice=key,
                )
                # Note: a bit repetitive to bind loop variable
                #       See: `ruff rule B023`
                add_adc_to_children(
                    core=core.subcores[key],
                    adc_factory=lambda unique_adc=unique_adc: unique_adc,
                )


def add_dac_to_core(
    xsim_parameters: CrossSimParameters,
    core_parameters: CoreParameters,
    core: ICore,
):
    """Creates the DAC(s) for a core to use.

    If dac_scheme is "SKIP_CONVERSION", then no action is taken.

    Args:
        xsim_parameters: CrossSim parameters
        core_parameters: Parameters of core to use
        core: Core to act on

    Raises:
        ValueError: Raised if the DAC cannot be created.
    """
    xbar_parameters = xsim_parameters.xbar.match(core.identifier)
    dac_scheme = core_parameters.dac_scheme
    if dac_scheme == ConvertingConfiguration.SKIP_CONVERSION:
        pass
    elif dac_scheme == ConvertingConfiguration.SHARED_PER_CHILD:
        shared_dac = DAC(
            dac_params=xbar_parameters.dac,
            core_params=core_parameters,
        )
        add_dac_to_children(core=core, dac_factory=lambda: shared_dac)
    elif dac_scheme == ConvertingConfiguration.UNIQUE_PER_CHILD:
        # Special care is taken to support upper cores and lower cores
        # Lower cores don't define subcore parameters because they will
        # create a numeric core from xbar parameters.
        is_lower_core = isinstance(core, ICoreLower)
        subcores = core_parameters.subcores
        if is_lower_core:
            subcores = core.subcores
        elif subcores is None:
            raise ValueError(
                "Cannot create unique DAC per child, no subcores defined.",
            )
        for raw_key, subcore_parameters in subcores.items():
            if is_lower_core:
                subcore_parameters = core.core_params
            for key in expand_key(raw_key=raw_key):
                unique_dac = DAC(
                    dac_params=xbar_parameters.dac,
                    core_params=subcore_parameters,
                )
                # Note: a bit repetitive to bind loop variable
                #       See: `ruff rule B023`
                add_dac_to_children(
                    core=core.subcores[key],
                    dac_factory=lambda unique_dac=unique_dac: unique_dac,
                )


def add_adc_to_children(core: ICore, adc_factory: Callable[[], ADC]) -> None:
    """Adds an ADC to each physical core that is a child to the specified core.

    Args:
        core: Parent core, all subcores will have ADC added
        adc_factory: Callable that returns an ADC for the physical cores
    """
    for physical_core in find_physical_cores(core):
        if physical_core["core"].adc is not None:
            msg = f"Existing ADC has been overwritten (path={physical_core['path']})"
            log.warning(msg=msg)
        physical_core["core"].adc = adc_factory()


def add_dac_to_children(core: ICore, dac_factory: Callable[[], DAC]) -> None:
    """Adds an DAC to each physical core that is a child to the specified core.

    Args:
        core: Parent core, all subcores will have DAC added
        dac_factory: Callable that returns a DAC for the physical cores
    """
    for physical_core in find_physical_cores(core):
        if physical_core["core"].dac is not None:
            msg = f"Existing DAC has been overwritten (path={physical_core['path']})"
            log.warning(msg=msg)
        physical_core["core"].dac = dac_factory()


def find_physical_cores(core: ICore) -> list[dict]:
    """Returns a dictionary listing all paths to physical cores.

    The returned paths may be nested to make its way through the core hierarchy.

    Args:
        core: Core to list all physical cores below it

    Returns:
        list[dict]: List where each element is a result for a physical core
            Each element is a dictionary with two keys.
            Keys:
                'path': list of keys to traverse the subcore hierarchy from
                        the provided core.
                'core': The physical core at the end of the path
    """
    if core.subcores is None:
        return [{"path": [], "core": core}]
    results = []
    for key, subcore in core.subcores.items():
        subresults = find_physical_cores(subcore)
        for subresult in subresults:
            subresult["path"] = [key, *subresult["path"]]
            results.append(subresult)
    return results


def check_logging(ignore_check: bool = False):
    """Checks if logging has been configured. If not, warns user."""
    if ignore_check is False:
        return

    logging_configured = logging.getLogger().hasHandlers()
    if not logging_configured:
        message = textwrap.dedent(
            """
            Logging is not configured, some critical warnings and error messages may not be shown.

            Logging can be enabled by adding the following to your code:

                import logging
                logging.basicConfig(level=logging.WARNING)

                # Your code here
                # ...

            To surpress this warning, set ignore_logging_check=True in simulation parameters

            For more information on how to configure the logger, please reference the
            official Python documentation:
                https://docs.python.org/3/library/logging.html
            """,  # noqa: E501
        )
        warnings.warn(message=message, category=RuntimeWarning, stacklevel=2)
