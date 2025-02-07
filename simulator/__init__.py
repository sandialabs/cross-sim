#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())


def load_registered_builtins():
    """Load all sources of registered CrossSim objects that are built-in."""
    # Importing everything is desirable because otherwise the user may run into
    # unexpected errors. For example, if they use a .json that uses an unsigned
    # core, but they only imported AnalogCore and CrossSimParameters from
    # simulator.api, they might get an error saying CrossSim could not find
    # a core by the name UnsignedCore.
    #
    # While the user could manually import each component used by the json, this
    # is both tedious and an unrealistic expectation. An end user should not
    # need to know the internal file structure to use components that come built
    # into CrossSim.

    # Import modules within the scope of the function to prevent
    # namespace polution.

    # Some modules need to be loaded prior to prevent circular dependencies.
    import simulator.cores.utils

    # Load registered object types
    import simulator.backend.converters.registered_converter  # noqa: F401
    import simulator.parameters.base  # noqa: F401
    import simulator.parameters.adc  # noqa: F401
    import simulator.parameters.core  # noqa: F401
    import simulator.parameters.dac  # noqa: F401
    import simulator.parameters.device  # noqa: F401
    import simulator.cores.interfaces.icore  # noqa: F401
    import simulator.circuits.adc.iadc  # noqa: F401
    import simulator.circuits.dac.idac  # noqa: F401
    import simulator.devices.idevice  # noqa: F401

    # Load specific instances of registered types
    import simulator.backend.converters.builtin  # noqa: F401
    import simulator.circuits.adc.models  # noqa: F401
    import simulator.circuits.dac.models  # noqa: F401

    # TODO: Think if there is a better way to
    #       grab all the core imports at once
    import simulator.cores.physical.numeric_core  # noqa: F401
    import simulator.cores.lower.signed_core  # noqa: F401
    import simulator.cores.lower.unsigned_core  # noqa: F401
    import simulator.cores.upper.bitsliced_core  # noqa: F401
    import simulator.cores.upper.offset_core  # noqa: F401

    import simulator.parameters.adc.types  # noqa: F401
    import simulator.parameters.dac.types  # noqa: F401
    import simulator.parameters.device.types  # noqa: F401

    # TODO: Think if there is a better way to
    #       grab all the core param imports at once
    import simulator.parameters.core.upper_core  # noqa: F401
    import simulator.parameters.core.lower_core  # noqa: F401
    import simulator.parameters.core.analog_core  # noqa: F401


# Python does *not* re-import modules that it already has imported
# therefore the first call to this function may take a second
# but all subsequent calls will be instant, and act as a no-op.
load_registered_builtins()
