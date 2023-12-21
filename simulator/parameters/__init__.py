#
# Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government
# retains certain rights in this software.
#
# See LICENSE for full license details
#

from .base_parameters import BaseParameters, BasePairedParameters
from .core_parameters import CoreParameters
from .simulation_parameters import SimulationParameters
from .crosssim_parameters import CrossSimParameters


# TODO:
#   Some things remain for the parameter refactor (priority to be determined)
#       2. Gracefully handle optional type hints in parameters
# 			| difficulty = 8 | importance = 2 |
#       3. Allow params to be 'None' if unused
# 			| difficulty = 2 | importance = 3 |
#          - Could be done pretty simply with short circuit set_attr logic
