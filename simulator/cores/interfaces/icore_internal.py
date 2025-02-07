#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""Defines an interface for internal core objects."""

from __future__ import annotations

import logging
from abc import abstractmethod

import numpy.typing as npt

from simulator.cores.interfaces.icore import ICore

log = logging.getLogger(__name__)


class ICoreInternal(ICore):
    """Iterface for internal MVM/VMM capable cores."""

    @abstractmethod
    def run_xbar_vmm(self, vector: npt.NDArray | None = None) -> npt.NDArray:
        """Simulates a vector matrix multiplication using the crossbar.

        Args:
            vector: Vector to use. If no vector is specified then the input
                vector for mvm currently set is used instead. Defaults to None.

        Returns:
            npt.NDArray: Result of the matrix vector multiply using the crossbar
        """
        raise NotImplementedError

    @abstractmethod
    def run_xbar_mvm(self, vector: npt.NDArray | None = None) -> npt.NDArray:
        """Simulates a matrix vector multiplication using the crossbar.

        Args:
            vector: Vector to use. If no vector is specified then the input
                vector for mvm currently set is used instead. Defaults to None.

        Returns:
            npt.NDArray: Result of the matrix vector multiply using the crossbar
        """
        raise NotImplementedError
