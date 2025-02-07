#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from __future__ import annotations

from typing import Any
from dataclasses import dataclass

from .parameters import BaseParameters


# This matching behavior is ugly, it needs to go.
@dataclass(repr=False)
class BasePairedParameters(BaseParameters):
    """Base class for paired parameters.

    Attributes:
        _match: Whether or not to sync parameters
        mvm: Matrix-vector-multiply variant of the paired parameter
        vmm: Vector-matrix-multiply variant of the paired parameter

    Raises:
        ValueError: if match is True, but mvm and vmm are not equal
    """

    _match: bool
    mvm: BaseParameters
    vmm: BaseParameters

    def __post_init__(self):
        """Initializes a paired parameter and syncs them if necessary."""
        super().__post_init__()
        if self.match:
            self.vmm = self.mvm

    def __setattr__(self, name: str, value: Any) -> None:
        """Sets an attribute to the dataclass.

        Behaves normally except when assigning to a registered CrossSim class.
        In which case, an attempt will be made to convert the value to the
        expected CrossSim class.

        If self.match is True, then any value set to the 'mvm' key will also be
        set on the 'vmm' key, and vice versa.

        Args:
            name: Name of attribute to set
            value: Value of attribute to set
        """
        if name in ["mvm", "vmm"] and self.match:
            if isinstance(value, BaseParameters):
                value._parent = self
            super().__setattr__("mvm", value)
            super().__setattr__("vmm", value)
        return super().__setattr__(name, value)

    def validate(self) -> None:
        """Validates settings provided to the paired parameters.

        Raises:
            ValueError: Raised if set to match but mvm and vmm are not equal.
        """
        super().validate()
        if self._match and (self.mvm != self.vmm):
            raise ValueError(
                f"{self.__class__} parameters set to match, "
                "but different values given.",
            )

    @property
    def match(self) -> bool:
        """Flag indicating if paired parameters should sync."""
        return self._match

    @match.setter
    def match(self, value: bool) -> None:
        """Sets the match attribute.

        When the attribute is changed the paired parameters will sync/desync
        """
        # QUESTION2: I'm replicating the behavior defined in the original
        #            OperationPairBase
        #            Do we need this functionality? Are we (anyone?) ever going
        #            to modify `match` half way through? I don't know anywhere
        #            we do it in `core`, is this an actual use case? Or just an
        #            unused 'feature'?
        self._match = value
        if self._match:
            self.vmm = self.mvm
        else:
            self.vmm = self.vmm.copy()
