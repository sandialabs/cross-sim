#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from __future__ import annotations

import json
import os
import sys
import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pprint import pformat
from pathlib import Path
from typing import Any, Union, get_origin, get_type_hints

from simulator.configs import CONFIGS_BASE_DIR

log = logging.getLogger(__name__)


@dataclass(repr=False)
class BaseParameters:
    """Base class for CrossSim parameters.

    Attributes:
        parent (BaseParameters): Parent of this parameter in a nested parameter tree
        parents (list[BaseParameters]): List of all parameters directly above the current parameter.
            The first index of the list is the root parameter
        root (BaseParameters): Root of the parameter tree
    """

    def __new__(cls, *args, **kwargs) -> BaseParameters:
        """Create a new BaseParameters object.

        Returns:
            BaseParameters: Returns a new BaseParameters object
        """
        param = super().__new__(cls)
        # Add parent before __init__ is called, allowing us to use the
        # parameter tree during initialization
        param._parent = param
        # Flag to mark if a parameter was initialized implicitly
        # (i.e. It was automatically created because 'None' was provided during init)
        param._implicitly_initialized = False
        return param

    def __post_init__(self):
        """Runs after default dataclass initialization to validate the state of the dataclass."""
        self.validate()

    @property
    def parent(self) -> BaseParameters:
        """Returns the parent of the parameter in the parameter tree.

        Returns:
            BaseParameters: Returns the Parameters object that holds this parameter object.
                If parameter has no parent, self.parent returns itself.
        """
        return self._parent

    @property
    def parents(self) -> list[BaseParameters]:
        """Returns a list of parent of the parameters directly above the current parameter in the
            parameter tree. The first index of the list is the root, the last parameter is the
            current parameter.

        Returns:
            list[BaseParameters]: Returns the list of parameters above the current parameter in
                the parameter tree.
        """
        if self is self.root:
            return [self.root]
        return [*self.parent.parents, self]

    @property
    def root(self) -> BaseParameters:
        """Root of the parameter tree.

        Returns:
            BaseParameters: Parameter object at the base of the parameter tree
        """
        if self.parent is self:
            return self
        else:
            return self.parent.root

    def validate(self) -> None:
        """Validates the parameters. Will warn if values do not match type hints.
        Subclassed parameters may raise error if parameters provided unusable or inconsistent.
        """
        for key, value in self.__dataclass_fields__.items():
            # Handle some typing generics
            # TODO: This could be more robust, but I'm not worrying about that as I'm not sure if we want this feature
            #       - This line behaves as follows
            #       - list[int], etc. -> Checks if the object is an int. Does not check if the elements are ints
            #       - Union[int, float], etc. -> This is not handled and it will cause an error
            # If we want to keep this type checking we might want to just wrap this in a try/except
            # This type validation could also probably be moved inside __setattr__ which would make it automatic
            value_type = _resolve_type(value.type, self.__module__)
            if not isinstance(getattr(self, key), value_type):
                message = f"Key {key} must be of type '{value.type}', got {type(getattr(self, key))}"
                log.debug(message)

    def copy(self) -> BaseParameters:
        """Returns a deep copy of the parameter.

        Returns:
            Parameters: A deep copy of the original parameter object
        """
        return deepcopy(self)

    def update(self, update: dict[str, Any]) -> None:
        """Updates the parameters with a provided dictionary.

        Args:
            update (dict[str, Any]): Values to update in the parameters. Keys in the
                dictionary are the value of the field to update. Recursive updates can
                be performed using 'dot' notation (ex: foo.bar)
        """
        for key, value in update.items():
            # Recursive case
            if "." in key:
                base_key, _, sub_key = key.partition(".")
                param: BaseParameters = getattr(self, base_key)
                param.update({sub_key: value})
                continue
            # Base case
            if key not in self.__dataclass_fields__:
                raise KeyError(
                    f"Type {type(self).__qualname__} does not contain key {key}",
                )
            self.__setattr__(key, value)

    def as_dict(self) -> dict[str, Any]:
        """Return dataclass as a dictionary using the built in dataclasses.asdict function.

        Returns:
            dict[str, Any]: Dictionary representation of the parameter
        """
        return asdict(self)

    def to_json(self, filepath: Union[str, bytes, os.PathLike], **kwargs) -> None:
        """Dump a JSON string representation of the parameter.

        Args:
            filepath (Union[str, bytes, os.PathLike]): Path of file to write to
            **kwargs: Keyword arguments for open()

        """
        with open(filepath, "w", **kwargs) as outfile:
            json.dump(self.as_dict(), outfile, indent=4)

    @classmethod
    def from_json(
        cls,
        filepath: Union[str, bytes, os.PathLike],
        **kwargs,
    ) -> BaseParameters:
        """Load parameter object from JSON file.

        Args:
            filepath (Union[str, bytes, os.PathLike]): Path of file to read from
            **kwargs: Keyword arguments for open()

        Returns:
            Parameters: Parameters object specified in JSON file
        """
        json_file = Path(filepath)
        xsim_file: Path = (CONFIGS_BASE_DIR / filepath).with_suffix(".json")
        if Path(filepath).is_file():
            filepath = json_file
        elif xsim_file.is_file():
            filepath = xsim_file
        else:
            raise FileNotFoundError(f"Cannot find config: {filepath}")

        with open(filepath, mode="r", **kwargs) as infile:
            data = json.load(infile)
        return cls(**data)

    def __repr__(self) -> str:
        """Create a string representation of a CrossSim parameter object.
        j
                Returns:
                    str: String representation of the parameter.
        """
        return f"{self.__class__.__name__}: \n{pformat(self.as_dict())}"

    def __str__(self) -> str:
        return repr(self)

    def __setattr__(self, name: str, value: Any) -> None:
        """Sets an attribute to the dataclass. Behaves normally except when assigning to an Enum or Parameter.
        If assigning to an Enum an attempt will be made to cast non-enum values to the appropriate Enum.
        If assigning to Parameter, an attempt will be made to unpack the values to the appropriate Parameter.

        Args:
            name (str): Name of attribute to set
            value (Any): Value of attribute to set
        """
        # Only modify behavior on dataclass fields
        #   1. Ensure if name is for dataclass field
        #   2. Ensure field is for an enum or dataclass
        #   if enum
        #       3. Convert value to enum
        #   if dataclass
        #       3. Convert value to dataclass
        #       4. Mark self as parent of new dataclass
        # Otherwise, use default __setattr__
        if name in self.__dataclass_fields__:
            value_type = self.__dataclass_fields__[name].type
            value_type = _resolve_type(value_type, self.__module__)
            if issubclass(value_type, Enum):
                # Syntax for getting by name of enum vs. value of enum is different
                try:
                    value = (
                        value_type[value]
                        if isinstance(value, str)
                        else value_type(value)
                    )
                except KeyError as e:
                    error_msg = f"Error setting parameter '{self.__class__.__name__}.{name}'. Valid values are: {', '.join(value_type._member_names_)}"
                    raise KeyError(error_msg) from e
            if is_dataclass(value_type) and not isinstance(value, value_type):
                # NOTE: This does not automatically convert a list of type list[Parameter]
                #       I don't think we have a use case for that and it seems overengineered

                # If None given for dataclass, instantiate with default values
                _implicitly_initialized = False
                if value is None:
                    value = {}
                    _implicitly_initialized = True
                try:
                    value = value_type(**value)
                except TypeError as e:
                    # This isn't necessary but it helps the user find where they incorrectly configured parameters
                    error_path = (
                        ".".join([param.__class__.__name__ for param in self.parents])
                        + f".{value_type.__name__}"
                    )
                    error_msg = f"Error setting {self.__class__.__name__}.{name}, initializing in {error_path}: {e.args}"
                    raise KeyError(error_msg) from e
                value._parent = self
                value._implicitly_initialized = _implicitly_initialized
        super().__setattr__(name, value)


@dataclass(repr=False)
class BasePairedParameters(BaseParameters):
    """Base class for paired parameters.

    Attributes:
        _match (bool): Whether or not to sync parameters

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

    def validate(self) -> None:
        """Validates settings provided to the paired parameters.

        Raises:
            ValueError: Raised if set to match but mvm and vmm are not equal.
        """
        super().validate()
        if self._match and (self.mvm != self.vmm):
            raise ValueError(
                f"{self.__class__} parameters set to match, but different values given.",
            )

    @property
    def match(self) -> bool:
        """Flag indicating if paired parameters should sync."""
        return self._match

    @match.setter
    def match(self, value: bool) -> None:
        """Sets the match attribute. When the attribute is changed the paired parameters will
        sync/desync appropriately. When syncing, mvm takes precedence over vmm.
        """
        self._match = value
        if self._match:
            self.vmm = self.mvm
        else:
            self.vmm = self.vmm.copy()


def _resolve_type(value_type: Any, module_name) -> type:
    """Returns the type of the corresponding type name from the module. This allows casting
    to be done in the BaseParameter object without any extra boilerplate on any other
    subclassed parameter object.

    Because the value to be cast as could be subclassed anywhere else the type is not
    in the namespace on loading. This function uses the module of origin to retrieve
    the constructor for that type.

    This is always required if we use `from __future__ import annotations`
    NOTE: We might be able to avoid this if we switch to a from_dict() pattern
          but it would add boilerplate to everything.

    Args:
        value_type (Any): Either a string of the type's name, or the type itself.
            If a type is passed, it is returned as is as no lookup is needed.
            If a string is passed, the type of the same name is returned from
                specified module.
        module_name (str): Name of module to use name namespace for

    Returns:
        type: The type specified
    """
    if isinstance(value_type, str):
        # This is a workaround to handle *some* generics until the issues with PEP 563
        # are resolved in a future version of python
        namespace = sys.modules[module_name].__dict__.copy()
        dummy_type = type("_", (), {"__annotations__": {"type": value_type}})
        value_type = get_type_hints(dummy_type, localns=namespace)["type"]
    # Get base type for generic types (e.g. list[bool] -> list)
    value_type = get_origin(value_type) or value_type
    return value_type
