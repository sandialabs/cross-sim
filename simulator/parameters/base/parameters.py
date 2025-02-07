#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
"""Provides base functionality for parameters in CrossSim.

This module provides the following classes:

BaseParameters:
    Main class of a parameters which defines the behavior of all other
    parameters. Provides the following features:
        - Automatic casting from string to other parameters or registered enums
        - Saving/loading from JSON format
        - Run time type checking to warn users of incorrect types
        - Automatic parameter verification
        - Keeps track of parameter tree
        - Keeps track if parameter was explicitly/implicityly defined by user
PairedParameters:
    A class of parameters where a set of coupled parameters (named mvm and vmm)
    can be optionally synced automatically.
RegisteredEnum:
    A registered CrossSim base class for any enums that might are used in any
    parameter objects. Inhereting from RegisteredEnum will allow CrossSim to
    automatically find and create the appropriate enum.
"""
from __future__ import annotations

import json
import os
import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from enum import IntEnum
from pprint import pformat
from pathlib import Path
from typing import Any, Union, TypeVar

from simulator.configs import CONFIGS_BASE_DIR
from simulator.backend.registry import register_subclasses, RegistryManager
import simulator.parameters.utils as param_utils


log = logging.getLogger(__name__)

T = TypeVar("T")
ParameterType = TypeVar("ParameterType", bound="BaseParameters")


@register_subclasses
class RegisteredEnum(IntEnum):
    """Base class for CrossSim enums."""

    pass


@register_subclasses
@dataclass(repr=False)
class BaseParameters:
    """Base class for CrossSim parameters.

    Attributes:
        parent (BaseParameters): Parent of this parameter in a nested parameter
            tree
        parents (list[BaseParameters]): List of all parameters directly above
            the current parameter. The first index of the list is the root
            parameter
        root (BaseParameters): Root of the parameter tree
    """

    def __new__(cls, *args, **kwargs) -> BaseParameters:
        """Create a new BaseParameters object.

        Returns:
            BaseParameters: Returns a new BaseParameters object
        """
        param = super().__new__(cls)
        # Here we use super setattr because the base parameters
        # have overloaded setattr operations
        #
        # Add parent before __init__ is called, allowing us to use the
        # parameter tree during initialization
        super().__setattr__(param, "_parent", param)
        # Flag to mark if a parameter was initialized implicitly
        # (i.e. It was automatically created because 'None' was provided
        #       during init)
        super().__setattr__(param, "_implicitly_initialized", False)
        return param

    def __post_init__(self):
        """Runs after dataclass initialization."""
        self.validate()

    @property
    def parent(self) -> BaseParameters:
        """Returns the parent of the parameter in the parameter tree.

        Returns:
            BaseParameters: Returns the BaseParameters object that holds this
                parameter object. If parameter has no parent, self.parent
                returns itself.
        """
        return self._parent

    @property
    def parents(self) -> list[BaseParameters]:
        """Returns a list of parent of the parameters.

        The first index of the list is the root, the last parameter is the
        current parameter.

        Returns:
            list[BaseParameters]: Returns the list of parameters above the
                current parameter in the parameter tree.
        """
        if self.parent is self:
            return [self]
        return [*self.parent.parents, self]

    @property
    def root(self) -> BaseParameters:
        """Root of the parameter tree.

        Returns:
            BaseParameters: Parameter object at the base of the parameter tree
        """
        return self.parents[0]

    def get_path_from_root(self) -> str:
        """Returns the path from the root parameter to the parameter."""
        if self.root is self:
            return ""

        results = [k for k, v in self.root.search("**").items() if v is self]
        if len(results) == 0:
            raise LookupError(
                "Unexpected error: Could not find self in the root param",
            )
        elif len(results) > 1:
            raise LookupError(
                "Unexpected error: Multiple paths to param found from root parameter",
            )
        return results[0]

    def validate(self) -> None:
        """Validates the parameters.

        Will warn if values do not match type hints. Subclassed parameters may
        raise error if parameters provided unusable or inconsistent.
        """
        for key, field in self.__dataclass_fields__.items():
            # Log potential typing mismatch
            expected_type_name = field.type
            if not isinstance(field.type, str):
                # field.type may or may not be a string annotation
                # of the type hint. If is in actual type, get string
                # version of the name
                expected_type_name = field.type.__name__
            value = getattr(self, field.name)
            inferred_type_name = type(value).__name__

            if inferred_type_name == expected_type_name:
                log.debug(f"Key {key} matches expected type of {expected_type_name}.")
            else:
                log.info(
                    f"Key {key} expected type {expected_type_name}, "
                    f"got {inferred_type_name}",
                )

    def copy(self: ParameterType) -> ParameterType:
        """Returns a deep copy of the parameter.

        Returns:
            BaseParameters: A deep copy of the original parameter object
        """
        return deepcopy(self)

    def _recursive_post_init(self, post_init_self: bool = True):
        """Runs __post_init__ on all child parameters then on itself."""
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if isinstance(value, BaseParameters):
                value._recursive_post_init()
        if post_init_self:
            self.__post_init__()

    def _recursive_validate(self, validate_self: bool = True):
        """Runs validate on all child parameters, then on itself."""
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if isinstance(value, BaseParameters):
                value._recursive_validate(True)
        if validate_self:
            self.validate()

    def update(
        self: ParameterType,
        update_: dict[str, Any],
        post_init: bool = True,
    ) -> ParameterType:
        """Updates the parameters in place with a provided dictionary.

        Args:
            update_: Values to update in the parameters. Keys in the dictionary
                are the value of the field to update. Recursive updates can be
                performed using 'dot' notation (ex: foo.bar)
            post_init: If True, the parameter and child parameters will rerun
                their __post_init__ commands.

        Returns:
            BaseParameters: Returns the parameter itself, useful for
                initializing with a given update.
                (e.g param = Parameter().update(my_update))
        """
        for key, value in update_.items():
            # Recursive case
            if "." in key:
                base_key, _, sub_key = key.partition(".")
                param: BaseParameters = getattr(self, base_key)
                param.update({sub_key: value}, post_init=False)
                continue
            # Base case
            if key not in self.__dataclass_fields__:
                raise KeyError(
                    f"Type {type(self).__qualname__} does not contain key {key}",
                )
            self.__setattr__(key, value)
        if post_init:
            self._recursive_post_init()
        return self

    def as_dict(
        self,
        flat: bool = False,
        exclude_non_init: bool = True,
        exclude_non_repr: bool = True,
    ) -> dict[str, Any]:
        """Return the dataclass as a dictionary.

        Args:
            flat: If True, nested keys will be represented with '.' seperated
                keys. If False, the dictionary may contain nested dictionaries.
                Defaults to False.
            exclude_non_init: If True, the resulting dictionary will not include
                any dataclass fields that are not part of the dataclass's init
                function. Defaults to True.
            exclude_non_repr: If True, the resulting dictionary will not include
                any dataclass fields that are marked to not be shown in the
                dataclass's repr. Defaults to True.

        Returns:
            dict[str, Any]: Dictionary representation of the parameter
        """
        if flat:
            return param_utils.flatten_param(
                self.as_dict(
                    flat=False,
                    exclude_non_init=exclude_non_init,
                    exclude_non_repr=exclude_non_repr,
                ),
            )
        if exclude_non_init is False and exclude_non_repr is False:
            return asdict(self)
        dict_ = {}
        for field in fields(self):
            field_name = field.name
            field_instance = getattr(self, field_name)
            if not field.repr and exclude_non_repr:
                continue
            if not field.init and exclude_non_init:
                continue
            elif not isinstance(field_instance, BaseParameters):
                dict_[field_name] = field_instance
            else:
                dict_[field_name] = field_instance.as_dict(
                    exclude_non_init=exclude_non_init,
                    exclude_non_repr=exclude_non_repr,
                )
        return dict_

    @classmethod
    def from_dict(cls: ParameterType, dict_: dict[str, Any]) -> ParameterType:
        """Creates a parameter using the dictionary specified.

        Args:
            dict_: Dictionary specifiying the settings of the parameter.
                If any values are missing, default values will be used.

        Returns:
            ParameterType: Parameter from the dictionary specified.
        """
        dict_ = param_utils.nest_dict(param=dict_)
        return cls(**dict_)

    def to_json(self, filepath: Union[str, bytes, os.PathLike], **kwargs) -> None:
        """Dump a JSON string representation of the parameter.

        Args:
            filepath: Path of file to write to
            **kwargs: Keyword arguments for open()
        """
        with open(filepath, "w", **kwargs) as outfile:  # noqa: PTH123
            json.dump(self.as_dict(), outfile, indent=4)

    @classmethod
    def from_json(
        cls: ParameterType,
        filepath: Union[str, bytes, os.PathLike],
        **kwargs,
    ) -> ParameterType:
        """Load parameter object from JSON file.

        Args:
            filepath: Path of file to read from
            **kwargs: Keyword arguments for open()

        Returns:
            BaseParameters: BaseParameters object specified in JSON file
        """
        registry_manager = RegistryManager()
        json_file = Path(filepath)
        xsim_file: Path = (CONFIGS_BASE_DIR / filepath).with_suffix(".json")
        if Path(filepath).is_file():
            filepath = json_file
        elif xsim_file.is_file():
            filepath = xsim_file
        else:
            raise FileNotFoundError(f"Cannot find config: {filepath}")

        with open(filepath, mode="r", **kwargs) as infile:  # noqa: PTH123
            data = json.load(infile)
        return registry_manager.convert(type_=cls, value=data)

    def search(self, key: str) -> dict[str, Any]:
        """Returns a dictionary of all values that match the specified key.

        Args:
            key: Key or glob pattern to match.

        Returns:
            dict[str, Any]: Dictionary of search results. Key values are the
                keys that matched the search key. Values are the corresponding
                values at the respective key.
        """
        flat_param = self.as_dict(flat=True)
        matching_keys = param_utils.get_matching_keys(flat_param=flat_param, key=key)
        result = {key: self.get(key) for key in matching_keys}
        return result

    def __str__(self) -> str:
        """Create a string representation of a CrossSim parameter object.

        Returns:
            str: String representation of the parameter.
        """
        return f"{self.__class__.__name__}: \n{pformat(self.as_dict())}"

    def get(self, key: str) -> Any:
        """Gets the value of at the specified key.

        Args:
            key: Key to get value from. Supports nested keys using a
                '.' notation. (e.g. foo.bar.baz)

        Returns:
            Any: Value at the specified key

        Raises:
            KeyError: Raised if no value is at the specified key.
        """
        value = self
        if key == "":
            return value
        parts = param_utils.split_key(key)
        try:
            while len(parts):
                value, parts = value._get(key=parts)
        except Exception as e:
            classname = self.__class__.__name__
            msg = (
                f"{classname} object cannot get value for key '{key}': "
                f"sub-key '{'.'.join(parts)}' is invalid."
            )
            raise KeyError(msg) from e
        return value

    def _get(self, key: list[str]) -> tuple[Any, list[str]]:
        """Called internally by get().

        This function is responsible for taking a key and interpretting the key
        for the respective parameter then fetching the appropriate value at that
        key.

        This logic is split so that the error handling and recursive logic can
        be shared between params, while subclasses are free to define their own
        behavior on how to interpret a key, which is needed especially in dict
        params.

        Args:
            key: List of key parts to get fetch value with.

        Returns:
            Tuple[Any, list]: Value at the first matching result from the key,
                and a list of the key parts remaining
        """
        if len(key) == 0:
            return self, []
        return getattr(self, key[0]), key[1:]

    def __getitem__(self, key: str) -> Any:
        """Gets the value of at the specified key.

        Args:
            key: Key to get value from. Supports nested keys using a
                '.' notation. (e.g. foo.bar.baz)

        Returns:
            Any: Value at the specified key

        Raises:
            KeyError: Raised if no value is at the specified key.
        """
        return self.get(key=key)

    def set(self, key: str, value: Any):
        """Sets all matching keys to the specified value.

        Args:
            key: Key or glob pattern to match.
            value: Value to set all matching keys to.

        Raises:
            KeyError: Raised if no key matches when globbing is not used.
        """
        special_chars = ["*", "?"]
        matching_keys = [key]
        if any(c in key for c in special_chars):
            flat_param = self.as_dict(flat=True)
            matching_keys = param_utils.get_matching_keys(
                flat_param=flat_param,
                key=key,
            )
        for matching_key in matching_keys:
            parent = self
            child_key = matching_key
            if "." in matching_key:
                parent_key, _, child_key = matching_key.rpartition(".")
                parent = self.get(parent_key)
            parent.set_value(name=child_key, value=value)

    def set_value(self, name: str, value: Any) -> None:
        """Sets a single attribute to the dataclass.

        Behaves normally except when assigning to a registered CrossSim class.
        In which case, an attempt will be made to convert the value to the
        expected CrossSim class.

        Args:
            name: Name of attribute to set
            value: Value of attribute to set
        """
        registry_manager = RegistryManager()
        if name in self.__dataclass_fields__:
            value_type = self.__dataclass_fields__[name].type
            is_registered_type = value_type in registry_manager
            if is_registered_type:
                from .utils import convert_type

                value = convert_type(
                    value_type=value_type,
                    value=value,
                    key_name=name,
                    parent=self,
                )
            super().__setattr__(name, value)
        elif name in dir(self):
            super().__setattr__(name, value)
        else:
            raise AttributeError(
                f"Cannot set attribute '{name}' to {self.__class__.__name__}",
            )

    def __setattr__(self, name: str, value: Any) -> None:
        """Sets an attribute to the dataclass.

        Behaves normally except when assigning to a registered CrossSim class.
        In which case, an attempt will be made to convert the value to the
        expected CrossSim class.

        Args:
            name: Name of attribute to set
            value: Value of attribute to set
        """
        self.set_value(name=name, value=value)

    def __setitem__(self, key: str, value: Any):
        """Sets all matching keys to the specified value.

        Args:
            key: Key or glob pattern to match.
            value: Value to set all matching keys to.

        Raises:
            KeyError: Raised if no key matches when globbing is not used.
        """
        self.set(key=key, value=value)
