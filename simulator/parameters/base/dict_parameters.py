#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from __future__ import annotations

from typing import Any, get_origin, get_args, KeysView, ItemsView, ValuesView
from dataclasses import dataclass, field, fields

from simulator.parameters.utils import flatten_param

from .parameters import BaseParameters
from .utils import resolve_registered_and_builtin_types


@dataclass(repr=False)
class BaseDictParameters(BaseParameters):
    """Base class for dictionary-like parameters.

    Dictionary parameters internally manage a dictionary, allowing a
    parameter-compliant interface, but allowing arbitrary keys.
    """

    dict_field_name: str = field(init=False, repr=False)

    def validate(self) -> None:
        """Validates that a DictParameter was written correctly by a developer.

        Raises:
            TypeError: Raised in the following cases:
                1. If 'dict_field_name' is not a string
                2. If the dictionary field does not use registered and
                    built in types.
                3. If the dictionary field is not type annotated with generics
                    (e.g. dict[int, MyParameter])
                4. If the dictionary field is not dict
                5. If the dictionary field values are not BaseParameters
            AttributeError: Raised if the parameter does not have the dictionary
                field, or if it has unexpected fields.
        """
        field_names = {f.name for f in fields(self)}
        expected_field_names = {"dict_field_name", self.dict_field_name}
        if field_names != expected_field_names:
            raise AttributeError(
                f"{self.__class__.__name__} should only have two fields, "
                f"{expected_field_names}. Found {field_names} instead.",
            )
        if not isinstance(self.dict_field_name, str):
            raise TypeError(
                f"Field 'dict_field_name' for {self.__class__.__name__} "
                f"must be of type 'str', found '{type(self.dict_field_name)}'.",
            )
        field_ = self.__dataclass_fields__[self.dict_field_name]
        try:
            field_type = resolve_registered_and_builtin_types(field_.type)
        except NameError as e:
            raise TypeError(
                f"Dictionary field {self.dict_field_name} must be an annotated "
                f"with builtins and registered CrossSim types.\n"
                f"To register a type with CrossSim, see: "
                f"simulator.backend.registry.register_subclasses\n\n",
            ) from e
        field_origin = get_origin(field_type)
        if field_origin is None:
            raise TypeError(
                f"Dictionary field {self.dict_field_name} must be an annotated "
                f"dictionary (e.g. dict[str, int]). got '{field_.type}'",
            )
        if not issubclass(field_origin, dict):
            raise TypeError(
                f"Dictionary field '{self.dict_field_name}' must be of type dict. "
                f"Got '{field_.type}'",
            )
        _, field_value_type = get_args(field_type)
        if not issubclass(field_value_type, BaseParameters):
            field_value_type_name = field_value_type.__name__
            raise TypeError(
                f"Values of dictionary field must be a parameter. "
                f"In '{field_.type}', '{field_value_type_name}' is not a parameter.",
            )
        return super().validate()

    def keys(self) -> KeysView:
        """A set-like object providing a view on dict parameter's keys."""
        return self._dict.keys()

    def values(self) -> ValuesView:
        """A set-like object providing a view on dict parameter's values."""
        return self._dict.values()

    def items(self) -> ItemsView:
        """A set-like object providing a view on dict parameter's items."""
        return self._dict.items()

    @property
    def _dict(self) -> dict:
        return self.__getattribute__(self.dict_field_name)

    @classmethod
    def _field_value_type(cls) -> type:
        field_ = cls.__dataclass_fields__[cls.dict_field_name]
        field_type = resolve_registered_and_builtin_types(field_.type)
        _, field_value_type = get_args(field_type)
        return field_value_type

    def _get(self, key: str) -> Any:
        """Called internally by get().

        This function is responsible for taking a key and interpretting the key
        for the respective parameter then fetching the appropriate value at that
        key.

        This logic is split so that the error handling and recursive logic can
        be shared between params, while subclasses are free to define their own
        behavior on how to interpret a key, which is needed especially in dict
        params.
        """
        try:
            return self._dict[key]
        except KeyError as e:
            classname = self.__class__.__name__
            raise KeyError(
                f"{classname} instance does not kave key '{key}'.",
            ) from e

    def set_value(self, name: str, value: Any) -> None:
        """Sets a single attribute to the dataclass.

        Behaves normally except when assigning to a registered CrossSim class.
        In which case, an attempt will be made to convert the value to the
        expected CrossSim class.

        Args:
            name: Name of attribute to set
            value: Value of attribute to set
        """
        if name in dir(self):
            return object.__setattr__(self, name, value)

        if name == self.dict_field_name:
            if not hasattr(self, self.dict_field_name):
                object.__setattr__(self, name, {})
            for k, v in value.items():
                self.set_key(key=k, value=v)
        else:
            self.set_key(key=name, value=value)

    def set_key(self, key: str, value):
        """Sets a value to a key in the dictionary.

        Args:
            key: Key to set value at
            value: Value to set.
        """
        from .utils import convert_type

        value_type = self._field_value_type()
        converted_value = convert_type(
            value_type=value_type,
            value=value,
            key_name=key,
            parent=self,
        )
        self._dict[key] = converted_value
        converted_value._parent = self.parent

    def pop_key(self, key: str) -> Any:
        """Pops and returns a value at a given key.

        Args:
            key: Key to pop value from.

        Returns:
            Any: Value at the specified key.
        """
        return self._dict.pop(key)

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
            return flatten_param(
                self.as_dict(
                    flat=False,
                    exclude_non_init=exclude_non_init,
                    exclude_non_repr=exclude_non_repr,
                ),
            )
        dict_ = {
            key: value.as_dict(
                flat=False,
                exclude_non_init=exclude_non_init,
                exclude_non_repr=exclude_non_repr,
            )
            for key, value in self._dict.items()
        }
        return dict_

    def __setitem__(self, key: Any, value: Any):
        """Sets an key in the internal dictionary to a value."""
        self.set_key(key=key, value=value)
