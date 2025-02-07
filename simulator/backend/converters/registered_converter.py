#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from typing import TypeVar, Any
from abc import ABC, abstractmethod

from simulator.backend.registry import register_subclasses

S = TypeVar("S")
T = TypeVar("T")


@register_subclasses
class RegisteredConverter(ABC):
    """Converter for classes registered by CrossSim.

    Attributes:
        converter_type: The type of object the converter creates.
    """

    converter_type: T

    @classmethod
    def create(
        cls: S,
        value_type: T,
        value: Any,
    ) -> T:
        """Create a new object of appropriate type given args and kwargs.

        Args:
            value_type: Type of object to create, should be a of type or subtype
                cls.converter_type
            value: Value to be given to value_type

        Raises:
            ValueError: Raised if the value provided to the converter could not
                be converted to the value_type provided.

        Returns:
            T: Object of type value_type created using associate with the given
                value.
        """
        cls._is_subclass_of_converter_type(value_type)
        try:
            value = cls._create(
                value_type=value_type,
                value=value,
            )
            return value
        except Exception as e:
            error_msg = cls.create_error_msg(
                value_type=value_type,
                value=value,
            )
            raise ValueError(error_msg) from e

    @classmethod
    @abstractmethod
    def _create(
        cls: S,
        value_type: T,
        value: Any,
    ) -> T:
        """Create a new object of appropriate type given args and kwargs.

        This function should not try to catch exeptions.

        Args:
            value_type: Type of object to create, should be a of type or subtype
                cls.converter_type
            value: Value to be given to value_type
            **context: Additional context that may be used when creating object,
                or when creating error messages.

        Raises:
            ValueError: Raised if the value provided to the converter could not
                be converted to the value_type provided.

        Returns:
            T: Object of type value_type created using associate with the given
                value.
        """
        cls._is_subclass_of_converter_type(value_type)
        try:
            value = cls._create(
                value_type=value_type,
                value=value,
            )
            return value
        except Exception as e:
            error_msg = cls.create_error_msg(
                value_type=value_type,
                value=value,
            )
            raise ValueError(error_msg) from e

    @classmethod
    @abstractmethod
    def _create(
        cls: S,
        value_type: T,
        value: Any,
    ) -> T:
        """Create a new object of appropriate type given args and kwargs.

        This function should not try to catch exeptions.

        Args:
            value_type: Type of object to create, should be a of type or subtype
                cls.converter_type
            value: Value to be given to value_type
            **context: Additional context that may be used when creating object,
                or when creating error messages.

        Returns:
            T: Object of type value_type created using associate with the given
                value.
        """
        raise NotImplementedError

    @classmethod
    def _is_subclass_of_converter_type(cls: S, value_type: T):
        msg = (
            f"Cannot use {cls.__name__} to create {value_type.__name__}, "
            f"{value_type.__name__} is not a subclass of {cls.converter_type.__name__}"
        )
        is_subclass = issubclass(value_type, cls.converter_type)
        if not is_subclass:
            raise TypeError(msg)

    @classmethod
    def create_error_msg(
        cls: S,
        value: Any,
        value_type: T,
    ) -> str:
        """Create a user friendly error message if the conversion fails.

        Args:
            value_type: Type of object to create, should be a of type or subtype
                cls.converter_type
            value: Value to be given to value_type

        Returns:
            str: Error message to raise.
        """
        classname = value_type.__name__
        error_msg = (
            f"{cls.__name__}: {classname} object could not be created from '{value}"
        )
        return error_msg
