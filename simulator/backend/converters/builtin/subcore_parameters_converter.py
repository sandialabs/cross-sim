#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from typing import TypeVar, Any

from simulator.backend.registry import RegistryManager
from simulator.parameters.core.core import SubcoreParameters
from simulator.backend.converters.builtin.dict_parameters_converter import (
    DictParametersConverter,
)
from ast import literal_eval
import re


PATTERN_ABBREVIATED_CORE_INFO = re.compile(
    pattern=r"""
    (?P<core_type>[_A-z]\w+) # The name of the core type
    (?::(?P<subcore_keys>[^\[]+))? # The keys of of it's subcores.
    (?:\[\s*(?P<kwargs>[^\]]+)\s*\])? # Any arguments to pass into the core
    """,
    flags=re.VERBOSE,
)

S = TypeVar("S")
T = TypeVar("T")


class SubcoreParametersConverter(DictParametersConverter):
    """Converter for CrossSim parameters."""

    converter_type: T = SubcoreParameters

    @classmethod
    def _create(
        cls: S,
        value_type: T,
        value: Any,
    ) -> T:
        """Create a new parameter of appropriate type given a value.

        Args:
            value_type: Type of object to create, should be a of type or subtype
                cls.converter_type
            value: Value to be given to value_type

        Returns:
            T: Object of type value_type created using associate with the given
                value.
        """
        if isinstance(value, list):
            value = expand_abbreviated_core_stack(value)
        return super()._create(value_type=value_type, value=value)


def expand_abbreviated_core_stack(core_stack: list[str]) -> SubcoreParameters:
    """Expands an abbreviated core stack into a subcore parameter.

    Abbreviated core stacks assume that a homogeneous stack is being built.

    Args:
        core_stack: List of cores to build stack with.

    Returns:
        SubcoreParameters: Subcore parameters built using the stack supplied.
    """
    registry_manager = RegistryManager()
    subcores = SubcoreParameters()
    for core_info_str in reversed(core_stack):
        core_info = get_abbreviated_info(core_info=core_info_str)
        core_type = registry_manager.get_from_key(
            parent="CoreParameters",
            key_name="core_type",
            key_value=core_info["core_type"],
        )
        kwargs = core_info["kwargs"]
        core_param = core_type(**kwargs)
        subcores = SubcoreParameters(
            subcores={core_info["subcore_keys"]: core_param},
        )
    return subcores


def _get_open_closing_pairs(pairs) -> tuple[dict, dict]:
    """Returns a tuple of open/closing dicts.

    Raises an error if the pairs are not a group of two.
    """
    opening = {}
    closing = {}
    if len(pairs) % 2 != 0:
        raise ValueError("Pairs must be of groups of 2")

    for open_char, close_char in zip(pairs[::2], pairs[1::2], strict=True):
        if open_char == close_char:
            raise ValueError("Open and close symbols must be different.")
        opening[open_char] = close_char
        closing[close_char] = open_char
    return opening, closing


def _key_is_balanced(s: str, pairs: str) -> bool:
    """Check if the string has balanced pairs of open/close characters.

    Args:
        s: The input string to check.
        pairs: A list of strings representing pairs of open/close characters.

    Returns:
        bool: True if the string is balanced, False otherwise.
    """
    stack = []
    consuming = set(""""'""")
    opening, closing = _get_open_closing_pairs(pairs=pairs)

    in_consuming = False
    for char in s:
        if in_consuming:
            if stack[-1] == char:
                stack.pop()
                in_consuming = False
        else:
            if char in consuming:
                stack.append(char)
                in_consuming = True
            elif char in opening:
                stack.append(char)
            elif char in closing:
                if not stack or stack[-1] != closing[char]:
                    return False
                stack.pop()
    return not stack


def _multi_index_split(text: str, idxs: list[int]) -> list[str]:
    """Splits a string at multiple indexes.

    Args:
        text: Text to be split.
        idxs: Indexes to split the text at.

    Returns:
        list[str]: List of each substring resulting from the split.
    """
    result = []
    last_pos = 0
    for pos in idxs:
        result.append(text[last_pos:pos])
        last_pos = pos + 1
    result.append(text[last_pos:])
    return result


def _split_outside(text: str, delimiter: str, pairs: str) -> list[str]:
    """Split the string on a delimiter when the delimiter is outside any of the
    specified brackets, quotes, etc.

    Args:
        text: The input string to split.
        pairs: A list of strings representing pairs of open/close characters.
        delimiter: The delimiter to split the string on.

    Returns:
        list[str]: A list of substrings split by the delimiter outside the
            specified brackets, quotes, etc.
    """
    if not _key_is_balanced(text, pairs):
        raise ValueError("The input string is not balanced.")

    stack = []
    consuming = set(""""'""")
    opening, closing = _get_open_closing_pairs(pairs=pairs)

    in_consuming = False
    split_positions = []

    for i, char in enumerate(text):
        if char in consuming:
            if not in_consuming:
                stack.append(char)
                in_consuming = True
            elif stack[-1:] == [char]:
                stack.pop()
                in_consuming = False
        elif char in opening:
            stack.append(char)
        elif char in closing:
            if stack[-1:] == [closing[char]]:
                stack.pop()
        elif char == delimiter and not stack:
            split_positions.append(i)

    result = _multi_index_split(text=text, idxs=split_positions)
    return result


def get_abbreviated_info(core_info: str) -> dict[str, Any]:
    """Returns a dictionary containing the processed info for a core stack.

    Args:
        core_info: A user specified string defining a core.

    Raises:
        ValueError: Raised if the provided value could not be interpretted.

    Returns:
        dict[str, Any]: Parsed value in the form of a dictionary with three keys
            "core_type": The name of the core type
            "subcore_keys": Keys of the subcores
            "kwargs": Keyword arguments to pass into the core
    """
    match = PATTERN_ABBREVIATED_CORE_INFO.match(string=core_info)
    if not match:
        raise ValueError(f"Could not parse abbreviated core '{core_info}'")
    info = match.groupdict()

    if info["subcore_keys"] is None:
        info["subcore_keys"] = "0"

    if info["kwargs"] is None:
        info["kwargs"] = {}
    else:
        kwargs = _split_outside(text=info["kwargs"], delimiter=",", pairs="()[]{}")
        kwargs = [
            _split_outside(text=kwarg, delimiter="=", pairs="()[]{}")
            for kwarg in kwargs
        ]
        if not all(len(kwarg) == 2 for kwarg in kwargs):
            raise ValueError(f"Could not parse keyword args for '{core_info}'")
        parsed_kwargs = {}
        for k, v in kwargs:
            try:
                parsed_kwargs[k.strip()] = literal_eval(v)
            except ValueError as e:
                raise ValueError(
                    "Error parsing abbreviated core '{core_info}, "
                    "'{v}' is not a literal expression.",
                ) from e
        info["kwargs"] = parsed_kwargs
    return info
