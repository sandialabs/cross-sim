#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from __future__ import annotations

import re
from typing import Any

from simulator.backend.globbing import globfilter


# Splitting patterns
# Finds all commas that are not inside a parenthesis
PATTERN_MULTIKEY_SPLIT = re.compile(r",\s*(?![^()]*\))")
# Finds all dots that are not inside a parenthesis
PATTERN_KEY_SPLIT = re.compile(r"\.(?![^()]*\))")

# Key interpretation patterns
PATTERN_SINGLE_INTEGER = re.compile("^\\s*-?\\d+\\s*$")
PATTERN_INTEGER_SEQUENCE = re.compile("^\\s*\\d+\\s*-\\s*\\d+\\s*$")
PATTERN_INTEGER_COORDINATE = re.compile("^\\s*\\(\\s*-?\\d+\\s*,\\s*-?\\d+\\s*\\)\\s*$")


def split_multikeys(key: str) -> list[str]:
    """Splits a comma separated list of keys, not splitting on commas inside of
    parentheses.

    Example:
        "foo,bar,baz" -> ["foo", "bar", "baz"]
        "foo.(0,0).0, bar.(0,0).0" -> ["foo.(0,0).0", "bar.(0,0).0"]

    Args:
        key: String of potentially multiple comma separated keys

    Returns:
        list[str]: List of keys after splitting
    """
    if isinstance(key, list):
        return key
    return PATTERN_MULTIKEY_SPLIT.split(key)


def split_key(key: str) -> list[str]:
    """Splits a key into subparts.

    Example:
        "foo.(0,0).0" -> ["foo", "(0,0)", "0"]
        "bar.(1.0,1.0).0" -> ["bar", "(1.0,1.0)", "0"]

    Args:
        key: String of potentially multiple comma separated keys

    Returns:
        list[str]: List of keys after splitting
    """
    if isinstance(key, list):
        return key
    return PATTERN_KEY_SPLIT.split(key)


def expand_key(raw_key: str) -> list[Any]:
    """Expands abbreviated key notation for core parameter subcores.

    A key of "0-3" will expand to [0, 1, 2, 3].
    A key of "2-1" will error.
    A key of "positive" will return ["positive"]
    """
    if not isinstance(raw_key, str):
        return [raw_key]
    if PATTERN_SINGLE_INTEGER.match(raw_key):
        return [int(raw_key)]
    elif PATTERN_INTEGER_SEQUENCE.match(raw_key):
        start, _, finish = raw_key.partition("-")
        start = int(start)
        finish = int(finish)
        if start > finish:
            raise ValueError(
                "Improperly formatted shorthand key core subcore parameters. "
                "Subcore keys must either be an integer, or a string of the form "
                '"a-b" for integers a <= b',
            )
        return list(range(start, finish + 1))
    else:
        return [raw_key]


def interpret_key(key: str) -> tuple | int | str:
    """Interprets a string key and converts it into its intended type.

    Args:
        key: String to interpret

    Raises:
        TypeError: If the key is not an int, tuple, or str, or if the str
            is not in the form of an int or tuple

    Returns:
        The input key changed to type int or tuple
    """
    if isinstance(key, str):
        if re.match(PATTERN_SINGLE_INTEGER, key):
            return int(key)
        elif re.match(PATTERN_INTEGER_COORDINATE, key):
            return key_as_coord(key)
    return key


def key_as_coord(key: str) -> tuple[int, int]:
    """Interprets a string key as a 2D coordinate.

    Expects keys as a tuple of ints, e.g. (0, 0), (-1, -1), (-1, 10), etc.

    Args:
        key: String version of a 2D coordinate.

    Raises:
        ValueError: Raised if the string can not be interpretted as coordinate.

    Returns:
        tuple[int, int]: Literal version of coordinates from string.
    """
    x, _, y = key.strip("( )").partition(",")
    try:
        coord = (int(x), int(y))
        return coord
    except ValueError as e:
        raise ValueError(
            f"Failed to interpret key: '{key}' as a coordinate.",
        ) from e


def flatten_param(param: dict) -> dict[str, Any]:
    """Converts a parameter into a flat dictionary.

    Args:
        param: Parameter to flatten.
        **kwargs: Keyword arguments to pass to param.as_dict

    Returns:
        dict[str, Any]: Flattened dictionary representation of the parameter.
    """
    if not isinstance(param, dict):
        raise TypeError("Flatten param now requires a dictionary")
    flat = {}
    for key, value in param.items():
        if isinstance(value, dict):
            subflat = flatten_param(value)
            for k, v in subflat.items():
                flat[f"{key}.{k}"] = v
        else:
            flat[str(key)] = value
    return flat


def nest_dict(param: dict[str, Any]) -> dict[str, Any]:
    """Nests a flattened dictionary. Nesting occurs on dotted keys.
    (e.g. foo.bar.baz).

    Args:
        param: Flattened dictionary to nest.

    Returns:
        dict[str, Any]: A nested version of a flattened parameter.
    """
    # Base case
    nested = {k: v for k, v in param.items() if "." not in k}

    # Recursive case
    needs_nesting = {k: v for k, v in param.items() if "." in k}
    prefixes = {str(k).partition(".")[0] for k in needs_nesting.keys()}
    for prefix in prefixes:
        prefix_dict = {}
        for k, v in param.items():
            if str(k).split(".")[0] != prefix:
                continue
            postfix = str(k).partition(".")[-1]
            prefix_dict[postfix] = v
        nested[prefix] = nest_dict(prefix_dict)
    return nested


def get_matching_keys(flat_param: dict, key: str) -> set[str]:
    """Returns a set of all keys that match on the parameter.

    Args:
        flat_param: Parameter to get matching keys on
        key: Key to match on for setting values.

    Returns:
        set[str]: Set of keys that match the search key
    """
    search_keys = split_multikeys(key)
    keys = set()
    for child_node in flat_param.keys():
        child_key = child_node
        while child_key != "":
            keys.add(child_key)
            child_key, _, _ = child_key.rpartition(".")

    matching_keys = set()
    for search_key in search_keys:
        search_result = globfilter(keys, search_key)
        matching_keys.update(search_result)
    return matching_keys
