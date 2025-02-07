#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import re
from typing import Iterable


def convert_to_re(glob_pattern: str) -> re.Pattern:
    """Converts a glob-like dot (.) seperated pattern into a regular expression.

    Supports the following patterns:
        "*": Greedily matches all characters, except "."
        "?": Wild card, matches any character, except "."
        "**": Greedily matches all characters, including "."

    Args:
        glob_pattern: Glob to convert to a regular expression.

    Returns:
        Pattern: An re.Pattern object corresponding to the input glob.
    """
    patterns = glob_pattern.split(",")
    for jdx, pat in enumerate(patterns):
        parts = pat.split(".")
        for idx, part in enumerate(parts):
            if part == "**":
                part = ".*"
                parts[idx] = part
                continue
            if "*" in part:
                part = part.replace("*", "[^.]*")
            if "?" in part:
                part = part.replace("?", "[^.]?")
            parts[idx] = part
        patterns[jdx] = "\\.".join(parts)
        patterns[jdx] = f"^{patterns[jdx]}$"
    glob_pattern = "|".join(patterns)
    glob_pattern = re.compile(glob_pattern)
    return glob_pattern


def globfilter(names: Iterable[str], glob_pattern: str) -> list[str]:
    """Returns all provided names that match the glob pattern specified.

    Args:
        names: List of names to filter with the glob pattern.
        glob_pattern: Pattern to use.

    Returns:
        list[str]: List of all names that match the provided glob pattern.
    """
    results = []
    pattern = convert_to_re(glob_pattern=glob_pattern)
    for name in names:
        match_ = re.match(pattern=pattern, string=name)
        if match_ is not None:
            results.append(match_.string)
    return results
