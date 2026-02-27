#!/usr/bin/env python3
from __future__ import annotations

import pytest

from jarvis_operas.name_utils import normalize_full_name, split_full_name, try_split_full_name


def test_split_full_name_accepts_dot_separator() -> None:
    assert split_full_name("math.add") == ("math", "add")


def test_split_full_name_requires_namespace_dot_name_format() -> None:
    with pytest.raises(ValueError):
        split_full_name("math_add")
    with pytest.raises(ValueError):
        normalize_full_name("math_add")
    assert try_split_full_name("math_add") is None
