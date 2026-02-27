from __future__ import annotations

import pytest

from jarvis_operas.catalog import get_catalog_declarations


def test_catalog_default_namespaces_do_not_include_interpolations() -> None:
    declarations = get_catalog_declarations()
    namespaces = {item.namespace for item in declarations}

    assert {"math", "stat", "helper"}.issubset(namespaces)
    assert "interp1" not in namespaces
    assert "dmdd" not in namespaces


def test_catalog_can_load_interpolation_namespaces_on_demand() -> None:
    declarations = get_catalog_declarations(
        namespaces=["interp1", "dmdd"],
        include_interpolations=True,
    )
    names = {item.full_name for item in declarations}

    assert "interp1.interp1_xy_flat" in names
    assert "dmdd.LZSI2024" in names


def test_catalog_rejects_unknown_namespace() -> None:
    with pytest.raises(KeyError):
        get_catalog_declarations(namespaces=["unknown_namespace"])
