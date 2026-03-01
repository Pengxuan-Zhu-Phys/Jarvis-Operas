from __future__ import annotations

import pytest

from jarvis_operas.catalog import get_catalog_declarations


def test_catalog_default_namespaces_do_not_include_interpolations() -> None:
    declarations = get_catalog_declarations()
    namespaces = {item.namespace for item in declarations}

    assert {"math", "stat", "helper", "cmb"}.issubset(namespaces)
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


def test_catalog_all_declarations_include_cli_examples() -> None:
    declarations = get_catalog_declarations(include_interpolations=True)
    missing: list[str] = []
    for item in declarations:
        metadata = dict(item.metadata)
        examples = metadata.get("examples")
        has_example = False
        if isinstance(examples, list):
            has_example = any(isinstance(x, str) and x.strip() for x in examples)
        elif isinstance(examples, dict):
            has_example = isinstance(examples.get("cli"), str) and bool(examples.get("cli").strip())
        elif isinstance(examples, str):
            has_example = bool(examples.strip())
        if not has_example:
            missing.append(item.full_name)

    assert not missing, f"missing metadata.examples: {missing}"


def test_catalog_all_declarations_include_support_capabilities_metadata() -> None:
    declarations = get_catalog_declarations(include_interpolations=True)
    for item in declarations:
        metadata = dict(item.metadata)
        supports = metadata.get("supports")
        assert isinstance(supports, dict)
        assert supports.get("call") is True
        assert supports.get("acall") is True
        assert isinstance(supports.get("numpy"), bool)
        assert isinstance(supports.get("polars"), bool)

        supported_types = metadata.get("supported_types")
        assert isinstance(supported_types, list)
        assert "call" in supported_types
        assert "acall" in supported_types
