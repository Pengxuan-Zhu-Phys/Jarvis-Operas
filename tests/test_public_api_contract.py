from __future__ import annotations

import asyncio

import jarvis_operas as jo
from jarvis_operas import Operas, OperasRegistry


def test_public_api_exports_stable_minimum_symbols() -> None:
    required = {
        "OperaFunction",
        "OperasRegistry",
        "Operas",
        "oper",
        "get_global_operas_registry",
        "get_global_operas",
        "load_user_ops",
        "persist_user_ops",
        "list_persisted_user_ops",
    }
    for name in required:
        assert hasattr(jo, name), f"missing public symbol: {name}"


def test_public_global_registry_contract() -> None:
    registry = jo.get_global_operas_registry()
    assert isinstance(registry, OperasRegistry)

    resolved = registry.resolve_name("math.add")
    assert resolved == "math.add"
    assert "math.add" in registry.list(namespace="math")
    assert registry.call("math.add", a=1, b=2) == 3

    info = registry.info("math.add")
    required_keys = {
        "id",
        "name",
        "namespace",
        "short_name",
        "arity",
        "return_dtype",
        "supports_numpy",
        "supports_polars_native",
        "supports_polars_fallback",
        "metadata",
        "signature",
        "docstring",
        "is_async",
        "supports_async",
    }
    assert required_keys.issubset(set(info.keys()))


def test_public_global_operas_contract_sync_and_async() -> None:
    operas = jo.get_global_operas()
    assert isinstance(operas, Operas)
    assert operas.call("math.add", a=2, b=3) == 5

    async def _runner() -> int:
        return await operas.acall("math.add", a=4, b=5)

    assert asyncio.run(_runner()) == 9


def test_public_oper_decorator_contract_with_explicit_registry() -> None:
    registry = OperasRegistry()

    @jo.oper("triple", namespace="contract", registry=registry, summary="x * 3")
    def _triple(x):
        return x * 3

    assert registry.call("contract.triple", x=7) == 21
    assert registry.info("contract.triple")["metadata"]["summary"] == "x * 3"


def test_public_load_and_persist_user_ops_contract(tmp_path, monkeypatch) -> None:
    source = tmp_path / "my_ops.py"
    source.write_text(
        (
            "from jarvis_operas import oper\n"
            "\n"
            "@oper('double')\n"
            "def double(x):\n"
            "    return x * 2\n"
            "\n"
            "def inc(x):\n"
            "    return x + 1\n"
            "\n"
            "__JARVIS_OPERAS__ = {'inc': inc}\n"
        ),
        encoding="utf-8",
    )

    registry = OperasRegistry()
    loaded = jo.load_user_ops(str(source), registry)
    assert set(loaded) == {"my_ops.double", "my_ops.inc"}
    assert registry.call("my_ops.double", x=4) == 8
    assert registry.call("my_ops.inc", x=4) == 5

    monkeypatch.setenv("JARVIS_OPERAS_SOURCES_FILE", str(tmp_path / "sources.json"))
    monkeypatch.setenv("JARVIS_OPERAS_OVERRIDES_FILE", str(tmp_path / "overrides.json"))
    monkeypatch.setenv("JARVIS_OPERAS_USER_SOURCE_DIR", str(tmp_path / "user_sources"))

    persisted = jo.persist_user_ops(str(source))
    assert "entry" in persisted
    entries = jo.list_persisted_user_ops()
    assert any(item.get("path") == str(source.resolve()) for item in entries)
