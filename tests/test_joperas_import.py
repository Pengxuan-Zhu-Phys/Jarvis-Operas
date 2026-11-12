"""Script imports of the form ``from joperas.<namespace> import <name>``."""

from __future__ import annotations

import importlib
import inspect
import keyword
import sys

import pytest

from jarvis_operas.core.spec import OperaFunction


def _reset_import_state(monkeypatch, tmp_path, *, curves: bool) -> None:
    monkeypatch.setenv("JARVIS_OPERAS_SOURCES_FILE", str(tmp_path / "sources.json"))
    monkeypatch.setenv("JARVIS_OPERAS_OVERRIDES_FILE", str(tmp_path / "overrides.json"))
    monkeypatch.setenv("JARVIS_OPERAS_USER_SOURCE_DIR", str(tmp_path / "user_sources"))
    monkeypatch.setenv("JARVIS_OPERAS_PERSIST_FILE", str(tmp_path / "persist.json"))
    index = tmp_path / "curve-index.json"
    cache = tmp_path / "curve-cache"
    monkeypatch.setenv("JARVIS_OPERAS_CURVE_CACHE_ROOT", str(cache))
    monkeypatch.setenv("JARVIS_OPERAS_CURVE_INDEX", str(index))
    if curves:
        from jarvis_operas.curves import init_builtin_curve_cache

        init_builtin_curve_cache(cache_root=str(cache), index_path=str(index))

    import jarvis_operas.api as api

    monkeypatch.setattr(api, "_global_operas_registry", None)
    monkeypatch.setattr(api, "_global_operas", None)
    for name in list(sys.modules):
        if name == "joperas" or name.startswith("joperas."):
            del sys.modules[name]


def test_importing_the_package_does_not_bootstrap_the_registry(tmp_path, monkeypatch) -> None:
    _reset_import_state(monkeypatch, tmp_path, curves=False)
    import jarvis_operas.api as api

    import joperas

    assert api._global_operas_registry is None
    assert joperas.math.add(1, 2) == 3
    assert api._global_operas_registry is not None


def test_builtin_functions_import_and_match_registry_calls(tmp_path, monkeypatch) -> None:
    _reset_import_state(monkeypatch, tmp_path, curves=False)
    from jarvis_operas import get_global_operas_registry
    from joperas.helper import eggbox
    from joperas.math import add, identity
    from joperas.pdg import mZ
    from joperas.stat import chi2_cov

    registry = get_global_operas_registry()
    assert add(1, 2) == 3
    assert add(a=2, b=5) == registry.call("math.add", a=2, b=5)
    assert identity(x={"k": 1}) == {"k": 1}
    assert mZ() == registry.call("pdg.mZ")
    assert chi2_cov(residual=[1.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]) == registry.call(
        "stat.chi2_cov",
        residual=[1.0, 0.0],
        cov=[[1.0, 0.0], [0.0, 1.0]],
    )
    assert eggbox(0.5, 0.0) == registry.call("helper.eggbox", 0.5, 0.0)
    assert list(inspect.signature(add).parameters) == ["a", "b"]
    assert str(inspect.signature(mZ)) == "()"
    assert "addition" in (add.__doc__ or "").lower()
    assert add.full_name == "math.add"


def test_star_import_exports_only_real_function_names(tmp_path, monkeypatch) -> None:
    _reset_import_state(monkeypatch, tmp_path, curves=False)
    namespace: dict[str, object] = {}
    exec("from joperas.math import *", namespace)
    assert namespace["add"](4, 5) == 9
    assert namespace["identity"]("ok") == "ok"
    assert set(namespace) >= {"add", "identity"}


def test_unknown_namespace_and_function_report_the_miss(tmp_path, monkeypatch) -> None:
    _reset_import_state(monkeypatch, tmp_path, curves=False)
    with pytest.raises(ModuleNotFoundError, match="math"):
        importlib.import_module("joperas.mat")
    with pytest.raises(ImportError, match="not_a_function"):
        exec("from joperas.math import not_a_function", {})


def test_user_operator_import_follows_registry_replacement(tmp_path, monkeypatch) -> None:
    _reset_import_state(monkeypatch, tmp_path, curves=False)
    from jarvis_operas import get_global_operas_registry, oper

    @oper("double", namespace="scriptops")
    def double(x):
        return x * 2

    @oper("bad-name", namespace="scriptops")
    def bad_name(x):
        return x + 1

    from joperas.scriptops import double as imported

    assert imported(21) == 42
    assert imported(x=3) == 6
    module = importlib.import_module("joperas.scriptops")
    assert getattr(module, "bad-name")(4) == 5
    assert "bad-name" not in module.__all__
    assert "double" in module.__all__

    registry = get_global_operas_registry()
    registry.register(
        OperaFunction(
            namespace="scriptops",
            name="double",
            arity=1,
            return_dtype=None,
            numpy_impl=lambda x: x * 3,
        ),
        replace=True,
    )
    assert imported(2) == 6


def test_every_registered_function_can_be_imported(tmp_path, monkeypatch) -> None:
    _reset_import_state(monkeypatch, tmp_path, curves=True)
    from jarvis_operas import get_global_operas_registry

    import joperas

    registry = get_global_operas_registry()
    namespaces = set()
    for full_name in registry.list():
        namespace, short_name = full_name.split(".", 1)
        namespaces.add(namespace)
        assert short_name.isidentifier() and not keyword.iskeyword(short_name), full_name
        module = importlib.import_module(f"joperas.{namespace}")
        function = getattr(module, short_name)
        assert function.full_name == full_name
        assert short_name in module.__all__
        assert short_name in dir(module)

    assert {"math", "stat", "helper", "pdg", "cmb", "HTs", "interp1", "dmddxn", "dmddxe"}.issubset(
        namespaces
    )
    assert namespaces.issubset(set(dir(joperas)))

    from joperas.dmddxn import LZSI2024
    from joperas.dmddxe import CDEXLM2022

    assert list(inspect.signature(LZSI2024).parameters) == ["x"]
    sample = registry.get("dmddxn.LZSI2024")
    x_value = (sample.metadata["x_min"] * sample.metadata["x_max"]) ** 0.5
    assert LZSI2024(x_value) == registry.call("dmddxn.LZSI2024", x_value)
    assert LZSI2024(x=x_value) == registry.call("dmddxn.LZSI2024", x=x_value)
    electron = registry.get("dmddxe.CDEXLM2022")
    electron_x = (electron.metadata["x_min"] * electron.metadata["x_max"]) ** 0.5
    assert CDEXLM2022(electron_x) == registry.call("dmddxe.CDEXLM2022", electron_x)

    with pytest.raises(ModuleNotFoundError, match="dmddxn") as missing:
        importlib.import_module("joperas.dmdd")
    assert "dmddxe" in str(missing.value)
