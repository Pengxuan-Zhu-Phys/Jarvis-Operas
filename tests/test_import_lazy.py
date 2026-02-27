from __future__ import annotations

import importlib


def test_package_reload_does_not_bootstrap_globals(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("JARVIS_OPERAS_CURVE_INDEX", str(tmp_path / "curve-index.json"))
    monkeypatch.setenv("JARVIS_OPERAS_PERSIST_FILE", str(tmp_path / "persist-store.json"))

    import jarvis_operas as jo
    import jarvis_operas.api as api_mod

    monkeypatch.setattr(api_mod, "_global_operas_registry", None)
    monkeypatch.setattr(api_mod, "_global_operas", None)
    jo.__dict__.pop("operas_registry", None)
    jo.__dict__.pop("operas", None)

    importlib.reload(jo)

    assert api_mod._global_operas_registry is None
    assert api_mod._global_operas is None


def test_lazy_operas_exports_bootstrap_on_access(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("JARVIS_OPERAS_CURVE_INDEX", str(tmp_path / "curve-index.json"))
    monkeypatch.setenv("JARVIS_OPERAS_PERSIST_FILE", str(tmp_path / "persist-store.json"))

    import jarvis_operas as jo
    import jarvis_operas.api as api_mod

    monkeypatch.setattr(api_mod, "_global_operas_registry", None)
    monkeypatch.setattr(api_mod, "_global_operas", None)
    jo.__dict__.pop("operas_registry", None)
    jo.__dict__.pop("operas", None)
    importlib.reload(jo)

    assert api_mod._global_operas_registry is None
    assert api_mod._global_operas is None

    registry = jo.operas_registry
    assert registry is api_mod.get_global_operas_registry()
    assert api_mod._global_operas_registry is registry

    operas = jo.operas
    assert operas is api_mod.get_global_operas()
    assert api_mod._global_operas is operas
