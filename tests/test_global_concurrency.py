from __future__ import annotations

import jarvis_operas.api as api_mod
from jarvis_operas.core.registry import recommended_numpy_concurrency


def test_recommended_numpy_concurrency_scales_with_cpu_count() -> None:
    assert recommended_numpy_concurrency(cpu_count=1) == 3
    assert recommended_numpy_concurrency(cpu_count=4) == 12
    assert recommended_numpy_concurrency(cpu_count=0) == 3
    assert recommended_numpy_concurrency(cpu_count=-3) == 3


def test_get_global_operas_keeps_first_numpy_concurrency(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("JARVIS_OPERAS_CURVE_INDEX", str(tmp_path / "curve-index.json"))
    monkeypatch.setenv("JARVIS_OPERAS_PERSIST_FILE", str(tmp_path / "persist-store.json"))
    monkeypatch.setattr(api_mod, "_global_operas_registry", None)
    monkeypatch.setattr(api_mod, "_global_operas", None)

    first = api_mod.get_global_operas(numpy_concurrency=2)
    assert first.registry.get_numpy_concurrency() == 2

    second = api_mod.get_global_operas(numpy_concurrency=5)
    assert second is first
    assert second.registry.get_numpy_concurrency() == 2


def test_get_global_operas_default_concurrency_is_cpu_scaled(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("JARVIS_OPERAS_CURVE_INDEX", str(tmp_path / "curve-index.json"))
    monkeypatch.setenv("JARVIS_OPERAS_PERSIST_FILE", str(tmp_path / "persist-store.json"))
    monkeypatch.setattr(api_mod, "_global_operas_registry", None)
    monkeypatch.setattr(api_mod, "_global_operas", None)

    operas = api_mod.get_global_operas()
    concurrency = operas.registry.get_numpy_concurrency()

    assert concurrency >= 3
    assert concurrency % 3 == 0
