"""Home directory, curve artifacts, and unit-call dispatch."""

from __future__ import annotations

import json

import numpy as np
import pytest

from jarvis_operas import OperaFunction, OperasRegistry
from jarvis_operas.curves import init_curve_cache, load_hot_curve_function_table


def _isolate_home(monkeypatch, tmp_path):
    import jarvis_operas.home as home

    monkeypatch.setattr(home, "user_home", lambda: tmp_path)
    for name in (
        "JARVIS_HOME",
        "JARVIS_OPERAS_HOME",
        "JARVIS_OPERAS_CURVE_CACHE_ROOT",
        "JARVIS_OPERAS_CURVE_INDEX",
        "JARVIS_OPERAS_SOURCES_FILE",
        "JARVIS_OPERAS_OVERRIDES_FILE",
        "JARVIS_OPERAS_USER_SOURCE_DIR",
        "JARVIS_OPERAS_PERSIST_FILE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_state_paths_use_jarvis_operas_and_ignore_the_old_directory(tmp_path, monkeypatch) -> None:
    _isolate_home(monkeypatch, tmp_path)
    legacy = tmp_path / ".jarvis-operas"
    legacy.mkdir()
    (legacy / "sources.json").write_text('{"entries": [{"path": "old"}]}\n', encoding="utf-8")
    (legacy / "curve-cache").mkdir()
    (legacy / "curve-cache" / "index.json").write_text("{}\n", encoding="utf-8")

    from jarvis_operas.curve_storage import get_curve_cache_root
    from jarvis_operas.persistence import get_sources_store_path, get_user_source_store_dir

    operas = (tmp_path / ".jarvis" / "operas").resolve()
    assert get_curve_cache_root() == operas / "curve-cache"
    assert get_sources_store_path() == operas / "sources.json"
    assert get_user_source_store_dir() == operas / "user_sources"
    assert (legacy / "sources.json").is_file()
    assert not (operas / "sources.json").exists()


def test_curve_cache_writes_npz_and_rebuilds_retired_pickle(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "curve.json").write_text(
        json.dumps({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 4.0]}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "curves": [
                    {
                        "curve_id": "quadratic",
                        "source": "source/curve.json",
                        "kind": "linear",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    init_curve_cache(str(manifest), cache_root=str(cache))
    artifacts = list((cache / "curves").iterdir())
    assert [path.suffix for path in artifacts] == [".npz"]
    with np.load(artifacts[0], allow_pickle=False) as payload:
        assert payload["x"].shape == (3,)

    retired = cache / "curves" / "old.pkl"
    retired.write_bytes(b"this must not be unpickled")
    index_path = cache / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["curves"]["quadratic"] = {
        "curve_id": "quadratic",
        "format": "pickle-v1",
        "pickle": "curves/old.pkl",
        "hash": "stale",
    }
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="retired pickle"):
        load_hot_curve_function_table(cache_root=str(cache))

    rebuilt = init_curve_cache(str(manifest), cache_root=str(cache))
    assert rebuilt["compiled"] == ["quadratic"]
    assert not retired.exists()
    table = load_hot_curve_function_table(cache_root=str(cache))
    assert np.isclose(table["quadratic"](1.0), 1.0)


def test_object_backend_reaches_the_operator_and_unit_plan_is_reused(monkeypatch) -> None:
    import jarvis_operas.core.registry as registry_mod

    calls = {"n": 0}
    original = registry_mod.prepare_unit_call

    def counting(declaration, **kwargs):
        if not kwargs:
            calls["n"] += 1
        return original(declaration, **kwargs)

    monkeypatch.setattr(registry_mod, "prepare_unit_call", counting)
    registry = OperasRegistry()
    token = object()
    seen = {}

    def build(prediction, backend):
        seen["backend"] = backend
        return prediction + 1

    def add_radians(x, y):
        return float(x) + float(y)

    registry.register(
        OperaFunction(
            namespace="probe",
            name="build",
            arity=None,
            return_dtype=None,
            numpy_impl=build,
        )
    )
    registry.register(
        OperaFunction(
            namespace="probe",
            name="add_radians",
            arity=2,
            return_dtype=None,
            numpy_impl=add_radians,
            metadata={
                "unit_contract": {
                    "version": "1.0",
                    "inputs": {
                        "x": {"unit": "rad", "shape": "scalar", "nullable": False},
                        "y": {"unit": "rad", "shape": "scalar", "nullable": False},
                    },
                    "outputs": {
                        "": {"unit": "rad", "shape": "scalar", "nullable": False},
                    },
                }
            },
        )
    )
    assert registry.call("probe.build", 1, backend=token) == 2
    assert seen["backend"] is token
    assert registry.call("probe.add_radians", x=1.0, y=2.0) == 3.0
    assert registry.call("probe.add_radians", x=1.0, y=2.0) == 3.0
    assert calls["n"] == 1
