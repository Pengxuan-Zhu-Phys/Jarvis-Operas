from __future__ import annotations

import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jarvis_operas import OperaFunction, OperasRegistry, func_locals, numeric_funcs
from jarvis_operas.api import get_global_operas_registry
from jarvis_operas.curves import (
    add_interpolation_namespace,
    Curve1DInterpolator,
    get_curve_index_path,
    init_curve_cache,
    interpolation_manifest_resource,
    list_interpolation_namespace_entries,
    list_interpolation_namespaces,
    load_hot_curve_function_table,
    load_hot_curve_namespace_function_table,
    load_interpolation_manifest_library,
    remove_interpolation_namespace,
    register_hot_curves,
    register_hot_curve_namespace_in_registry,
    register_hot_curves_in_registry,
    validate_interpolation_namespace_index,
)
from jarvis_operas.integration import refresh_sympy_dicts_if_global_registry


def _write_curve(path, x_values, y_values) -> None:
    path.write_text(
        json.dumps({"x": list(x_values), "y": list(y_values)}),
        encoding="utf-8",
    )


def test_curve_interpolator_scalar_and_vector_calls_are_consistent() -> None:
    curve = Curve1DInterpolator(
        curve_id="flat_linear",
        x_values=np.array([1.0, 2.0, 3.0]),
        y_values=np.array([10.0, 20.0, 30.0]),
        kind="linear",
        log_x=False,
        log_y=False,
    )

    scalar = curve(2.5)
    vector = curve(np.array([2.5, 3.0]))

    assert isinstance(scalar, float)
    assert np.isclose(scalar, 25.0)
    assert isinstance(vector, np.ndarray)
    assert np.allclose(vector, np.array([25.0, 30.0]))


def test_curve_interpolator_clip_mode_applies_before_logx() -> None:
    curve = Curve1DInterpolator(
        curve_id="clip_logx",
        x_values=np.array([1.0, 2.0, 4.0, 8.0]),
        y_values=np.array([1.0, 4.0, 16.0, 64.0]),
        kind="linear",
        log_x=True,
        log_y=False,
        extrapolation="clip",
    )

    baseline = curve(1.0)
    assert np.isclose(curve(-10.0), baseline)
    assert np.isclose(curve(np.array([-10.0]))[0], baseline)


def test_curve_interpolator_pickle_roundtrip_keeps_call_behavior() -> None:
    curve = Curve1DInterpolator(
        curve_id="pickle_roundtrip",
        x_values=np.array([1.0, 2.0, 4.0, 8.0]),
        y_values=np.array([1.0, 2.0, 4.0, 8.0]),
        kind="linear",
        log_x=True,
        log_y=True,
        extrapolation="extrapolate",
    )
    restored = pickle.loads(pickle.dumps(curve, protocol=5))

    assert np.isclose(restored(3.0), curve(3.0))
    assert np.allclose(restored(np.array([2.0, 3.0, 5.0])), curve(np.array([2.0, 3.0, 5.0])))


def test_curve_interpolator_nan_mode_returns_nan_for_nan_and_out_of_bounds() -> None:
    curve = Curve1DInterpolator(
        curve_id="nan_mode",
        x_values=np.array([1.0, 2.0, 4.0, 8.0]),
        y_values=np.array([1.0, 2.0, 4.0, 8.0]),
        kind="linear",
        log_x=True,
        log_y=True,
        extrapolation="nan",
    )

    assert np.isnan(curve(np.nan))
    assert np.isnan(curve(0.5))
    values = curve(np.array([np.nan, 0.5, 2.0]))
    assert values.shape == (3,)
    assert np.isnan(values[0])
    assert np.isnan(values[1])
    assert np.isfinite(values[2])


def test_init_curve_cache_and_load_hot_functions(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_curve(source_dir / "hot.json", [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    _write_curve(source_dir / "cold.json", [0.0, 1.0, 2.0], [0.0, 2.0, 4.0])

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "curves": [
                    {
                        "curve_id": "line_hot",
                        "source": "source/hot.json",
                        "kind": "linear",
                        "hot": True,
                    },
                    {
                        "curve_id": "line_cold",
                        "source": "source/cold.json",
                        "kind": "linear",
                        "hot": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    cache_root = tmp_path / "cache"
    summary = init_curve_cache(str(manifest_path), cache_root=str(cache_root))
    assert summary["total"] == 2
    assert sorted(summary["compiled"]) == ["line_cold", "line_hot"]
    assert get_curve_index_path(cache_root=str(cache_root)).exists()

    hot_table = load_hot_curve_function_table(cache_root=str(cache_root))
    assert set(hot_table.keys()) == {"line_hot"}
    assert np.isclose(hot_table["line_hot"](1.5), 1.5)

    full_table = load_hot_curve_function_table(cache_root=str(cache_root), include_cold=True)
    assert set(full_table.keys()) == {"line_hot", "line_cold"}
    assert np.isclose(full_table["line_cold"](1.5), 3.0)


def test_init_curve_cache_reuses_existing_pickle(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_curve(source_dir / "curve.json", [0.0, 1.0, 2.0], [0.0, 1.0, 4.0])

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
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

    cache_root = tmp_path / "cache"
    first = init_curve_cache(str(manifest_path), cache_root=str(cache_root))
    second = init_curve_cache(str(manifest_path), cache_root=str(cache_root))

    assert first["compiled"] == ["quadratic"]
    assert second["compiled"] == []
    assert second["cached"] == ["quadratic"]


def test_register_hot_curves_updates_function_table(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_curve(source_dir / "curve.json", [0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "curves": [
                    {
                        "curve_id": "offset_line",
                        "source": "source/curve.json",
                        "kind": "linear",
                        "hot": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cache_root = tmp_path / "cache"
    init_curve_cache(str(manifest_path), cache_root=str(cache_root))

    function_table: dict[str, object] = {}
    updated = register_hot_curves(function_table, cache_root=str(cache_root))

    assert updated == ["offset_line"]
    assert "offset_line" in function_table
    assert callable(function_table["offset_line"])


def test_packaged_interpolation_manifest_library_is_available() -> None:
    payload = load_interpolation_manifest_library()
    assert payload["kind"] == "jarvis_operas_interpolation_namespace_index"
    assert isinstance(payload["namespaces"], list)
    assert isinstance(payload["curves"], list)
    assert interpolation_manifest_resource().endswith("manifests/interpolations.manifest.json")


def test_packaged_interpolation_manifest_sources_exist() -> None:
    payload = load_interpolation_manifest_library()
    base = resources.files("jarvis_operas").joinpath("manifests")
    for item in payload["curves"]:
        source = item.get("source")
        assert isinstance(source, str) and source.strip()
        path = base.joinpath(Path(source))
        assert path.is_file()


def test_list_interpolation_namespaces_works() -> None:
    namespaces = list_interpolation_namespaces()
    assert "interp1" in namespaces
    assert "dmdd" in namespaces


def test_init_curve_cache_can_filter_namespaces(tmp_path) -> None:
    base = resources.files("jarvis_operas").joinpath("manifests")
    manifest_path = base.joinpath("interpolations.manifest.json")

    index_path = tmp_path / "cache" / "index.json"

    summary = init_curve_cache(
        str(manifest_path),
        index_path=str(index_path),
        namespaces=["dmdd"],
        force=True,
    )
    assert summary["total"] == 1
    table = load_hot_curve_function_table(index_path=str(index_path))
    assert set(table.keys()) == {"LZSI2024"}


def test_register_hot_curves_in_registry_uses_group_namespace_and_updates(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_curve(source_dir / "curve.json", [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "curves": [
                    {
                        "curve_id": "LZSI2024",
                        "source": "source/curve.json",
                        "kind": "cubic",
                        "logX": True,
                        "logY": True,
                        "hot": True,
                        "metadata": {"group": "interp"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cache_root = tmp_path / "cache"
    init_curve_cache(str(manifest_path), cache_root=str(cache_root))

    registry = OperasRegistry()
    first = register_hot_curves_in_registry(registry, cache_root=str(cache_root))
    assert first == ["interp.LZSI2024"]
    assert "interp.LZSI2024" in registry.list(namespace="interp")

    # Move group to dmdd and re-init; registry update should replace old namespace item.
    manifest_path.write_text(
        json.dumps(
            {
                "curves": [
                    {
                        "curve_id": "LZSI2024",
                        "source": "source/curve.json",
                        "kind": "cubic",
                        "logX": True,
                        "logY": True,
                        "hot": True,
                        "metadata": {"group": "dmdd"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    init_curve_cache(str(manifest_path), cache_root=str(cache_root))
    second = register_hot_curves_in_registry(registry, cache_root=str(cache_root))
    assert second == ["dmdd.LZSI2024"]
    assert "dmdd.LZSI2024" in registry.list(namespace="dmdd")
    assert registry.list(namespace="interp") == []


def test_register_hot_curves_in_registry_overwrite_keeps_non_interpolation_same_short_name(
    tmp_path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_curve(source_dir / "curve.json", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "curves": [
                    {
                        "curve_id": "LZSI2024",
                        "source": "source/curve.json",
                        "kind": "linear",
                        "hot": True,
                        "metadata": {"group": "dmdd"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cache_root = tmp_path / "cache"
    init_curve_cache(str(manifest_path), cache_root=str(cache_root))

    registry = OperasRegistry()
    registry.register(
        OperaFunction(
            namespace="user_ops",
            name="LZSI2024",
            arity=1,
            return_dtype=None,
            numpy_impl=lambda x: x,
            metadata={"category": "user_defined"},
        )
    )

    loaded = register_hot_curves_in_registry(
        registry,
        cache_root=str(cache_root),
        overwrite=True,
    )

    assert loaded == ["dmdd.LZSI2024"]
    assert "dmdd.LZSI2024" in registry.list(namespace="dmdd")
    assert "user_ops.LZSI2024" in registry.list(namespace="user_ops")


def test_register_hot_curves_in_global_registry_refreshes_sympy_dicts(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_curve(source_dir / "curve.json", [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0])

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "curves": [
                    {
                        "curve_id": "AutoRefreshCurve",
                        "source": "source/curve.json",
                        "kind": "cubic",
                        "hot": True,
                        "metadata": {"group": "dmdd_refresh"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    index_path = tmp_path / "cache" / "index.json"
    init_curve_cache(str(manifest_path), index_path=str(index_path))

    registry = get_global_operas_registry()
    registry.delete_namespace("dmdd_refresh")
    refresh_sympy_dicts_if_global_registry(registry)
    full_name = "dmdd_refresh.AutoRefreshCurve"
    assert "dmdd_refresh" not in func_locals

    try:
        loaded = register_hot_curves_in_registry(registry, index_path=str(index_path))
        assert full_name in loaded
        assert "dmdd_refresh" in func_locals
        symbol_name = str(getattr(func_locals["dmdd_refresh"], "AutoRefreshCurve"))
        assert symbol_name in numeric_funcs
    finally:
        registry.delete_namespace("dmdd_refresh")
        refresh_sympy_dicts_if_global_registry(registry)


def test_register_hot_curves_in_registry_can_filter_namespaces(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_curve(source_dir / "curve_a.json", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    _write_curve(source_dir / "curve_b.json", [1.0, 2.0, 3.0], [2.0, 3.0, 4.0])

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "kind": "jarvis_operas_interpolation_namespace_index",
                "namespaces": [
                    {
                        "namespace": "ns1",
                        "manifest": "ns1.json",
                    },
                    {
                        "namespace": "ns2",
                        "manifest": "ns2.json",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "ns1.json").write_text(
        json.dumps(
            {
                "namespace": "ns1",
                "functions": [
                    {
                        "name": "f1",
                        "source": "source/curve_a.json",
                        "kind": "linear",
                        "hot": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "ns2.json").write_text(
        json.dumps(
            {
                "namespace": "ns2",
                "functions": [
                    {
                        "name": "f2",
                        "source": "source/curve_b.json",
                        "kind": "linear",
                        "hot": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    index_path = tmp_path / "cache" / "index.json"
    init_curve_cache(str(manifest_path), index_path=str(index_path))

    registry = OperasRegistry()
    loaded = register_hot_curves_in_registry(
        registry,
        index_path=str(index_path),
        namespaces=["ns2"],
    )
    assert loaded == ["ns2.f2"]
    assert registry.list(namespace="ns1") == []
    assert registry.list(namespace="ns2") == ["ns2.f2"]


def test_namespace_specific_curve_helpers(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_curve(source_dir / "curve_a.json", [0.0, 1.0], [0.0, 1.0])
    _write_curve(source_dir / "curve_b.json", [0.0, 1.0], [1.0, 2.0])

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "kind": "jarvis_operas_interpolation_namespace_index",
                "namespaces": [
                    {"namespace": "alpha", "manifest": "alpha.json"},
                    {"namespace": "beta", "manifest": "beta.json"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "alpha.json").write_text(
        json.dumps(
            {
                "namespace": "alpha",
                "functions": [
                    {"name": "f1", "source": "source/curve_a.json", "kind": "linear", "hot": True}
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "beta.json").write_text(
        json.dumps(
            {
                "namespace": "beta",
                "functions": [
                    {"name": "f2", "source": "source/curve_b.json", "kind": "linear", "hot": True}
                ],
            }
        ),
        encoding="utf-8",
    )
    index_path = tmp_path / "cache" / "index.json"
    init_curve_cache(str(manifest_path), index_path=str(index_path))

    alpha_table = load_hot_curve_namespace_function_table("alpha", index_path=str(index_path))
    beta_table = load_hot_curve_namespace_function_table("beta", index_path=str(index_path))
    assert set(alpha_table.keys()) == {"f1"}
    assert set(beta_table.keys()) == {"f2"}

    registry = OperasRegistry()
    loaded = register_hot_curve_namespace_in_registry(
        registry,
        "beta",
        index_path=str(index_path),
    )
    assert loaded == ["beta.f2"]
    assert registry.list(namespace="alpha") == []
    assert registry.list(namespace="beta") == ["beta.f2"]


def test_interpolation_namespace_entry_management_and_validation(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_curve(source_dir / "a.json", [0.0, 1.0], [0.0, 1.0])
    _write_curve(source_dir / "b.json", [0.0, 1.0], [1.0, 2.0])
    _write_curve(source_dir / "c.json", [0.0, 1.0], [2.0, 3.0])

    root_manifest = tmp_path / "interpolations.manifest.json"
    root_manifest.write_text(
        json.dumps(
            {
                "version": 2,
                "kind": "jarvis_operas_interpolation_namespace_index",
                "namespaces": [
                    {"namespace": "alpha", "manifest": "alpha.manifest.json"}
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "alpha.manifest.json").write_text(
        json.dumps(
            {
                "namespace": "alpha",
                "functions": [
                    {"name": "a1", "source": "source/a.json", "kind": "linear", "hot": True}
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "beta.manifest.json").write_text(
        json.dumps(
            {
                "namespace": "beta",
                "functions": [
                    {"name": "b1", "source": "source/b.json", "kind": "linear", "hot": True}
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "beta2.manifest.json").write_text(
        json.dumps(
            {
                "namespace": "beta",
                "functions": [
                    {"name": "b2", "source": "source/c.json", "kind": "linear", "hot": True}
                ],
            }
        ),
        encoding="utf-8",
    )

    add_result = add_interpolation_namespace(
        manifest_path=str(root_manifest),
        namespace="beta",
        namespace_manifest="beta.manifest.json",
        description="beta namespace",
    )
    assert add_result["added"] is True
    assert add_result["updated"] is False
    assert add_result["namespace"] == "beta"

    unchanged = add_interpolation_namespace(
        manifest_path=str(root_manifest),
        namespace="beta",
        namespace_manifest="beta.manifest.json",
        description="beta namespace",
    )
    assert unchanged["unchanged"] is True

    with pytest.raises(ValueError):
        add_interpolation_namespace(
            manifest_path=str(root_manifest),
            namespace="beta",
            namespace_manifest="beta2.manifest.json",
        )

    updated = add_interpolation_namespace(
        manifest_path=str(root_manifest),
        namespace="beta",
        namespace_manifest="beta2.manifest.json",
        overwrite=True,
    )
    assert updated["updated"] is True

    entries = list_interpolation_namespace_entries(manifest_path=str(root_manifest))
    assert [item["namespace"] for item in entries] == ["alpha", "beta"]
    assert entries[1]["manifest"] == "beta2.manifest.json"

    report = validate_interpolation_namespace_index(manifest_path=str(root_manifest))
    assert report["ok"] is True
    assert report["namespace_count"] == 2
    assert report["function_count"] == 2

    remove_result = remove_interpolation_namespace(
        manifest_path=str(root_manifest),
        namespace="beta",
    )
    assert remove_result["removed"] is True
    assert remove_result["remaining"] == 1

    missing_remove = remove_interpolation_namespace(
        manifest_path=str(root_manifest),
        namespace="beta",
    )
    assert missing_remove["removed"] is False


def test_interpolation_function_supports_async_and_polars_backend(tmp_path) -> None:
    pl = pytest.importorskip("polars")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_curve(source_dir / "curve.json", [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "curves": [
                    {
                        "curve_id": "line",
                        "source": "source/curve.json",
                        "kind": "linear",
                        "hot": True,
                        "metadata": {"group": "interp"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    index_path = tmp_path / "cache" / "index.json"
    init_curve_cache(str(manifest_path), index_path=str(index_path))

    registry = OperasRegistry()
    register_hot_curves_in_registry(registry, index_path=str(index_path))

    sync_value = registry.call("interp.line", x=np.array([0.5, 1.5]))
    assert np.allclose(sync_value, np.array([0.5, 1.5]))

    import asyncio

    async_value = asyncio.run(registry.acall("interp.line", x=np.array([0.25, 1.25])))
    assert np.allclose(async_value, np.array([0.25, 1.25]))

    expr = registry.call("interp.line", pl.col("x"), backend="polars")
    frame = pl.DataFrame({"x": [0.0, 1.0, 2.0]})
    values = frame.select(expr.alias("y")).get_column("y").to_list()
    assert values == [0.0, 1.0, 2.0]


def test_interpolation_info_note_contains_input_range(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_curve(source_dir / "curve.json", [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "curves": [
                    {
                        "curve_id": "line_note",
                        "source": "source/curve.json",
                        "kind": "linear",
                        "hot": True,
                        "metadata": {
                            "group": "interp",
                            "note": "Custom interpolation note.",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    index_path = tmp_path / "cache" / "index.json"
    init_curve_cache(str(manifest_path), index_path=str(index_path))

    registry = OperasRegistry()
    register_hot_curves_in_registry(registry, index_path=str(index_path))
    info = registry.info("interp.line_note")
    note = info["metadata"].get("note")
    assert isinstance(note, str)
    assert note == "Custom interpolation note.\n\tInput range: x in [0, 2]."


def test_init_curve_cache_concurrent_writes_are_atomic(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_curve(source_dir / "curve.json", [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "curves": [
                    {
                        "curve_id": "line",
                        "source": "source/curve.json",
                        "kind": "linear",
                        "hot": True,
                        "metadata": {"group": "interp"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    index_path = tmp_path / "cache" / "index.json"

    def _run_once(_: int) -> dict[str, Any]:
        return init_curve_cache(
            str(manifest_path),
            index_path=str(index_path),
            force=True,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_run_once, range(16)))

    assert results
    assert index_path.exists()
    table = load_hot_curve_function_table(index_path=str(index_path))
    assert "line" in table
    assert np.isclose(float(table["line"](1.5)), 1.5)

    temp_files = list(index_path.parent.glob("*.tmp"))
    assert temp_files == []


def test_registered_curve_reuses_prepared_interpolator(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_curve(source_dir / "curve.json", [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "curves": [
                    {
                        "curve_id": "line",
                        "source": "source/curve.json",
                        "kind": "linear",
                        "hot": True,
                        "metadata": {"group": "interp"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    index_path = tmp_path / "cache" / "index.json"
    init_curve_cache(str(manifest_path), index_path=str(index_path), force=True)

    registry = OperasRegistry()
    register_hot_curves_in_registry(registry, index_path=str(index_path))
    declaration = registry.get("interp.line")
    curve_fn = declaration.numpy_impl
    assert isinstance(curve_fn, Curve1DInterpolator)
    interp_id = id(curve_fn._interp)

    curve_fn._build_interp = lambda: (_ for _ in ()).throw(AssertionError("should not rebuild"))  # type: ignore[method-assign]

    first = registry.call("interp.line", x=0.5)
    second = registry.call("interp.line", x=np.array([0.5, 1.5]))
    assert np.isclose(float(first), 0.5)
    assert np.allclose(second, np.array([0.5, 1.5]))
    assert id(curve_fn._interp) == interp_id
    assert registry.get("interp.line").numpy_impl is curve_fn


def test_dmdd_curve_returns_nan_on_out_of_bounds_and_nan_input() -> None:
    base = resources.files("jarvis_operas").joinpath("manifests")
    manifest_path = base.joinpath("interpolations.manifest.json")

    # Compile against current packaged manifests to avoid stale local cache behavior.
    temp_index = Path.cwd() / ".pytest-dmdd-index.json"
    try:
        init_curve_cache(
            str(manifest_path),
            index_path=str(temp_index),
            namespaces=["dmdd"],
            force=True,
        )

        registry = OperasRegistry()
        loaded = register_hot_curves_in_registry(registry, index_path=str(temp_index))
        assert "dmdd.LZSI2024" in loaded

        scalar = registry.call("dmdd.LZSI2024", x=1.0)
        assert np.isnan(float(scalar))

        mixed = registry.call("dmdd.LZSI2024", x=np.array([1.0, np.nan, 1.0e3]))
        assert mixed.shape == (3,)
        assert np.isnan(mixed[0])
        assert np.isnan(mixed[1])
        assert np.isfinite(mixed[2])
    finally:
        if temp_index.exists():
            temp_index.unlink()
        temp_curve_dir = temp_index.parent / "curves"
        if temp_curve_dir.exists():
            for child in temp_curve_dir.glob("*"):
                child.unlink()
            temp_curve_dir.rmdir()


def test_global_registry_bootstrap_refreshes_builtin_curve_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "curve-cache"
    monkeypatch.setenv("JARVIS_OPERAS_CURVE_CACHE_ROOT", str(cache_root))

    manifests_resource = resources.files("jarvis_operas").joinpath("manifests")
    with resources.as_file(manifests_resource) as manifests_dir:
        stale_namespace_manifest = tmp_path / "stale-dmdd.manifest.json"
        stale_namespace_manifest.write_text(
            json.dumps(
                {
                    "version": 1,
                    "kind": "jarvis_operas_interpolation_namespace_manifest",
                    "namespace": "dmdd",
                    "functions": [
                        {
                            "name": "LZSI2024",
                            "source": str((manifests_dir / "dmdd" / "LZSI_2024.json").resolve()),
                            "kind": "cubic",
                            "logX": True,
                            "logY": True,
                            "extrapolation": "extrapolate",
                            "hot": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        stale_root_manifest = tmp_path / "stale.manifest.json"
        stale_root_manifest.write_text(
            json.dumps(
                {
                    "version": 2,
                    "kind": "jarvis_operas_interpolation_namespace_index",
                    "namespaces": [
                        {
                            "namespace": "dmdd",
                            "manifest": str(stale_namespace_manifest.resolve()),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        init_curve_cache(str(stale_root_manifest), force=True)

    stale_table = load_hot_curve_function_table(cache_root=str(cache_root))
    assert "LZSI2024" in stale_table
    assert np.isinf(float(stale_table["LZSI2024"](1.0)))

    import jarvis_operas.api as api_mod

    monkeypatch.setattr(api_mod, "_global_operas_registry", None)
    monkeypatch.setattr(api_mod, "_global_operas", None)
    registry = api_mod.get_global_operas_registry()

    value = registry.call("dmdd.LZSI2024", x=1.0)
    assert np.isnan(float(value))
