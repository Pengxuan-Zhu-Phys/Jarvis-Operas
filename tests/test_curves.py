from __future__ import annotations

import json

import numpy as np

from jarvis_operas.curves import (
    get_curve_index_path,
    init_curve_cache,
    interpolation_manifest_resource,
    load_hot_curve_function_table,
    load_interpolation_manifest_library,
    register_hot_curves,
    register_hot_curves_in_registry,
)
from jarvis_operas.registry import OperatorRegistry


def _write_curve(path, x_values, y_values) -> None:
    path.write_text(
        json.dumps({"x": list(x_values), "y": list(y_values)}),
        encoding="utf-8",
    )


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
    assert payload["kind"] == "jarvis_operas_interpolation_manifest"
    assert isinstance(payload["curves"], list)
    assert interpolation_manifest_resource().endswith("manifests/interpolations.manifest.json")


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

    registry = OperatorRegistry()
    first = register_hot_curves_in_registry(registry, cache_root=str(cache_root))
    assert first == ["interp:LZSI2024"]
    assert "interp:LZSI2024" in registry.list(namespace="interp")

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
    assert second == ["dmdd:LZSI2024"]
    assert "dmdd:LZSI2024" in registry.list(namespace="dmdd")
    assert registry.list(namespace="interp") == []
