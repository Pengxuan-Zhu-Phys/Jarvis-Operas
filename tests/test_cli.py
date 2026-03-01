from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

from jarvis_operas import cli as cli_mod


def _run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    return subprocess.run(
        [sys.executable, "-m", "jarvis_operas.cli", *args],
        check=False,
        capture_output=True,
        text=True,
        env=merged_env,
    )


def test_cli_list_contains_builtin_add() -> None:
    result = _run_cli("list", "--namespace", "math", "--json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert any(item["name"] == "math.add" for item in payload)
    assert all(re.match(r"^[a-z][0-9]{3,5}$", item["id"]) for item in payload)


def test_cli_list_human_output_groups_namespace() -> None:
    result = _run_cli("list", "--namespace", "math")

    assert result.returncode == 0, result.stderr
    assert "math (" in result.stdout
    assert "math.add" in result.stdout
    assert "\t" in result.stdout
    assert re.search(r"\b[a-z][0-9]{3,5}\b", result.stdout)


def test_cli_no_args_shows_start_card() -> None:
    result = _run_cli()

    assert result.returncode == 0, result.stderr
    assert "Start here:" in result.stdout
    assert "jopera list" in result.stdout


def test_cli_help_examples() -> None:
    result = _run_cli("help", "examples")

    assert result.returncode == 0, result.stderr
    assert "Quick examples:" in result.stdout
    assert "jopera info stat.chi2_cov" in result.stdout


def test_cli_help_does_not_show_discover() -> None:
    result = _run_cli("--help")

    assert result.returncode == 0, result.stderr
    assert "discover" not in result.stdout
    assert "update" not in result.stdout
    assert "delete-func" not in result.stdout
    assert "delete-namespace" not in result.stdout


def test_cli_call_add_with_kwargs() -> None:
    result = _run_cli("call", "math.add", "--kwargs", '{"a": 2, "b": 3}')

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "5"


def test_cli_load_and_call_user_op(tmp_path) -> None:
    op_file = tmp_path / "cli_ops.py"
    op_file.write_text(
        textwrap.dedent(
            """
            def triple(x):
                return x * 3

            __JARVIS_OPERAS__ = {
                "triple": triple,
            }
            """
        ),
        encoding="utf-8",
    )

    result = _run_cli(
        "call",
        "cli_ops.triple",
        "--user-ops",
        str(op_file),
        "--arg",
        "x=4",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "12"


def test_cli_version_short_flag() -> None:
    result = _run_cli("-v")

    assert result.returncode == 0, result.stderr
    assert re.match(r"^jopera\s+\S+\s*$", result.stdout)


def test_cli_not_found_shows_suggestion() -> None:
    result = _run_cli("info", "helper.egbox")

    assert result.returncode == 1
    assert "Function 'helper.egbox' not found." in result.stderr
    assert "helper.eggbox" in result.stderr


def test_cli_debug_flag_supported_before_and_after_command(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(cli_mod, "configure_cli_logger", lambda mode: calls.append(mode))
    monkeypatch.setattr(cli_mod, "_cmd_help", lambda args: 0)

    assert cli_mod.main(["-d", "help", "examples"]) == 0
    assert cli_mod.main(["help", "examples", "-d"]) == 0
    assert calls == ["debug", "debug"]


def test_cli_info_shows_operator_id() -> None:
    result = _run_cli("info", "math.add", "--json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "id" in payload
    assert re.match(r"^[a-z][0-9]{3,5}$", payload["id"])
    assert payload["supports_async"] is True


def test_print_info_human_hides_metadata_and_shows_note(capsys) -> None:
    info = {
        "name": "dmdd.LZSI2024",
        "id": "d12345",
        "namespace": "dmdd",
        "signature": "(x)",
        "supports_async": True,
        "docstring": "LZ 2024 SI upper limit.",
        "metadata": {
            "category": "interpolation",
            "note": "From arXiv:2410.17036",
            "source": "/tmp/LZSI_2024.json",
        },
    }
    cli_mod._print_info_human(info)
    out = capsys.readouterr().out

    assert "Summary:\tLZ 2024 SI upper limit." in out
    assert "Note:\tFrom arXiv:2410.17036" in out
    assert "Metadata:\t" not in out


def test_print_info_human_uses_metadata_examples_for_try_next(capsys) -> None:
    info = {
        "name": "helper.eggbox",
        "id": "h12345",
        "namespace": "helper",
        "signature": "(x, y)",
        "supports_async": True,
        "docstring": "Evaluate EggBox benchmark surface z(x, y).",
        "metadata": {
            "examples": [
                "jopera call helper.eggbox --kwargs '{\"x\":0.5,\"y\":0.0}'",
            ],
        },
    }
    cli_mod._print_info_human(info)
    out = capsys.readouterr().out
    assert "Try next:" in out
    assert "jopera call helper.eggbox --kwargs '{\"x\":0.5,\"y\":0.0}'" in out


def test_print_info_human_without_examples_hides_try_next(capsys) -> None:
    info = {
        "name": "math.add",
        "id": "m12345",
        "namespace": "math",
        "signature": "(a, b)",
        "supports_async": True,
        "docstring": "Add numbers.",
        "metadata": {},
    }
    cli_mod._print_info_human(info)
    out = capsys.readouterr().out
    assert "Try next:" not in out
    assert "UserSources:\t" not in out
    assert "UserOverrides:\t" not in out


def test_print_info_human_cmb_namespace_hides_user_store_paths(capsys) -> None:
    info = {
        "name": "cmb.loglike",
        "id": "c12345",
        "namespace": "cmb",
        "signature": "(payload=None, **params)",
        "supports_async": True,
        "docstring": "Evaluate CMB TT log-likelihood.",
        "metadata": {},
    }
    cli_mod._print_info_human(info)
    out = capsys.readouterr().out
    assert "UserSources:\t" not in out
    assert "UserOverrides:\t" not in out


def test_print_info_human_user_function_shows_user_store_paths(capsys, monkeypatch, tmp_path) -> None:
    sources_path = tmp_path / "sources.json"
    overrides_path = tmp_path / "overrides.json"
    monkeypatch.setenv("JARVIS_OPERAS_SOURCES_FILE", str(sources_path))
    monkeypatch.setenv("JARVIS_OPERAS_OVERRIDES_FILE", str(overrides_path))

    info = {
        "name": "my_ops.foo",
        "id": "u12345",
        "namespace": "my_ops",
        "signature": "(x)",
        "supports_async": True,
        "docstring": "User op.",
        "metadata": {},
    }
    cli_mod._print_info_human(info)
    out = capsys.readouterr().out
    assert f"UserSources:\t{sources_path}" in out
    assert f"UserOverrides:\t{overrides_path}" in out


def test_print_value_serializes_numpy_arrays_as_json_arrays(capsys) -> None:
    cli_mod._print_value(
        {
            "model_cl": np.array([1.0, 2.0, 3.0]),
            "meta": {"n": np.int64(3)},
        },
        as_json=False,
    )
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["model_cl"] == [1.0, 2.0, 3.0]
    assert payload["meta"]["n"] == 3


def test_cli_load_persists_for_new_process(tmp_path) -> None:
    op_file = tmp_path / "persist_cli_ops.py"
    op_file.write_text(
        textwrap.dedent(
            """
            def plus_one(x):
                return x + 1

            __JARVIS_OPERAS__ = {
                "plus_one": plus_one,
            }
            """
        ),
        encoding="utf-8",
    )
    store_path = tmp_path / "persist_store.json"
    env = {"JARVIS_OPERAS_PERSIST_FILE": str(store_path)}

    first = _run_cli("load", str(op_file), env=env)
    assert first.returncode == 0, first.stderr

    op_file.unlink()

    second = _run_cli("call", "persist_cli_ops.plus_one", "--arg", "x=10", env=env)
    assert second.returncode == 0, second.stderr
    assert second.stdout.strip() == "11"

    third = _run_cli("acall", "persist_cli_ops.plus_one", "--arg", "x=10", env=env)
    assert third.returncode == 0, third.stderr
    assert third.stdout.strip() == "11"

    info_result = _run_cli("info", "persist_cli_ops.plus_one", "--json", env=env)
    assert info_result.returncode == 0, info_result.stderr
    info_payload = json.loads(info_result.stdout)
    assert "user_store" in info_payload
    assert info_payload["user_store"]["sources_path"].endswith("sources.json")
    assert info_payload["user_store"]["overrides_path"].endswith("overrides.json")


def test_cli_load_session_only_does_not_persist(tmp_path) -> None:
    op_file = tmp_path / "ephemeral_cli_ops.py"
    op_file.write_text(
        textwrap.dedent(
            """
            def plus_two(x):
                return x + 2

            __JARVIS_OPERAS__ = {
                "plus_two": plus_two,
            }
            """
        ),
        encoding="utf-8",
    )
    store_path = tmp_path / "persist_store.json"
    env = {"JARVIS_OPERAS_PERSIST_FILE": str(store_path)}

    first = _run_cli("load", str(op_file), "--session-only", env=env)
    assert first.returncode == 0, first.stderr

    second = _run_cli("call", "ephemeral_cli_ops.plus_two", "--arg", "x=10", env=env)
    assert second.returncode == 1
    assert "not found" in second.stderr


def test_cli_init_curve_cache(tmp_path) -> None:
    curve_dir = tmp_path / "curves"
    curve_dir.mkdir()
    (curve_dir / "line.json").write_text(
        json.dumps({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "curves": [
                    {
                        "curve_id": "line_curve",
                        "source": "curves/line.json",
                        "kind": "linear",
                        "hot": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cache_root = tmp_path / "curve-cache"

    result = _run_cli(
        "init",
        "--manifest",
        str(manifest_path),
        "--cache-root",
        str(cache_root),
        "--json",
    )
    assert result.returncode == 0, result.stderr

    payload = json.loads(result.stdout)
    assert payload["total"] == 1
    assert payload["compiled"] == ["line_curve"]
    assert Path(payload["index_path"]).exists()


def test_cli_init_curve_cache_without_manifest_uses_packaged_library(tmp_path) -> None:
    cache_root = tmp_path / "curve-cache"
    result = _run_cli(
        "init",
        "--cache-root",
        str(cache_root),
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["total"] >= 0
    assert Path(payload["index_path"]).exists()


def test_cli_init_curve_cache_with_namespace_filter(tmp_path) -> None:
    curve_dir = tmp_path / "curves"
    curve_dir.mkdir()
    (curve_dir / "a.json").write_text(
        json.dumps({"x": [0.0, 1.0], "y": [0.0, 1.0]}),
        encoding="utf-8",
    )
    (curve_dir / "b.json").write_text(
        json.dumps({"x": [0.0, 1.0], "y": [1.0, 2.0]}),
        encoding="utf-8",
    )

    (tmp_path / "ns1.json").write_text(
        json.dumps(
            {
                "namespace": "ns1",
                "functions": [
                    {"name": "fa", "source": "curves/a.json", "kind": "linear", "hot": True}
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
                    {"name": "fb", "source": "curves/b.json", "kind": "linear", "hot": True}
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "kind": "jarvis_operas_interpolation_namespace_index",
                "namespaces": [
                    {"namespace": "ns1", "manifest": "ns1.json"},
                    {"namespace": "ns2", "manifest": "ns2.json"},
                ],
            }
        ),
        encoding="utf-8",
    )
    index_path = tmp_path / "cache" / "index.json"

    result = _run_cli(
        "init",
        "--manifest",
        str(manifest_path),
        "--index-path",
        str(index_path),
        "--namespaces",
        "ns2",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["total"] == 1
    assert payload["compiled"] == ["fb"]

    env = {
        "JARVIS_OPERAS_CURVE_INDEX": str(index_path),
        "JARVIS_OPERAS_PERSIST_FILE": str(tmp_path / "persist_store.json"),
    }
    list_ns1 = _run_cli("list", "--namespace", "ns1", "--json", env=env)
    list_ns2 = _run_cli("list", "--namespace", "ns2", "--json", env=env)
    assert list_ns1.returncode == 0, list_ns1.stderr
    assert list_ns2.returncode == 0, list_ns2.stderr
    assert json.loads(list_ns1.stdout) == []
    assert any(item["name"] == "ns2.fb" for item in json.loads(list_ns2.stdout))


def test_cli_init_then_list_shows_interp_namespace(tmp_path) -> None:
    curve_dir = tmp_path / "curves"
    curve_dir.mkdir()
    (curve_dir / "lzsi.json").write_text(
        json.dumps({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "curves": [
                    {
                        "curve_id": "LZSI2024",
                        "source": "curves/lzsi.json",
                        "kind": "cubic",
                        "logX": True,
                        "logY": True,
                        "hot": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    index_path = tmp_path / "cache" / "index.json"

    init_result = _run_cli(
        "init",
        "--manifest",
        str(manifest_path),
        "--index-path",
        str(index_path),
        "--json",
    )
    assert init_result.returncode == 0, init_result.stderr

    env = {
        "JARVIS_OPERAS_CURVE_INDEX": str(index_path),
        "JARVIS_OPERAS_PERSIST_FILE": str(tmp_path / "persist_store.json"),
    }
    list_result = _run_cli("list", "--namespace", "interp", "--json", env=env)
    assert list_result.returncode == 0, list_result.stderr
    payload = json.loads(list_result.stdout)
    assert any(item["name"] == "interp.LZSI2024" for item in payload)


def test_cli_init_then_list_uses_group_namespace(tmp_path) -> None:
    curve_dir = tmp_path / "curves"
    curve_dir.mkdir()
    (curve_dir / "lzsi.json").write_text(
        json.dumps({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "curves": [
                    {
                        "curve_id": "LZSI2024",
                        "source": "curves/lzsi.json",
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
    index_path = tmp_path / "cache" / "index.json"

    init_result = _run_cli(
        "init",
        "--manifest",
        str(manifest_path),
        "--index-path",
        str(index_path),
        "--json",
    )
    assert init_result.returncode == 0, init_result.stderr

    env = {
        "JARVIS_OPERAS_CURVE_INDEX": str(index_path),
        "JARVIS_OPERAS_PERSIST_FILE": str(tmp_path / "persist_store.json"),
    }
    list_result = _run_cli("list", "--namespace", "dmdd", "--json", env=env)
    assert list_result.returncode == 0, list_result.stderr
    payload = json.loads(list_result.stdout)
    assert any(item["name"] == "dmdd.LZSI2024" for item in payload)


def test_cli_load_rejects_protected_namespace_override(tmp_path) -> None:
    op_file = tmp_path / "ops.py"
    op_file.write_text(
        textwrap.dedent(
            """
            def foo(x):
                return x

            __JARVIS_OPERAS__ = {
                "foo": foo,
            }
            """
        ),
        encoding="utf-8",
    )
    result = _run_cli("load", str(op_file), "--namespace", "math")
    assert result.returncode == 1
    assert "protected" in result.stderr.lower()


def test_cli_interp_list_and_validate_packaged_manifest() -> None:
    list_result = _run_cli("interp", "list", "--json")
    assert list_result.returncode == 0, list_result.stderr
    list_payload = json.loads(list_result.stdout)
    assert isinstance(list_payload.get("namespaces"), list)

    validate_result = _run_cli("interp", "validate", "--json")
    assert validate_result.returncode == 0, validate_result.stderr
    validate_payload = json.loads(validate_result.stdout)
    assert validate_payload["ok"] is True
    assert validate_payload["namespace_count"] >= 1


def test_cli_removed_mutation_commands_are_unrecognized(tmp_path) -> None:
    script_path = tmp_path / "ops.py"
    script_path.write_text("def foo(x):\n    return x\n", encoding="utf-8")

    removed_invocations = [
        ("update", str(script_path)),
        ("delete-func", "math.add"),
        ("delete-namespace", "math"),
        ("interp", "add", "--manifest", str(tmp_path / "manifest.json"), "ns", "ns.json"),
        ("interp", "remove", "--manifest", str(tmp_path / "manifest.json"), "ns"),
    ]
    for cmd in removed_invocations:
        result = _run_cli(*cmd)
        assert result.returncode != 0
        assert "invalid choice" in result.stderr
