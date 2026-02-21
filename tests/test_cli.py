from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap


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
    assert any(item["name"] == "math:add" for item in payload)
    assert all(re.match(r"^[a-z][0-9]{3,5}$", item["id"]) for item in payload)


def test_cli_list_human_output_groups_namespace() -> None:
    result = _run_cli("list", "--namespace", "math")

    assert result.returncode == 0, result.stderr
    assert "math (" in result.stdout
    assert "math:add" in result.stdout
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
    assert "jopera info stat:chi2_cov" in result.stdout


def test_cli_help_does_not_show_discover() -> None:
    result = _run_cli("--help")

    assert result.returncode == 0, result.stderr
    assert "discover" not in result.stdout
    assert "update" in result.stdout


def test_cli_call_add_with_kwargs() -> None:
    result = _run_cli("call", "math:add", "--kwargs", '{"a": 2, "b": 3}')

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
        "cli_ops:triple",
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
    result = _run_cli("info", "helper:egbox")

    assert result.returncode == 1
    assert "Operator 'helper:egbox' not found." in result.stderr
    assert "helper:eggbox" in result.stderr


def test_cli_info_shows_operator_id() -> None:
    result = _run_cli("info", "math:add", "--json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "id" in payload
    assert re.match(r"^[a-z][0-9]{3,5}$", payload["id"])
    assert payload["supports_async"] is True


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

    second = _run_cli("call", "persist_cli_ops:plus_one", "--arg", "x=10", env=env)
    assert second.returncode == 0, second.stderr
    assert second.stdout.strip() == "11"


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

    second = _run_cli("call", "ephemeral_cli_ops:plus_two", "--arg", "x=10", env=env)
    assert second.returncode == 1
    assert "not found" in second.stderr


def test_cli_update_function_from_script_and_delete_func(tmp_path) -> None:
    op_file = tmp_path / "func_manage_ops.py"
    op_file.write_text(
        textwrap.dedent(
            """
            def plus_three(x):
                return x + 3

            __JARVIS_OPERAS__ = {
                "plus_three": plus_three,
            }
            """
        ),
        encoding="utf-8",
    )
    store_path = tmp_path / "persist_store.json"
    env = {"JARVIS_OPERAS_PERSIST_FILE": str(store_path)}

    load_result = _run_cli("load", str(op_file), env=env)
    assert load_result.returncode == 0, load_result.stderr

    # Change script implementation, then update only this function from script.
    op_file.write_text(
        textwrap.dedent(
            """
            def plus_three(x):
                return x + 10

            __JARVIS_OPERAS__ = {
                "plus_three": plus_three,
            }
            """
        ),
        encoding="utf-8",
    )
    update_result = _run_cli(
        "update",
        str(op_file),
        "--function",
        "plus_three",
        env=env,
    )
    assert update_result.returncode == 0, update_result.stderr

    call_new = _run_cli("call", "func_manage_ops:plus_three", "--arg", "x=1", env=env)
    assert call_new.returncode == 0, call_new.stderr
    assert call_new.stdout.strip() == "11"

    info_result = _run_cli("info", "func_manage_ops:plus_three", "--json", env=env)
    assert info_result.returncode == 0, info_result.stderr
    payload = json.loads(info_result.stdout)
    operator_id = payload["id"]

    delete_builtin = _run_cli("delete-func", operator_id, env=env)
    assert delete_builtin.returncode == 0, delete_builtin.stderr

    call_deleted = _run_cli("call", "func_manage_ops:plus_three", "--arg", "x=1", env=env)
    assert call_deleted.returncode == 1
    assert "not found" in call_deleted.stderr


def test_cli_update_namespace_from_script_and_delete_namespace(tmp_path) -> None:
    op_file = tmp_path / "ns_manage_ops.py"
    op_file.write_text(
        textwrap.dedent(
            """
            def square(x):
                return x * x

            __JARVIS_OPERAS__ = {
                "square": square,
            }
            """
        ),
        encoding="utf-8",
    )
    store_path = tmp_path / "persist_store.json"
    env = {"JARVIS_OPERAS_PERSIST_FILE": str(store_path)}

    load_result = _run_cli("load", str(op_file), env=env)
    assert load_result.returncode == 0, load_result.stderr

    # Change script implementation, then update whole namespace from script.
    op_file.write_text(
        textwrap.dedent(
            """
            def square(x):
                return x * x * x

            __JARVIS_OPERAS__ = {
                "square": square,
            }
            """
        ),
        encoding="utf-8",
    )
    update_result = _run_cli(
        "update",
        str(op_file),
        env=env,
    )
    assert update_result.returncode == 0, update_result.stderr

    call_new = _run_cli("call", "ns_manage_ops:square", "--arg", "x=3", env=env)
    assert call_new.returncode == 0, call_new.stderr
    assert call_new.stdout.strip() == "27"

    delete_ns = _run_cli("delete-namespace", "ns_manage_ops", env=env)
    assert delete_ns.returncode == 0, delete_ns.stderr

    call_deleted = _run_cli("call", "ns_manage_ops:square", "--arg", "x=3", env=env)
    assert call_deleted.returncode == 1
    assert "not found" in call_deleted.stderr
