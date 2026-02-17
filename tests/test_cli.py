from __future__ import annotations

import json
import subprocess
import sys
import textwrap


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "jarvis_operas.cli", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_list_contains_builtin_add() -> None:
    result = _run_cli("list", "--namespace", "core")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "core:add" in payload


def test_cli_call_add_with_kwargs() -> None:
    result = _run_cli("call", "core:add", "--kwargs", '{"a": 2, "b": 3}')

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
        "user:triple",
        "--user-ops",
        str(op_file),
        "--arg",
        "x=4",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "12"
