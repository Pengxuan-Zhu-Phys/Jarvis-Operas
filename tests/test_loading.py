from __future__ import annotations

import textwrap

import pytest

from jarvis_operas import OperatorLoadError, load_user_ops
from jarvis_operas.registry import OperatorRegistry


def test_load_user_ops_with_decorator(tmp_path) -> None:
    op_file = tmp_path / "decor_ops.py"
    op_file.write_text(
        textwrap.dedent(
            """
            from jarvis_operas import oper

            @oper("double", namespace="user", tag="demo")
            def double(x):
                return x * 2
            """
        ),
        encoding="utf-8",
    )

    registry = OperatorRegistry()
    loaded = load_user_ops(str(op_file), registry)

    assert loaded == ["user:double"]
    assert registry.call("user:double", x=6) == 12
    assert registry.info("user:double")["metadata"]["tag"] == "demo"


def test_load_user_ops_with_export_whitelist(tmp_path) -> None:
    op_file = tmp_path / "whitelist_ops.py"
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

    registry = OperatorRegistry()
    loaded = load_user_ops(str(op_file), registry)

    assert loaded == ["user:triple"]
    assert registry.call("user:triple", x=5) == 15


def test_load_user_ops_wraps_import_errors(tmp_path) -> None:
    op_file = tmp_path / "broken_ops.py"
    op_file.write_text("def broken(:\n", encoding="utf-8")

    registry = OperatorRegistry()

    with pytest.raises(OperatorLoadError) as exc:
        load_user_ops(str(op_file), registry)

    assert "Failed to load operators" in str(exc.value)
    assert exc.value.__cause__ is not None
