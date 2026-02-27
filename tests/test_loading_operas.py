from __future__ import annotations

import textwrap

from jarvis_operas import OperaFunction, OperasRegistry, load_user_ops


def test_load_user_ops_with_decorator_into_operas_registry(tmp_path) -> None:
    op_file = tmp_path / "decor_ops.py"
    op_file.write_text(
        textwrap.dedent(
            """
            from jarvis_operas import oper

            @oper("double", arity=1, tag="demo")
            def double(x):
                return x * 2
            """
        ),
        encoding="utf-8",
    )

    registry = OperasRegistry()
    loaded = load_user_ops(str(op_file), registry)

    assert loaded == ["decor_ops.double"]
    assert registry.call("decor_ops.double", 6) == 12
    info = registry.info("decor_ops.double")
    assert info["arity"] == 1
    assert info["metadata"]["tag"] == "demo"


def test_load_user_ops_with_export_whitelist_into_operas_registry(tmp_path) -> None:
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

    registry = OperasRegistry()
    loaded = load_user_ops(str(op_file), registry)

    assert loaded == ["whitelist_ops.triple"]
    assert registry.call("whitelist_ops.triple", 5) == 15
    info = registry.info("whitelist_ops.triple")
    assert info["metadata"]["source"] == "__JARVIS_OPERAS__"


def test_load_user_ops_accepts_operafunction_values(tmp_path) -> None:
    op_file = tmp_path / "ops_decl.py"
    op_file.write_text(
        textwrap.dedent(
            """
            from jarvis_operas import OperaFunction

            def scale_three(x):
                return x * 3

            __JARVIS_OPERAS__ = {
                "scale_three": OperaFunction(
                    namespace="placeholder",
                    name="placeholder",
                    arity=1,
                    return_dtype=None,
                    numpy_impl=scale_three,
                    metadata={"kind": "decl"},
                ),
            }
            """
        ),
        encoding="utf-8",
    )

    registry = OperasRegistry()
    loaded = load_user_ops(str(op_file), registry, namespace="ops")

    assert loaded == ["ops.scale_three"]
    declaration = registry.get("ops.scale_three")
    assert isinstance(declaration, OperaFunction)
    assert declaration.namespace == "ops"
    assert declaration.name == "scale_three"
    assert registry.call("ops.scale_three", 4) == 12
    info = registry.info("ops.scale_three")
    assert info["metadata"]["kind"] == "decl"
    assert info["metadata"]["source"] == "__JARVIS_OPERAS__"

