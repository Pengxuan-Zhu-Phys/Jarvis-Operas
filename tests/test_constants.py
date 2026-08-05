#!/usr/bin/env python3
"""Namespace constants (D23): arity-0 OperaFunction + flags=constant + sp.Float fold."""

from __future__ import annotations

import inspect
import math
import pickle

import pytest
import sympy as sp
from sympy.utilities.lambdify import lambdify

from jarvis_operas import OperaFunction, OperasRegistry, OperatorConflict
from jarvis_operas.api import get_global_operas_registry
from jarvis_operas.integration import (
    ConstantNamespace,
    ConstantValue,
    build_constant_dicts,
    build_register_dicts,
    build_sympy_dicts,
)
from jarvis_operas.namespace_policy import is_protected_namespace
from jarvis_operas.namespaces._constants import constant_decl


def _register_const(registry: OperasRegistry, name: str, value: float) -> None:
    registry.register(constant_decl("pdg_test", name, value, unit="1", source="test"))


def test_constant_decl_flags_metadata_and_call() -> None:
    decl = constant_decl(
        "demo",
        "answer",
        42.0,
        unit="1",
        summary="test constant",
        source="unit-test",
        since="1.4.0",
        tags=["test"],
    )
    assert decl.arity == 0
    assert "constant" in decl.flags
    assert decl.metadata["value"] == 42.0
    assert decl.metadata["category"] == "constant"
    assert callable(decl.numpy_impl)
    assert decl.numpy_impl() == 42.0
    assert decl.numpy_impl() == decl.metadata["value"]


def test_build_constant_dicts_only_includes_flagged() -> None:
    registry = OperasRegistry()
    _register_const(registry, "mZ", 91.1876)
    registry.register(
        OperaFunction(
            namespace="pdg_test",
            name="scale",
            arity=1,
            return_dtype=None,
            numpy_impl=lambda x: x * 2.0,
        )
    )

    table = build_constant_dicts(registry)
    assert table == {"pdg_test.mZ": 91.1876}
    assert "pdg_test.scale" not in table


def test_build_sympy_dicts_folds_constants_and_skips_arity0_impl() -> None:
    registry = OperasRegistry()
    _register_const(registry, "mZ", 91.1876)
    _register_const(registry, "mW", 80.377)
    registry.register(
        OperaFunction(
            namespace="pdg_test",
            name="scale",
            arity=1,
            return_dtype=None,
            numpy_impl=lambda x: x * 2.0,
        )
    )

    mapping = build_register_dicts(registry)
    constants = build_constant_dicts(registry)
    # arity-0 impl is present in register dicts (required by core/spec)
    assert callable(mapping["pdg_test.mZ"])

    parse_locals, numeric_funcs = build_sympy_dicts(
        mapping,
        constants=constants,
        include_all=True,
    )

    assert isinstance(parse_locals["pdg_test"], ConstantNamespace)
    assert isinstance(parse_locals["pdg_test"].mZ, ConstantValue)
    assert float(parse_locals["pdg_test"].mZ) == 91.1876
    # constant impl must not land in lambdify module map
    assert not any("mZ" in name for name in numeric_funcs)
    assert any("scale" in name for name in numeric_funcs)

    expr = sp.sympify("sqrt(pdg_test.mZ**2 + pdg_test.mW**2) * x", locals=parse_locals)
    free = tuple(sorted(str(s) for s in expr.free_symbols))
    assert free == ("x",)
    body = str(expr)
    assert "121.555092541448" in body or abs(float(expr.subs({sp.Symbol("x"): 1.0})) - 121.555092541448) < 1e-9

    num = lambdify(["x"], expr, modules=[numeric_funcs, "numpy"])
    assert abs(float(num(1.0)) - 121.555092541448) < 1e-9


def test_build_sympy_dicts_v1_path_does_not_fold_without_constants_kwarg() -> None:
    """V1/PLOT: mapping given, constants omitted => arity-0 stays as Function."""

    registry = OperasRegistry()
    _register_const(registry, "mZ", 91.1876)
    mapping = build_register_dicts(registry)

    parse_locals, numeric_funcs = build_sympy_dicts(mapping, include_all=True)

    attr = parse_locals["pdg_test"].mZ
    assert not isinstance(attr, ConstantValue)
    assert isinstance(attr, sp.Function) or (
        isinstance(attr, type) and issubclass(attr, sp.Function)
    )
    # callable form still works for V1-style zero-arg functions
    assert any("mZ" in name for name in numeric_funcs)


def test_build_sympy_dicts_signature_contract() -> None:
    sig = inspect.signature(build_sympy_dicts)
    params = list(sig.parameters.values())
    assert params[0].name == "mapping"
    assert params[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert "constants" in sig.parameters
    assert sig.parameters["constants"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["constants"].default is None

    # positional return is still a 2-tuple
    result = build_sympy_dicts({}, include_all=True)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_float_roundtrip_precision() -> None:
    for value in (197.3269804, 6.62607015e-34, 91.1876):
        cv = ConstantValue(value, full_name="demo.x")
        assert float(cv) == value
        assert pickle.loads(pickle.dumps(cv)) == cv or float(
            pickle.loads(pickle.dumps(cv))
        ) == value


def test_constant_value_call_error_and_namespace_suggestion() -> None:
    registry = OperasRegistry()
    _register_const(registry, "mZ", 91.1876)
    _register_const(registry, "hbarc", 197.3269804)
    parse_locals, _ = build_sympy_dicts(
        build_register_dicts(registry),
        constants=build_constant_dicts(registry),
        include_all=True,
    )

    with pytest.raises(TypeError, match="without parentheses"):
        parse_locals["pdg_test"].mZ()

    with pytest.raises(AttributeError, match="Did you mean") as exc:
        _ = parse_locals["pdg_test"].mZZ
    message = str(exc.value)
    assert "pdg_test.mZZ" in message
    assert "mZ" in message


def test_constant_namespace_casefold_typo_suggests_mZ() -> None:
    """D23.11: pdg.mz (most common case typo) should suggest mZ."""
    registry = OperasRegistry()
    _register_const(registry, "mZ", 91.1876)
    _register_const(registry, "mt", 172.57)
    _register_const(registry, "mh", 125.25)
    parse_locals, _ = build_sympy_dicts(
        build_register_dicts(registry),
        constants=build_constant_dicts(registry),
        include_all=True,
    )
    with pytest.raises(AttributeError, match="Did you mean") as exc:
        _ = parse_locals["pdg_test"].mz
    assert "mZ" in str(exc.value)


def test_constant_namespace_pickle_and_deepcopy() -> None:
    """D23.10: ConstantNamespace must round-trip like SimpleNamespace."""
    import copy
    import pickle

    registry = OperasRegistry()
    _register_const(registry, "mZ", 91.1876)
    _register_const(registry, "hbarc", 197.3269804)
    parse_locals, _ = build_sympy_dicts(
        build_register_dicts(registry),
        constants=build_constant_dicts(registry),
        include_all=True,
    )
    ns = parse_locals["pdg_test"]
    assert isinstance(ns, ConstantNamespace)

    restored = pickle.loads(pickle.dumps(ns))
    assert isinstance(restored, ConstantNamespace)
    assert float(restored.mZ) == pytest.approx(91.1876)
    assert float(restored.hbarc) == pytest.approx(197.3269804)

    cloned = copy.deepcopy(ns)
    assert isinstance(cloned, ConstantNamespace)
    assert float(cloned.mZ) == pytest.approx(91.1876)


def test_coexist_with_bare_symbol_same_name() -> None:
    registry = OperasRegistry()
    _register_const(registry, "mZ", 91.1876)
    parse_locals, _ = build_sympy_dicts(
        build_register_dicts(registry),
        constants=build_constant_dicts(registry),
        include_all=True,
    )
    # Bare symbol mZ and qualified pdg_test.mZ are distinct lexical units.
    expr = sp.sympify("mZ + pdg_test.mZ", locals=parse_locals)
    free = tuple(sorted(str(s) for s in expr.free_symbols))
    assert free == ("mZ",)
    value = float(expr.subs({sp.Symbol("mZ"): 1.0}))
    assert abs(value - 92.1876) < 1e-12


def test_mixed_constant_and_function_in_same_namespace() -> None:
    registry = OperasRegistry()
    _register_const(registry, "mZ", 91.1876)
    _register_const(registry, "hbarc", 197.3269804)
    registry.register(
        OperaFunction(
            namespace="pdg_test",
            name="scale",
            arity=1,
            return_dtype=None,
            numpy_impl=lambda x: x * 2.0,
        )
    )
    parse_locals, numeric_funcs = build_sympy_dicts(
        build_register_dicts(registry),
        constants=build_constant_dicts(registry),
        include_all=True,
    )
    expr = sp.sympify(
        "pdg_test.scale(x) + pdg_test.mZ / pdg_test.hbarc",
        locals=parse_locals,
    )
    free = tuple(sorted(str(s) for s in expr.free_symbols))
    assert free == ("x",)
    num = lambdify(["x"], expr, modules=[numeric_funcs, "numpy"])
    expected = 2.0 * 1.0 + 91.1876 / 197.3269804
    assert abs(float(num(1.0)) - expected) < 1e-12


def test_duplicate_constant_registration_raises() -> None:
    registry = OperasRegistry()
    _register_const(registry, "mZ", 91.1876)
    with pytest.raises(OperatorConflict):
        _register_const(registry, "mZ", 90.0)


# --- bundled pdg namespace -------------------------------------------------


EXPECTED_PDG = {
    "pdg.mZ": 91.1876,
    "pdg.mW": 80.377,
    "pdg.mt": 172.57,
    "pdg.mh": 125.25,
    "pdg.me": 5.1099895000e-4,
    "pdg.mmu": 0.1056583755,
    "pdg.mtau": 1.77686,
    "pdg.alphaEM": 7.2973525693e-3,
    "pdg.alphaSMZ": 0.1180,
    "pdg.GF": 1.1663787e-5,
    "pdg.hbarc": 197.3269804,
    "pdg.c": 299792458.0,
}


def test_pdg_namespace_registered_and_protected() -> None:
    registry = get_global_operas_registry()
    names = set(registry.list(namespace="pdg"))
    assert set(EXPECTED_PDG) == names
    assert is_protected_namespace("pdg")

    for full_name, value in EXPECTED_PDG.items():
        decl = registry.get(full_name)
        assert "constant" in decl.flags
        assert decl.arity == 0
        assert decl.metadata.get("value") == value
        assert decl.metadata.get("unit")
        assert decl.metadata.get("source")
        assert decl.metadata.get("summary")
        assert registry.call(full_name) == value
        assert decl.numpy_impl() == decl.metadata["value"]


def test_pdg_module_level_func_locals_folds_constants() -> None:
    from jarvis_operas.integration import func_locals

    assert "pdg" in func_locals
    assert isinstance(func_locals["pdg"].mZ, ConstantValue)
    assert float(func_locals["pdg"].mZ) == 91.1876

    expr = sp.sympify("(mz - pdg.mZ)/pdg.mZ", locals={**func_locals, "mz": sp.Symbol("mz")})
    free = tuple(sorted(str(s) for s in expr.free_symbols))
    assert free == ("mz",)
    value = float(expr.subs({sp.Symbol("mz"): 92.0}))
    assert abs(value - 0.00890910606266515) < 1e-14


def test_pdg_algebraic_fold_lambdify_body() -> None:
    from jarvis_operas.integration import func_locals, numeric_funcs

    expr = sp.sympify("sqrt(pdg.mZ**2 + pdg.mW**2) * x", locals=func_locals)
    free = tuple(sorted(str(s) for s in expr.free_symbols))
    assert free == ("x",)
    # compile-time fold: body is a literal times x
    source = sp.printing.lambdarepr.lambdarepr(expr)
    assert "121.555092541448" in source or math.isclose(
        float(expr.subs({sp.Symbol("x"): 1.0})),
        121.555092541448,
        rel_tol=0,
        abs_tol=1e-9,
    )
    num = lambdify(["x"], expr, modules=[numeric_funcs, "numpy"])
    assert abs(float(num(1.0)) - 121.555092541448) < 1e-9


def test_all_constants_value_matches_impl() -> None:
    registry = get_global_operas_registry()
    constants = build_constant_dicts(registry)
    assert constants
    for full_name, value in constants.items():
        decl = registry.get(full_name)
        assert decl.numpy_impl() == value
        assert float(decl.metadata["value"]) == value
