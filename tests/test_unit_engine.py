from __future__ import annotations

import asyncio
import copy
import inspect

import numpy as np
import pandas as pd
import pytest

from jarvis_operas.core.spec import OperaFunction
from jarvis_operas.errors import OperatorCallError
from jarvis_operas.units import (
    UnitContractError, conversion_factor, convert_value, prepare_unit_call,
    unit_catalog, validate_unit_contract,
)


def _contract(*, shape="scalar", nullable=False):
    return {
        "version": "1.0",
        "inputs": {"mass": {"unit": "GeV", "shape": shape, "nullable": nullable}},
        "outputs": {"value": {"unit": "pb", "shape": "scalar", "nullable": True}},
    }


def _decl(fn, *, contract=None, arity=None, params=None):
    metadata = {} if contract is None else {"unit_contract": contract}
    if params is not None:
        metadata["params"] = params
    return OperaFunction(
        namespace="fixture", name="limit", arity=arity,
        return_dtype=None, numpy_impl=fn, metadata=metadata,
    )


def test_same_canonical_mass_projection_and_record_protection():
    seen = []
    record = {"value": 0.038332, "unit": "pb", "record_id": "original",
              "provenance": {"parents": ["source"]}, "status": "ok"}
    original = copy.deepcopy(record)

    def limit(mass):
        seen.append(mass)
        return record

    declaration = _decl(limit, contract=_contract())
    plan = prepare_unit_call(
        declaration, input_units={"mass": "TeV"},
        output_bindings={"limit_fb": "/value", "status": "/status"},
        output_units={"limit_fb": "fb"},
    )
    result = plan.run(mass=0.2)
    assert result.normalized_inputs == {"mass": 200.0}
    assert result.values == {"limit_fb": pytest.approx(38.332), "status": "ok"}
    assert result.units == {"limit_fb": "fb", "status": None}
    assert result.canonical_result is record
    assert record == original
    result.values["limit_fb"] = 123
    assert record == original
    prepare_unit_call(declaration, input_units={"mass": "GeV"}).run(200)
    prepare_unit_call(declaration, input_units={"mass": "MeV"}).run(125000)
    assert seen == [200.0, 200, 125.0]
    assert result.audit[0]["factor"] == 1000
    assert result.audit[0]["origin"] == "declared"
    assert result.audit[1]["contract_version"] == "1.0"


@pytest.mark.parametrize("setup,value,code", [
    ({"input_units": {"mass": "pb"}}, 1, "dimension_mismatch"),
    ({"input_units": {"mass": "gev"}}, 1, "unknown_unit"),
    ({"input_units": {"mass": "GeV*fm"}}, 1, "unsupported_unit"),
    ({"strict": True}, 1, "missing_unit"),
    ({"input_units": {"mass": "GeV"}, "source_units": {"mass": "TeV"}}, 1, "source_conflict"),
    ({"input_units": {"mass": "GeV"}}, True, "type_error"),
    ({"input_units": {"mass": "GeV"}}, None, "type_error"),
    ({"input_units": {"mass": "GeV"}}, float("nan"), "non_finite"),
    ({"input_units": {"mass": "GeV"}}, float("inf"), "non_finite"),
    ({"input_units": {"mass": "GeV"}}, "200", "type_error"),
    ({"input_units": {"mass": "GeV"}}, 1 + 2j, "type_error"),
    ({"input_units": {"mass": "GeV"}}, [1, 2], "shape_mismatch"),
])
def test_input_errors_prevent_execution(setup, value, code):
    calls = []

    def limit(mass):
        calls.append(mass)
        return {"value": 1}

    with pytest.raises(UnitContractError) as error:
        prepare_unit_call(_decl(limit, contract=_contract()), **setup).run(value)
    assert error.value.code == code
    assert code in str(error.value)
    assert calls == []


def test_missing_contract_and_uncontracted_field_are_never_ignored():
    calls = []

    def limit(mass):
        calls.append(mass)
        return mass

    declaration = _decl(limit)
    assert prepare_unit_call(declaration).run(7).canonical_result == 7
    for options in ({"input_units": {"mass": "GeV"}}, {"output_units": {"": "pb"}}):
        with pytest.raises(UnitContractError, match="contract_missing"):
            prepare_unit_call(declaration, **options)
    assert calls == [7]


def test_default_assumption_trusted_records_strict_and_defaults():
    def limit(mass=125, logger=None):
        return {"value": mass / 1000}

    declaration = _decl(limit, contract=_contract(), arity=1)
    compatibility = prepare_unit_call(declaration).run(-1)
    assert compatibility.audit[0]["origin"] == "assumed_canonical"
    strict = prepare_unit_call(declaration, strict=True)
    assert strict.run().normalized_inputs["mass"] == 125
    assert strict.run().audit[0]["origin"] == "default_canonical"
    assert strict.run_inputs((0,), {}, source_units={"mass": {"unit": "TeV", "record_id": "r"}}).normalized_inputs["mass"] == 0
    with pytest.raises(UnitContractError, match="source_conflict"):
        prepare_unit_call(declaration, input_units={"mass": "GeV"}).run_inputs((1,), {}, source_units={"mass": "TeV"})


def test_nullable_preserves_none_and_normalizer_preserves_call_layout():
    def limit(mass, logger=None):
        return {"value": None}

    plan = prepare_unit_call(_decl(limit, contract=_contract(nullable=True), arity=1))
    args, kwargs, audit = plan.normalize(None)
    assert args == (None,)
    assert kwargs == {}
    assert audit[0]["origin"] == "assumed_canonical"
    keyword_args, keyword_kwargs, _ = plan.normalize(mass=1, logger=None)
    assert keyword_args == ()
    assert keyword_kwargs == {"mass": 1, "logger": None}
    assert plan.run(mass=1, logger=None).canonical_result == {"value": None}
    assert plan.run(None).values["/value"] is None
    assert inspect.signature(plan) == inspect.signature(limit)


@pytest.mark.parametrize("value", [np.array([0., -2.]), [0., -2.], (0., -2.), pd.Series([0., -2.], index=["a", "b"])])
def test_containers_scale_without_mutation_or_type_changes(value):
    before = copy.deepcopy(value)
    projected = convert_value(value, "TeV", "GeV", shape=[2])
    assert type(projected) is type(value)
    np.testing.assert_allclose(projected, [0, -2000])
    np.testing.assert_array_equal(value, before)
    same = convert_value(value, "GeV", "GeV", shape=[2])
    assert type(same) is type(value)
    if not isinstance(value, tuple):
        assert same is not value
    if isinstance(value, pd.Series):
        assert projected.index.equals(value.index)
    np.testing.assert_array_equal(same, value)


def test_array_shape_bool_and_nonfinite_checks_and_no_broadcast():
    for value, code in (([True, 2], "type_error"), (np.array([True]), "type_error"),
                        ([1., float("nan")], "non_finite"), ([1, 2], "shape_mismatch")):
        with pytest.raises(UnitContractError) as error:
            convert_value(value, "GeV", "MeV", shape=[1])
        assert error.value.code == code
    with pytest.raises(UnitContractError, match="shape_mismatch"):
        convert_value(1, "GeV", "MeV", shape="array")
    assert convert_value(np.array(2.), "GeV", "MeV", shape=[]) == 2000.


def test_assumed_inputs_are_copied_even_if_implementation_mutates_them():
    value = np.array([1., 2.])

    def limit(mass):
        mass[0] = 99
        return {"value": 1.}

    result = prepare_unit_call(_decl(limit, contract=_contract(shape="array"))).run(value)
    np.testing.assert_array_equal(value, [1., 2.])
    np.testing.assert_array_equal(result.normalized_inputs["mass"], [1., 2.])


def test_alias_units_and_roundtrip_and_no_natural_unit_crossing():
    assert conversion_factor("pb", "fb") == 1000
    assert conversion_factor("cm^2", "pb") == 1e36
    assert conversion_factor("MeV^2", "GeV^2") == 1e-6
    assert conversion_factor("TeV^-2", "GeV^-2") == 1e-6
    assert conversion_factor("deg", "rad") == pytest.approx(np.pi / 180)
    assert conversion_factor("MeV*fm", "MeV·fm") == 1
    assert conversion_factor("µK^2", "uK^2") == 1
    for source, target in (("GeV^-2", "pb"), ("GeV", "mm"), ("rad", "1")):
        with pytest.raises(UnitContractError, match="dimension_mismatch"):
            conversion_factor(source, target)
    value = 0.038332
    assert convert_value(convert_value(value, "pb", "fb"), "fb", "pb") == pytest.approx(value, rel=1e-13)
    catalog = unit_catalog()
    catalog["GeV"]["scale"] = 999
    assert conversion_factor("GeV", "MeV") == 1000


def test_pointer_root_and_escaped_dictionary_keys_are_unambiguous():
    def scalar():
        return 2.

    plan = prepare_unit_call(_decl(scalar, contract={"version": "1.0", "outputs": {"": {"unit": "GeV"}}}), output_units={"": "MeV"})
    assert plan().values == {"": 2000.}

    def dictionary():
        return {"": 3., "a/b": {"~": 4.}}

    contract = {"version": "1.0", "outputs": {"/": {"unit": "GeV"}, "/a~1b/~0": {"unit": "GeV"}}}
    plan = prepare_unit_call(_decl(dictionary, contract=contract), output_bindings={"empty_key": "/", "nested": "/a~1b/~0"})
    assert plan.run().values == {"empty_key": 3., "nested": 4.}
    with pytest.raises(UnitContractError, match="output_path_missing"):
        prepare_unit_call(_decl(dictionary), output_bindings={"missing": "/missing"}).run()


@pytest.mark.parametrize("mutation,code", [
    (lambda c: c.update(version="2.0"), "invalid_contract"),
    (lambda c: c["inputs"].update(absent={"unit": "GeV"}), "invalid_contract"),
    (lambda c: c["inputs"]["mass"].update(shape="vector"), "invalid_contract"),
    (lambda c: c["inputs"]["mass"].update(nullable="yes"), "invalid_contract"),
    (lambda c: c["outputs"].update({"/a~2b": {"unit": "pb"}}), "invalid_contract"),
    (lambda c: c["outputs"].update({1: {"unit": "pb"}}), "invalid_contract"),
    (lambda c: c["inputs"]["mass"].update(unit="GEV"), "unknown_unit"),
    (lambda c: c.update(extra=True), "invalid_contract"),
])
def test_metadata_validation(mutation, code):
    contract = _contract()
    mutation(contract)
    with pytest.raises(UnitContractError) as error:
        validate_unit_contract(contract, lambda mass: mass)
    assert error.value.code == code


def test_contract_and_plan_metadata_are_independent():
    contract = _contract()
    validated = validate_unit_contract(contract, lambda mass: mass)
    validated["inputs"]["mass"]["unit"] = "MeV"
    assert contract["inputs"]["mass"]["unit"] == "GeV"
    declaration = _decl(lambda mass: {"value": 1}, contract=contract)
    plan = prepare_unit_call(declaration)
    contract["inputs"]["mass"]["unit"] = "TeV"
    assert plan.contract["inputs"]["mass"]["unit"] == "GeV"
    with pytest.raises(TypeError):
        plan.contract["inputs"]["mass"]["unit"] = "TeV"


def test_logical_documented_keyword_contracts_and_flat_snapshots():
    def limit(payload=None, **params):
        return {"value": params.get("mass", 125) / 1000}

    plan = prepare_unit_call(_decl(limit, contract=_contract(), params={"mass": "mass"}), input_units={"mass": "TeV"})
    result = plan.run(mass=0.2)
    assert result.normalized_inputs == {"payload": None, "mass": 200.0}
    assert prepare_unit_call(plan.declaration).run().audit == [{"field": "/value", "kind": "output", "source_unit": "pb", "target_unit": "pb", "factor": 1., "origin": "contract", "contract_version": "1.0"}]
    with pytest.raises(UnitContractError, match="contract_missing"):
        plan.run(payload={"mass": 0.2})
    with pytest.raises(UnitContractError, match="invalid_contract"):
        validate_unit_contract(_contract(), limit, documented_parameters={})


def test_argument_container_api_preserves_source_units_physical_parameter():
    def physical(source_units):
        return source_units

    contract = {"version": "1.0", "inputs": {"source_units": {"unit": "GeV"}}}
    plan = prepare_unit_call(_decl(physical, contract=contract))
    args, kwargs, audit = plan.normalize_inputs((), {"source_units": 125})
    assert args == ()
    assert kwargs == {"source_units": 125}
    assert audit[0]["field"] == "source_units"
    assert plan.run(source_units=125).canonical_result == 125
    assert asyncio.run(plan.arun(source_units=125)).canonical_result == 125


def test_nested_logical_values_cannot_bypass_assertions_or_strict_mode():
    calls = []

    def physical(payload=None, **params):
        calls.append(True)
        return {"value": params.get("mass", (payload or {}).get("mass", 1))}

    declaration = _decl(physical, contract=_contract(), params={"mass": "mass"})
    for options in ({"input_units": {"mass": "TeV"}}, {"strict": True}):
        plan = prepare_unit_call(declaration, **options)
        for payload in ({"mass": 0.2}, {"alias": 0.2}, {}):
            with pytest.raises(UnitContractError, match="contract_missing"):
                plan.run(payload=payload)
            with pytest.raises(UnitContractError, match="contract_missing"):
                plan.run(payload=payload, mass=0.2)
    assert calls == []
    assert prepare_unit_call(declaration).run(payload={"mass": 0.2}).canonical_result == {"value": 0.2}
    assert prepare_unit_call(declaration, strict=True).run().canonical_result == {"value": 1}
    asserted = prepare_unit_call(declaration, input_units={"mass": "GeV"})
    with pytest.raises(UnitContractError, match="source_conflict"):
        asserted.run_inputs((), {}, source_units={"mass": "TeV"})


def test_return_record_conflicts_and_nonfinite_results_are_explicit_errors():
    for record, code in (({"value": 1, "unit": "fb"}, "source_conflict"),
                         ({"value": float("nan"), "unit": "pb"}, "non_finite")):
        plan = prepare_unit_call(_decl(lambda mass: record, contract=_contract()))
        with pytest.raises(UnitContractError) as error:
            plan.run(1)
        assert error.value.code == code
    plan = prepare_unit_call(_decl(lambda: {"value": 1, "unit": "pb"}, contract={"version": "1.0", "outputs": {"": {"unit": "pb"}}}))
    with pytest.raises(UnitContractError, match="type_error"):
        plan.run()


def test_sync_async_and_concurrent_audits_have_no_shared_state():
    def limit(mass):
        return {"value": mass / 1000}

    async def async_limit(mass):
        await asyncio.sleep(0)
        return limit(mass)

    sync = prepare_unit_call(_decl(limit, contract=_contract()), output_units={"value": "fb"})
    asynchronous = prepare_unit_call(_decl(async_limit, contract=_contract()), output_units={"value": "fb"})

    async def exercise():
        results = await asyncio.gather(
            sync.arun_inputs((0.2,), {}, source_units={"mass": "TeV"}),
            sync.arun_inputs((200,), {}, source_units={"mass": "GeV"}),
            asynchronous.arun_inputs((200000,), {}, source_units={"mass": "MeV"}),
        )
        assert all(result.values["/value"] == 200 for result in results)
        assert [result.audit[0]["source_unit"] for result in results] == ["TeV", "GeV", "MeV"]
        results[0].audit[0]["source_unit"] = "changed"
        assert results[1].audit[0]["source_unit"] == "GeV"

    asyncio.run(exercise())
    with pytest.raises(OperatorCallError, match="awaitable"):
        asynchronous.run(200)


def test_physical_exceptions_keep_standard_wrapper():
    def boom(mass):
        raise RuntimeError("physical failure")

    plan = prepare_unit_call(_decl(boom, contract=_contract()))
    with pytest.raises(OperatorCallError) as error:
        plan.run(1)
    assert isinstance(error.value.__cause__, RuntimeError)
    with pytest.raises(OperatorCallError):
        asyncio.run(plan.arun(1))
