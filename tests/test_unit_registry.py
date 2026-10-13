from __future__ import annotations

import asyncio
import copy
import inspect
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from jarvis_operas import OperaFunction, Operas, OperasRegistry, UnitContractError, load_user_ops


CONTRACT = {
    "version": "1.0",
    "inputs": {"mass": {"unit": "GeV", "shape": "scalar", "nullable": False}},
    "outputs": {"/value": {"unit": "pb", "shape": "scalar", "nullable": True}},
}


def make_registry(*, asynchronous=False):
    registry = OperasRegistry(numpy_concurrency=2)
    calls = []
    result = {
        "value": 0.038332,
        "unit": "pb",
        "record_id": "canonical-pb-record",
        "status": "ok",
        "provenance": {"dependencies": ["mass-record"]},
    }

    def limit(mass):
        calls.append(mass)
        return result

    async def alimit(mass):
        await asyncio.sleep(0)
        return limit(mass)

    registry.register(OperaFunction(
        namespace="fixture", name="limit", arity=1, return_dtype=None,
        numpy_impl=alimit if asynchronous else limit,
        metadata={"params": {"mass": "Resonance mass"}, "unit_contract": CONTRACT},
    ))
    return registry, calls, result


def test_registry_canonical_normalization_and_original_record_protection():
    registry, calls, canonical = make_registry()
    original = copy.deepcopy(canonical)
    outcomes = [registry.call_units(
        "fixture.limit", mass=value, input_units={"mass": unit},
        output_units={"limit": "fb"},
        output_bindings={"limit": "/value", "status": "/status"},
    ) for value, unit in [(200, "GeV"), (0.2, "TeV"), (200000, "MeV")]]
    assert calls == [200, 200, 200]
    assert all(item.values["limit"] == pytest.approx(38.332) for item in outcomes)
    assert outcomes[0].values["status"] == "ok"
    assert outcomes[0].units["limit"] == "fb"
    assert canonical == original
    assert outcomes[0].canonical_result is canonical
    assert outcomes[1].normalized_inputs["mass"] == 200
    outcomes[0].audit.clear()
    assert outcomes[1].audit


@pytest.mark.parametrize("options,code", [
    ({"input_units": {"mass": "pb"}}, "dimension_mismatch"),
    ({"input_units": {"mass": "gev"}}, "unknown_unit"),
    ({"strict": True}, "missing_unit"),
    ({"source_units": {"mass": "TeV"}, "input_units": {"mass": "GeV"}}, "source_conflict"),
    ({"backend": "polars"}, "unsupported_backend"),
])
def test_registry_failures_do_not_execute(options, code):
    registry, calls, _ = make_registry()
    with pytest.raises(UnitContractError) as failure:
        registry.call_units("fixture.limit", mass=0.2, **options)
    assert failure.value.code == code
    assert calls == []


def test_registry_defaults_raw_and_audited_calls_compatible():
    registry, calls, result = make_registry()
    assert registry.call("fixture.limit", mass=200) is result
    audited = registry.call_units("fixture.limit", mass=200)
    assert audited.values["/value"] == 0.038332
    assert any(row["origin"] == "assumed_canonical" for row in audited.audit)
    assert calls == [200, 200]
    assert registry.info("fixture.limit")["unit_contract"] == CONTRACT
    snapshot = registry.info("fixture.limit")
    snapshot["unit_contract"]["inputs"]["mass"]["unit"] = "TeV"
    assert registry.get("fixture.limit").metadata["unit_contract"]["inputs"]["mass"]["unit"] == "GeV"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sync_async_batch_and_prepared_public_paths_agree(asynchronous):
    registry, calls, _ = make_registry(asynchronous=asynchronous)
    plan = registry.prepare("fixture.limit", input_units={"mass": "TeV"}, output_units={"/value": "fb"})
    assert str(inspect.signature(plan)) == str(inspect.signature(registry.get("fixture.limit").numpy_impl))

    async def run():
        public = await Operas(registry).acall_units(
            "fixture.limit", mass=0.2, input_units={"mass": "TeV"}, output_units={"/value": "fb"}
        )
        prepared = await plan.arun(mass=0.2)
        batches = await registry.acall_many_units(
            "fixture.limit", [{"mass": 0.2}, {"mass": 0.125}],
            input_units={"mass": "TeV"}, output_units={"/value": "fb"}, concurrency=2,
        )
        raw = await registry.acall_many("fixture.limit", [{"mass": 200}, {"mass": 125}])
        return public, prepared, batches, raw

    public, prepared, batches, raw = asyncio.run(run())
    assert public.values == prepared.values == batches[0].values
    assert public.values["/value"] == pytest.approx(38.332)
    assert batches[1].normalized_inputs["mass"] == 125
    assert all(item["value"] == 0.038332 for item in raw)
    assert sorted(calls) == [125, 125, 200, 200, 200, 200]
    assert batches[0].audit is not batches[1].audit
    if not asynchronous:
        assert plan.run(mass=0.2).values == public.values


def test_prepared_thread_and_fork_calls_have_no_shared_audit():
    registry, _, _ = make_registry()
    plan = registry.prepare("fixture.limit", input_units={"mass": "TeV"})
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda value: plan.run(mass=value), [0.1, 0.2, 0.3]))
    assert [item.normalized_inputs["mass"] for item in results] == [100, 200, 300]
    assert len({id(item.audit) for item in results}) == 3
    if "fork" in multiprocessing.get_all_start_methods():
        ctx = multiprocessing.get_context("fork")
        reader, writer = ctx.Pipe(duplex=False)

        def child():
            value = plan.run(mass=0.125)
            writer.send((value.normalized_inputs["mass"], value.audit))
            writer.close()

        process = ctx.Process(target=child)
        process.start()
        writer.close()
        assert reader.poll(10)
        mass, audit = reader.recv()
        process.join(10)
        assert process.exitcode == 0
        assert mass == 125 and audit
        assert plan.run(mass=0.2).normalized_inputs["mass"] == 200


def test_array_input_copy_and_default_arity_keeps_backend_result_type():
    registry = OperasRegistry()

    def array_limit(mass, logger=None, **params):
        mass[0] = -10  # Implementation mutation must not reach caller input.
        return np.asarray(mass)

    registry.register(OperaFunction(
        namespace="fixture", name="array", arity=1, return_dtype=None, numpy_impl=array_limit,
        metadata={"unit_contract": {
            "version": "1.0", "inputs": {"mass": {"unit": "GeV", "shape": [2], "nullable": False}},
            "outputs": {"": {"unit": "GeV", "shape": [2], "nullable": False}},
        }},
    ))
    original = np.array([0.2, 0.125])
    result = registry.call_units("fixture.array", original, input_units={"mass": "TeV"})
    np.testing.assert_array_equal(original, [0.2, 0.125])
    assert isinstance(result.canonical_result, np.ndarray)


def test_user_ops_metadata_uses_declaration_validation(tmp_path):
    source = tmp_path / "unit_ops.py"
    source.write_text(
        "from jarvis_operas import oper\n"
        "@oper('mass', unit_contract={'version':'1.0','inputs':{'x':{'unit':'GeV','shape':'scalar','nullable':False}},"
        "'outputs':{'':{'unit':'GeV','shape':'scalar','nullable':False}}})\n"
        "def mass(x): return x\n", encoding="utf-8",
    )
    registry = OperasRegistry()
    load_user_ops(str(source), registry)
    assert registry.call_units("unit_ops.mass", x=0.2, input_units={"x": "TeV"}).canonical_result == 200
    source.write_text(source.read_text().replace("'unit':'GeV'", "'unit':'bogus'"))
    from jarvis_operas import OperatorLoadError
    with pytest.raises(OperatorLoadError) as error:
        load_user_ops(str(source), registry)
    assert isinstance(error.value.__cause__, UnitContractError)
