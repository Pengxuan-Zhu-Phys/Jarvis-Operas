"""Physical unit declarations and compatibility of the bundled function library."""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from jarvis_operas import OperasRegistry
from jarvis_operas.catalog import get_catalog_declarations
from jarvis_operas.curves import init_builtin_curve_cache, register_hot_curves_in_registry
from jarvis_operas.namespaces._constants import constant_decl
from jarvis_operas.namespaces.cmb.defs.core import _load_internal_planck_tt_cl
from jarvis_operas.units import UnitContractError


@pytest.fixture
def builtins() -> OperasRegistry:
    registry = OperasRegistry()
    registry.register_many(get_catalog_declarations(include_interpolations=True))
    return registry


def test_pdg_constants_have_one_authoritative_root_unit(builtins) -> None:
    expected = {
        "mZ": "GeV", "mW": "GeV", "mt": "GeV", "mh": "GeV",
        "me": "GeV", "mmu": "GeV", "mtau": "GeV",
        "alphaEM": "1", "alphaSMZ": "1", "GF": "GeV^-2",
        "hbarc": "MeV·fm", "c": "m/s",
    }
    for name, unit in expected.items():
        declaration = builtins.get(f"pdg.{name}")
        contract = declaration.metadata["unit_contract"]
        assert contract["version"] == "1.0"
        assert contract["inputs"] == {}
        assert contract["outputs"][""] == {"unit": unit, "shape": "scalar", "nullable": False}
        assert declaration.metadata["unit"] == contract["outputs"][""]["unit"]
        execution = builtins.call_units(declaration.full_name)
        assert execution.canonical_result == builtins.call(declaration.full_name)
        assert execution.values[""] == declaration.metadata["value"]
        assert execution.units[""] == unit


def test_pdg_root_projection_is_separate_and_handles_inverse_square(builtins) -> None:
    mass = builtins.call_units("pdg.mZ", output_units={"": "MeV"})
    assert mass.canonical_result == 91.1876
    assert mass.values[""] == pytest.approx(91187.6)
    gf = builtins.call_units("pdg.GF", output_units={"": "MeV^-2"})
    assert gf.canonical_result == 1.1663787e-5
    assert gf.values[""] == pytest.approx(1.1663787e-11)
    with pytest.raises(UnitContractError) as exc:
        builtins.prepare("pdg.GF", output_units={"": "pb"})
    assert exc.value.code == "dimension_mismatch"


def test_constant_display_unit_is_derived_and_conflicts_are_rejected() -> None:
    contract = {
        "version": "1.0", "inputs": {},
        "outputs": {"": {"unit": "GeV", "shape": "scalar", "nullable": False}},
    }
    declaration = constant_decl("demo", "mass", 125.0, unit_contract=contract)
    assert declaration.metadata["unit"] == "GeV"
    alias = constant_decl("demo", "hbarc", 197.0, unit="MeV*fm")
    assert alias.metadata["unit"] == alias.metadata["unit_contract"]["outputs"][""]["unit"] == "MeV·fm"
    with pytest.raises(ValueError, match="conflicts"):
        constant_decl("demo", "mass", 125.0, unit="TeV", unit_contract=contract)
    assert "unit_contract" not in constant_decl("demo", "unlabelled", 7.0).metadata


@pytest.mark.parametrize("name,path", [("helper.eggbox", ""), ("helper.eggbox2d", "/z")])
def test_helpers_normalize_degrees_and_keep_broadcast_results(builtins, name, path) -> None:
    degrees = np.array([0.0, 90.0])
    original = degrees.copy()
    expected = builtins.call(name, x=np.deg2rad(degrees), y=0.0)
    result = builtins.call_units(name, x=degrees, y=0.0, input_units={"x": "deg", "y": "rad"})
    expected_value = expected if path == "" else expected["z"]
    assert np.allclose(result.values[path], expected_value)
    assert result.units[path] == "1"
    assert np.array_equal(degrees, original)
    assert np.allclose(result.normalized_inputs["x"], np.deg2rad(degrees))
    with pytest.raises(UnitContractError) as exc:
        builtins.prepare(name, input_units={"x": "1"})
    assert exc.value.code == "dimension_mismatch"


def test_helper_default_unit_audit_and_pandas_index_survive(builtins) -> None:
    x = pd.Series([0.0, 0.5], index=["first", "second"])
    result = builtins.call_units("helper.eggbox", x=x, y=0.0)
    assert isinstance(result.canonical_result, pd.Series)
    assert result.canonical_result.index.equals(x.index)
    assert np.allclose(result.values[""], [32.0, 93.70340035799192])
    assert all(item["origin"] == "assumed_canonical" for item in result.audit if item["kind"] == "input")
    async_result = asyncio.run(builtins.acall_units("helper.eggbox", x=x, y=0.0))
    assert isinstance(async_result.canonical_result, pd.Series)
    assert async_result.canonical_result.equals(result.canonical_result)


def test_math_remains_polymorphic_and_chi_square_only_declares_output(builtins) -> None:
    arbitrary = {"payload": object()}
    assert builtins.call("math.identity", x=arbitrary) is arbitrary
    assert builtins.call("math.add", a=[1, 2], b=[3]) == [1, 2, 3]
    for name, units in [("math.identity", {"x": "GeV"}), ("math.add", {"a": "GeV"})]:
        assert "unit_contract" not in builtins.get(name).metadata
        with pytest.raises(UnitContractError) as exc:
            builtins.prepare(name, input_units=units)
        assert exc.value.code == "contract_missing"
    residual = pd.DataFrame({"x": [1.0, 2.0], "y": [0.0, 1.0]}, index=["s1", "s2"])
    result = builtins.call_units("stat.chi2_cov", residual=residual, cov=[[2.0, 0.0], [0.0, 1.0]])
    assert isinstance(result.canonical_result, pd.Series)
    assert result.canonical_result.to_dict() == {"s1": 0.5, "s2": 3.0}
    assert result.units[""] == "1"
    with pytest.raises(UnitContractError) as exc:
        builtins.prepare("stat.chi2_cov", input_units={"residual": "GeV"})
    assert exc.value.code == "contract_missing"


def test_cmb_spectrum_contract_uses_verified_microkelvin_squared(builtins) -> None:
    model = _load_internal_planck_tt_cl() * 1.01
    expected = builtins.call("cmb.loglike", model_cl=model, noise_cl=None)
    result = builtins.call_units(
        "cmb.loglike", model_cl=model * 1e-12, noise_cl=None,
        input_units={"model_cl": "K^2"},
    )
    assert result.canonical_result["loglike"] == pytest.approx(expected["loglike"])
    assert result.values["/loglike"] == pytest.approx(expected["loglike"])
    assert result.units["/loglike"] == "1"
    with pytest.raises(UnitContractError) as exc:
        builtins.prepare("cmb.loglike", input_units={"payload": "uK^2"})
    assert exc.value.code == "contract_missing"


def test_cmb_cosmological_kwargs_are_dimensionless_and_output_grid_is_fixed(builtins) -> None:
    plan = builtins.prepare("cmb.tt_cosmopower", input_units={"h": "1"}, output_units={"/model_cl": "K^2"})
    _, kwargs, audit = plan.normalize(h=0.67)
    assert kwargs["h"] == 0.67
    assert next(item for item in audit if item["field"] == "h")["origin"] == "declared"
    canonical = {"model_cl": np.ones(111)}
    projected = plan.project(canonical)
    assert np.allclose(projected.values["/model_cl"], np.full(111, 1e-12), rtol=0, atol=1e-25)
    assert np.array_equal(canonical["model_cl"], np.ones(111))


def test_cmb_nested_payload_cannot_bypass_or_shadow_unit_bound_inputs(builtins) -> None:
    model = _load_internal_planck_tt_cl() * 1.01
    legacy = builtins.call("cmb.loglike", payload={"model_cl": model})
    direct = builtins.call("cmb.loglike", model_cl=model)
    assert legacy == direct
    for kwargs in (
        {"payload": {"model_cl": model * 2}, "model_cl": model * 1e-12,
         "input_units": {"model_cl": "K^2"}},
        {"payload": {"model_cl": model}, "strict": True},
        {"observables": {"model_cl": model}, "strict": True},
        {"payload": {"model": model * 2}, "model_cl": model * 1e-12,
         "input_units": {"model_cl": "K^2"}},
        {"payload": {}, "model_cl": model * 1e-12,
         "input_units": {"model_cl": "K^2"}},
    ):
        with pytest.raises(UnitContractError) as error:
            builtins.call_units("cmb.loglike", **kwargs)
        assert error.value.code == "contract_missing"


def test_all_physical_curves_have_explicit_manifest_units_and_conversion(builtins) -> None:
    expected_counts = {"dmddxn": 27, "dmddxe": 20}
    for namespace, count in expected_counts.items():
        names = builtins.list(namespace=namespace)
        assert len(names) == count
        expected_input = "GeV" if namespace == "dmddxn" else "MeV"
        for name in names:
            declaration = builtins.get(name)
            contract = declaration.metadata["unit_contract"]
            assert contract["inputs"]["x"]["unit"] == expected_input
            assert contract["outputs"][""]["unit"] == "cm^2"
            x = np.sqrt(declaration.metadata["x_min"] * declaration.metadata["x_max"])
            converted_x = x / 1000.0
            source = "TeV" if namespace == "dmddxn" else "GeV"
            expected = builtins.call(name, x=x)
            result = builtins.call_units(name, x=converted_x, input_units={"x": source}, output_units={"": "pb"})
            assert result.canonical_result == pytest.approx(expected)
            assert result.values[""] == pytest.approx(expected * 1e36, rel=1e-12, abs=0)
            assert result.units[""] == "pb"
    for name in builtins.list(namespace="interp1"):
        assert "unit_contract" not in builtins.get(name).metadata


def test_hot_curve_registration_retains_manifest_contracts(tmp_path, builtins) -> None:
    summary = init_builtin_curve_cache(cache_root=str(tmp_path / "curves"), namespaces=["dmddxn", "dmddxe"])
    hot_registry = OperasRegistry()
    registered = register_hot_curves_in_registry(hot_registry, index_path=summary["index_path"])
    assert len(registered) == 47
    for name in registered:
        assert hot_registry.get(name).metadata["unit_contract"] == builtins.get(name).metadata["unit_contract"]
    result = hot_registry.call_units("dmddxn.LZSI2024", x=0.03, input_units={"x": "TeV"}, output_units={"": "pb"})
    assert result.canonical_result == pytest.approx(builtins.call("dmddxn.LZSI2024", x=30.0))
    assert result.values[""] == pytest.approx(result.canonical_result * 1e36, rel=1e-12, abs=0)
