"""Verify the complete installed Operas wheel from an isolated Python outside checkout.

Usage: /path/to/venv/bin/python -I /path/to/scripts/smoke_install.py X.Y.Z
All bootstrap caches and persistence paths are confined to a temporary directory.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def _verify_install(expected: str, base: Path) -> dict[str, object]:
    import numpy as np
    import jarvis_operas
    from jarvis_operas import Operas, OperasRegistry, UnitContractError, oper

    assert metadata.version("Jarvis-Operas") == expected, "Installed package version disagrees"
    package_path = Path(jarvis_operas.__file__).resolve()
    assert package_path.is_relative_to(Path(sys.prefix).resolve()), package_path
    assert not package_path.is_relative_to(Path(__file__).resolve().parents[1]), package_path

    registry = jarvis_operas.get_global_operas_registry()
    assert registry.call("math.add", a=1.0, b=2.0) == 3.0
    assert registry.call("math.identity", x=2.5) == 2.5
    residual = np.array([[1.0, 0.0], [2.0, 1.0]])
    chi2 = registry.call_units("stat.chi2_cov", residual=residual, cov=[[2.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(chi2.values[""], [0.5, 3.0])
    assert chi2.units[""] == "1"

    for name, path in (("helper.eggbox", ""), ("helper.eggbox2d", "/z")):
        canonical = registry.call(name, x=np.pi / 2, y=0.0)
        result = registry.call_units(name, x=90.0, y=0.0, input_units={"x": "deg", "y": "rad"})
        assert math.isclose(result.values[path], canonical if path == "" else canonical["z"], rel_tol=1e-12)
        assert result.units[path] == "1"
        default = registry.call_units(name, x=0.0, y=0.0)
        assert any(row["origin"] == "assumed_canonical" for row in default.audit)

    constants = {
        "mZ": "GeV", "mW": "GeV", "mt": "GeV", "mh": "GeV", "me": "GeV",
        "mmu": "GeV", "mtau": "GeV", "alphaEM": "1", "alphaSMZ": "1",
        "GF": "GeV^-2", "hbarc": "MeV·fm", "c": "m/s",
    }
    for name, unit in constants.items():
        result = registry.call_units("pdg." + name)
        assert result.units[""] == unit
        assert result.canonical_result == registry.call("pdg." + name)
    mass = registry.call_units("pdg.mZ", output_units={"": "MeV"})
    assert mass.canonical_result == 91.1876 and math.isclose(mass.values[""], 91187.6)
    gf = registry.call_units("pdg.GF", output_units={"": "MeV^-2"})
    assert math.isclose(gf.values[""], 1.1663787e-11, rel_tol=1e-12)

    params = {"omegabh2": 0.022, "omegach2": 0.12, "tau": 0.055, "ns": 0.965, "As": 3.0, "h": 0.67}
    spectrum = registry.call_units("cmb.tt_cosmopower", input_units={name: "1" for name in params},
                                   output_units={"/model_cl": "K^2"}, strict=True, **params)
    canonical_spectrum = np.asarray(spectrum.canonical_result["model_cl"])
    assert canonical_spectrum.shape == (111,) and np.all(canonical_spectrum > 0)
    np.testing.assert_allclose(spectrum.values["/model_cl"], canonical_spectrum * 1e-12, rtol=1e-12)
    likelihood = registry.call_units("cmb.loglike", model_cl=spectrum.values["/model_cl"],
                                    input_units={"model_cl": "K^2"}, strict=True)
    raw_likelihood = registry.call("cmb.loglike", model_cl=canonical_spectrum)
    assert math.isclose(likelihood.values["/loglike"], raw_likelihood["loglike"], rel_tol=1e-10)
    assert likelihood.units["/loglike"] == "1" and likelihood.canonical_result["n_ell"] == 111

    curve_counts = {"dmddxn": 27, "dmddxe": 20}
    for namespace, count in curve_counts.items():
        names = registry.list(namespace=namespace)
        assert len(names) == count, (namespace, names)
        canonical_unit = "GeV" if namespace == "dmddxn" else "MeV"
        source_unit = "TeV" if namespace == "dmddxn" else "GeV"
        for name in names:
            declaration = registry.get(name)
            contract = declaration.metadata["unit_contract"]
            assert contract["inputs"]["x"]["unit"] == canonical_unit
            assert contract["outputs"][""]["unit"] == "cm^2"
            x = math.sqrt(declaration.metadata["x_min"] * declaration.metadata["x_max"])
            canonical = registry.call(name, x=x)
            result = registry.call_units(name, x=x / 1000, input_units={"x": source_unit},
                                         output_units={"": "pb"}, strict=True)
            assert math.isfinite(canonical) and canonical > 0, name
            assert math.isclose(result.canonical_result, canonical, rel_tol=1e-12), name
            assert math.isclose(result.values[""], canonical * 1e36, rel_tol=1e-12), name

    fixture_registry = OperasRegistry(numpy_concurrency=2)
    calls = []
    record = {"value": 0.038332, "unit": "pb", "record_id": "release-canonical-pb",
              "provenance": {"source": "release-smoke"}, "dependencies": ["mass-record"], "status": "ok"}
    original = copy.deepcopy(record)
    contract = {"version": "1.0", "inputs": {"mass": {"unit": "GeV", "shape": "scalar"}},
                "outputs": {"/value": {"unit": "pb", "shape": "scalar"}}}

    @oper("limit", namespace="release_smoke", registry=fixture_registry, unit_contract=contract)
    def limit(mass):
        calls.append(mass)
        return record

    @oper("alimit", namespace="release_smoke", registry=fixture_registry, unit_contract=contract)
    async def alimit(mass):
        await asyncio.sleep(0)
        return limit(mass)

    options = {"input_units": {"mass": "TeV"}, "source_units": {"mass": "TeV"},
               "output_units": {"limit": "fb"}, "output_bindings": {"limit": "/value", "status": "/status"},
               "strict": True}
    result = fixture_registry.call_units("release_smoke.limit", mass=0.2, **options)
    assert result.normalized_inputs["mass"] == 200
    assert math.isclose(result.values["limit"], 38.332)
    assert result.values["status"] == "ok" and result.units["limit"] == "fb"
    assert result.canonical_result is record and record == original
    count = len(calls)
    for failure_options, code in (
        ({"strict": True}, "missing_unit"),
        ({"input_units": {"mass": "pb"}}, "dimension_mismatch"),
        ({"input_units": {"mass": "GeV"}, "source_units": {"mass": "TeV"}}, "source_conflict"),
    ):
        try:
            fixture_registry.call_units("release_smoke.limit", mass=0.2, **failure_options)
        except UnitContractError as failure:
            assert failure.code == code, failure
        else:
            raise AssertionError(f"Expected unit failure: {code}")
        assert len(calls) == count, "Invalid units executed the physical function"

    plan = fixture_registry.prepare("release_smoke.limit", **options)
    assert plan.run(mass=0.2).values == result.values
    assert plan.run_inputs((), {"mass": 0.2}, source_units={"mass": "TeV"}).values == result.values

    async def asynchronous():
        wrapper = Operas(fixture_registry)
        one = await wrapper.acall_units("release_smoke.alimit", mass=0.2, **options)
        prepared = await plan.arun(mass=0.2)
        batch = await wrapper.acall_many_units("release_smoke.alimit", [{"mass": 0.2}, {"mass": 0.125}],
                                               concurrency=2, **options)
        assert one.values == prepared.values == batch[0].values == result.values
        assert batch[1].normalized_inputs["mass"] == 125
        assert batch[0].audit is not batch[1].audit
        assert all(item.canonical_result is record for item in batch)

    asyncio.run(asynchronous())
    assert record == original

    executable = Path(sys.prefix) / ("Scripts/jopera.exe" if os.name == "nt" else "bin/jopera")
    assert executable.is_file(), "Installed wheel is missing the jopera console script"

    def cli(*args):
        process = subprocess.run([str(executable), *args], cwd=base, env=os.environ.copy(),
                                 capture_output=True, text=True, timeout=90)
        assert process.returncode == 0, process.stdout + process.stderr
        return process.stdout

    assert "jopera" in cli("--help")
    assert expected in cli("--version")
    call_help = cli("call", "--help")
    assert all(option in call_help for option in ("--input-units", "--source-units", "--output-units", "--strict-units", "--unit-audit"))
    info = json.loads(cli("info", "helper.eggbox", "--json"))
    assert info["metadata"]["unit_contract"]["inputs"]["x"]["unit"] == "rad"
    projected = json.loads(cli("call", "pdg.mZ", "--output-units", '{"":"MeV"}', "--json"))
    assert projected["canonical_result"] == 91.1876
    assert math.isclose(projected["values"][""], 91187.6)
    assert projected["units"][""] == "MeV"
    return {"version": expected, "installed_smoke": "passed", "package_path": str(package_path),
            "physical_unit_contracts": 64, "curve_counts": curve_counts,
            "checks": ["installed CLI", "math/stat/helpers/PDG", "bundled CMB model and likelihood",
                       "47 packaged curve contracts", "public sync/async/batch unit API",
                       "strict/conflict/dimension rejection before execution", "canonical record protection"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected_version")
    args = parser.parse_args(argv)
    assert sys.flags.isolated, "Run the installed smoke with python -I"
    assert not Path.cwd().resolve().is_relative_to(Path(__file__).resolve().parents[1]), "Run outside the source checkout"
    previous = os.environ.copy()
    try:
        with tempfile.TemporaryDirectory(prefix="jarvis-operas-installed-smoke-") as directory:
            base = Path(directory)
            for key in ("PYTHONPATH", "PYTHONHOME", "JARVIS_OPERAS_CURVE_INDEX", "JARVIS_OPERAS_DEV_WRITE"):
                os.environ.pop(key, None)
            os.environ.update({
                "JARVIS_OPERAS_CURVE_CACHE_ROOT": str(base / "curves"),
                "JARVIS_OPERAS_PERSIST_FILE": str(base / "persist.json"),
                "JARVIS_OPERAS_SOURCES_FILE": str(base / "sources.json"),
                "JARVIS_OPERAS_OVERRIDES_FILE": str(base / "overrides.json"),
                "JARVIS_OPERAS_USER_SOURCE_DIR": str(base / "user_sources"),
                "JARVIS_OPERAS_LOG_MODE": "warning",
            })
            report = _verify_install(args.expected_version, base)
    finally:
        os.environ.clear()
        os.environ.update(previous)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
